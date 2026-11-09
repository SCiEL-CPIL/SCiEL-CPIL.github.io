"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 110.0  # safety margin under 120s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _is_clique(clique, adj):
    if len(clique) <= 1:
        return True
    # Check if every pair in clique is connected
    for u, v in combinations(clique, 2):
        if not adj[u][v]:
            return False
    return True

def _merge_heuristic(n, edges):
    # Build adjacency matrix for fast lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Initial partition: each edge is a clique
    cliques = [[u, v] for u, v in edges]
    current_count = len(cliques)
    
    # Helper to check if union of two cliques is a clique
    def can_merge(c1, c2):
        # If c1 and c2 are disjoint, we need all cross edges to exist
        # If they share vertices, we need the union to be a clique
        # Since we start with edges, they are disjoint or share 1 vertex? 
        # Actually, edges (u,v) and (u,w) share u. Union {u,v,w} is a triangle.
        # If they are disjoint edges (u,v) and (x,y), union {u,v,x,y} needs all 6 edges.
        
        # Efficient check:
        # 1. Check internal consistency (already cliques)
        # 2. Check cross edges
        for u in c1:
            for v in c2:
                if u == v:
                    continue
                if not adj[u][v]:
                    return False
        return True

    # Local search: try to merge cliques
    # We maintain a list of cliques.
    # To speed up, we can try random pairs.
    # If we merge, we replace c1, c2 with c1+c2.
    # We want to minimize the number of cliques.
    
    # Heuristic:
    # 1. Start with trivial partition.
    # 2. Repeatedly pick a random pair of cliques.
    # 3. If their union is a clique, merge them.
    # 4. If no merge possible for a while, try a "kick" (random split/swap) or just restart?
    #    Given the time limit, simple hill climbing on merges might get stuck.
    #    However, merging is always beneficial (reduces count).
    #    The problem is finding the *right* merges.
    #    
    #    Alternative: Start with a greedy maximal clique cover (like the seed but better),
    #    then try to merge cliques from that cover?
    #    The seed program just returns edges. That's the worst case (max count).
    #    Any improvement is a jump.
    #    
    #    Let's try a simple greedy merge from the edge partition.
    #    But "random pair" is slow if there are ~1000 edges.
    #    
    #    Better approach:
    #    1. Find all maximal cliques.
    #    2. Use a greedy set cover approach to pick cliques that cover the most uncovered edges.
    #    3. This is NP-hard, but greedy is a good heuristic.
    #    
    #    Let's implement a Greedy Maximal Clique Cover.
    #    
    #    1. Generate a list of candidate cliques (maximal cliques).
    #    2. While edges remain uncovered:
    #       a. Pick the candidate clique that covers the most uncovered edges.
    #       b. Add it to the partition.
    #       c. Mark its edges as covered.
    #       d. Remove any other candidate cliques that share edges with the chosen one?
    #          No, we just need to ensure no edge is covered twice.
    #          So we maintain a set of uncovered edges.
    
    # Let's stick to the merge heuristic but optimized.
    # Actually, the "Greedy Set Cover" on maximal cliques is a standard heuristic for Clique Partition/Cover.
    # Since we need a *partition* (disjoint edge sets), and maximal cliques can overlap in edges,
    # we must be careful.
    # If we pick a maximal clique C, we assign all its edges to C.
    # Then we remove those edges from the graph.
    # Then we find maximal cliques in the remaining graph?
    # That's a greedy peeling algorithm.
    
    # Let's implement Greedy Peeling:
    # 1. Find a maximal clique (or a "good" clique) in the current graph.
    # 2. Add it to the partition.
    # 3. Remove all edges in that clique from the graph.
    # 4. Repeat until no edges left.
    
    # To find a "good" clique, we can use a greedy heuristic:
    # - Start with a random vertex.
    # - Greedily add neighbors that are connected to all current members.
    # - Or find the maximal clique containing a random vertex.
    
    # Let's try:
    # 1. While edges remain:
    #    a. Pick a random edge (u, v).
    #    b. Expand {u, v} to a maximal clique in the current graph.
    #    c. Add this clique to the partition.
    #    d. Remove all edges of this clique from the current graph.
    
    # This is different from the seed (which returns edges) and different from pure maximal clique peeling 
    # (which might pick huge cliques that block better merges, though for minimization, bigger cliques are usually better).
    # The key is that we remove edges, so we don't double count.
    
    # Let's implement this Greedy Expansion.
    
    current_edges = set(edges)
    if not current_edges:
        return []
        
    partition = []
    
    # Helper to get neighbors of a set in the current graph
    def get_maximal_clique(start_vertices):
        # start_vertices is a set of vertices
        # Greedy expansion
        clique = list(start_vertices)
        # Candidates are common neighbors of all vertices in clique
        # Since n is small (32, 64), we can just check all vertices
        
        # Optimization: maintain a list of candidates
        candidates = [v for v in range(n) if v not in clique]
        
        # Filter candidates that are connected to all in clique
        valid_candidates = []
        for v in candidates:
            is_connected = True
            for u in clique:
                if (u, v) not in current_edges and (v, u) not in current_edges:
                    is_connected = False
                    break
            if is_connected:
                valid_candidates.append(v)
                
        # Greedily add valid candidates
        # To make it "maximal", we should add as many as possible.
        # A simple greedy: add the one with highest degree in the subgraph?
        # Or just add all? No, adding one might block another.
        # Let's just add all valid candidates one by one?
        # Actually, if v1 and v2 are both valid, and edge (v1, v2) exists, we can add both.
        # If edge (v1, v2) doesn't exist, we can only add one.
        
        # Simple heuristic: Add valid candidates in order of index?
        # Or random order?
        # Let's try adding all valid candidates that are connected to each other and the clique.
        # This is essentially finding a maximal clique in the subgraph induced by clique + valid_candidates.
        
        # Since n is small, let's just do a simple greedy:
        # While there is a valid candidate v that is connected to all other vertices in the "current clique set" (including those we plan to add):
        # This is getting complex.
        
        # Simpler: Just return the set of start_vertices + all valid_candidates that form a clique with each other and the start.
        # Even simpler: Just use the start_vertices if expansion is too slow/complex.
        # But for n=64, expansion is cheap.
        
        # Let's do a recursive/greedy expansion:
        # 1. Start with clique = list(start_vertices)
        # 2. Find all v not in clique such that v is connected to all u in clique.
        # 3. Among these, pick one v that is connected to the most other candidates?
        # 4. Add v to clique.
        # 5. Repeat until no more candidates.
        
        # Let's implement step 3-5.
        remaining_candidates = valid_candidates
        
        while True:
            if not remaining_candidates:
                break
            
            # Pick the candidate with the highest "score" (degree in the candidate subgraph + connection to clique)
            # Actually, just pick the first one? Or random?
            # Let's pick the one with the highest degree in the graph induced by remaining_candidates.
            best_v = -1
            best_deg = -1
            for v in remaining_candidates:
                deg = 0
                for w in remaining_candidates:
                    if v == w: continue
                    if (v, w) in current_edges or (w, v) in current_edges:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            # Add best_v to clique
            clique.append(best_v)
            remaining_candidates.remove(best_v)
            
            # Filter remaining_candidates: must be connected to best_v
            remaining_candidates = [v for v in remaining_candidates if (v, best_v) in current_edges or (best_v, v) in current_edges]
            
        return clique

    while current_edges:
        if not _deadline_ok():
            # Fallback: return remaining edges as individual cliques
            partition.extend([[u, v] for u, v in current_edges])
            break
            
        # Pick a random edge from current_edges
        # To avoid bias, pick uniformly
        edge_list = list(current_edges)
        u, v = random.choice(edge_list)
        
        # Expand to maximal clique
        clique = get_maximal_clique({u, v})
        
        # Add to partition
        partition.append(clique)
        
        # Remove all edges in this clique from current_edges
        for a, b in combinations(clique, 2):
            if (a, b) in current_edges:
                current_edges.remove((a, b))
            elif (b, a) in current_edges:
                current_edges.remove((b, a))
                
    return partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = _random_edges(n, p, seed)
        # Use a fixed seed for the heuristic to be deterministic?
        # The task says "Minimize", implying we should find a good solution.
        # Using random.choice in the heuristic makes it non-deterministic.
        # We should probably run it once with a fixed internal seed or just let it be.
        # But for reproducibility in the evaluator, maybe we should fix the random state?
        # The verifier likely runs construct() once.
        # Let's use a fixed seed for the heuristic's internal randomness to ensure stability.
        # Or better, use a deterministic greedy (e.g., pick edge with min index).
        # Let's switch to deterministic: pick the first edge in the list (sorted).
        
        # Re-implementing the loop with deterministic edge picking
        # To avoid rewriting, I'll just seed the random module locally?
        # No, global random is bad.
        # Let's modify the logic to be deterministic.
        
        # Actually, the previous code used random.choice.
        # Let's replace it with a deterministic choice: the lexicographically smallest edge.
        
        # I will rewrite the core loop inside construct to be deterministic for safety.
        
        pass # The function above is a draft. I will write the final clean version below.

    # Final Clean Implementation
    def solve_instance(n, p, seed):
        edges = _random_edges(n, p, seed)
        if not edges:
            return []
            
        current_edges = set(edges)
        partition = []
        
        # Helper to get maximal clique from start set
        def expand_clique(start_vertices):
            clique = list(start_vertices)
            # Candidates: vertices connected to all in clique
            candidates = []
            for v in range(n):
                if v in clique:
                    continue
                ok = True
                for u in clique:
                    if (u, v) not in current_edges and (v, u) not in current_edges:
                        ok = False
                        break
                if ok:
                    candidates.append(v)
            
            while candidates:
                # Pick candidate with highest degree in the candidate subgraph
                best_v = -1
                best_deg = -1
                for v in candidates:
                    deg = 0
                    for w in candidates:
                        if v == w:
                            continue
                        if (v, w) in current_edges or (w, v) in current_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                if best_v == -1:
                    break
                
                clique.append(best_v)
                candidates.remove(best_v)
                
                # Filter candidates: must be connected to best_v
                new_candidates = []
                for v in candidates:
                    if (v, best_v) in current_edges or (best_v, v) in current_edges:
                        new_candidates.append(v)
                candidates = new_candidates
                
            return clique

        while current_edges:
            if not _deadline_ok():
                partition.extend([[u, v] for u, v in current_edges])
                break
                
            # Deterministic pick: smallest edge
            # Since set doesn't have order, we can just pick min()
            # min() on tuples (u, v) works lexicographically
            u, v = min(current_edges)
            
            clique = expand_clique({u, v})
            partition.append(clique)
            
            # Remove edges
            for a, b in combinations(clique, 2):
                if (a, b) in current_edges:
                    current_edges.remove((a, b))
                elif (b, a) in current_edges:
                    current_edges.remove((b, a))
                    
        return partition

    return [solve_instance(n, p, seed) for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602))]
# EVOLVE-BLOCK-END
