"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # safety margin under 150s wall-clock

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

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency matrix for fast lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Map edge to index for quick lookup
    edge_set = set(edges)
    
    # Initial Solution: Greedy Peeling to get a decent starting point
    # We will then improve it with Simulated Annealing
    
    current_edges = set(edges)
    initial_cliques = []
    
    # Greedy initial partition
    while current_edges:
        if not _deadline_ok():
            # Fallback
            for e in current_edges:
                initial_cliques.append([e[0], e[1]])
            break
            
        # Pick a random edge to start a clique
        # Using a fixed random seed for the instance logic to be deterministic?
        # The task allows non-determinism, but let's use a local RNG seeded by instance seed for reproducibility
        # Actually, let's just use the global random module but seed it once per instance?
        # No, global random is fine if we don't rely on it for correctness, just quality.
        # Let's use a local RNG for the heuristic to ensure same input -> same output if needed, 
        # but SA is stochastic. Let's just use a local RNG seeded by 'seed'.
        
        # Re-initialize local RNG for the greedy phase? 
        # Let's just use the global random, but for the sake of "deterministic" behavior in tests,
        # it's often better to seed locally.
        
        # Let's pick the edge with the highest "potential" (sum of degrees of endpoints in remaining graph)
        # Or just min for determinism. Let's use min for the initial greedy to be stable.
        u, v = min(current_edges)
        
        # Expand to a maximal clique
        clique = [u, v]
        candidates = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
        
        while candidates:
            # Pick candidate with highest degree in the candidate subgraph
            best_v = -1
            best_deg = -1
            for w in candidates:
                deg = 0
                for x in candidates:
                    if w == x: continue
                    if adj[w][x]:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_v = w
            
            if best_v == -1:
                break
                
            clique.append(best_v)
            candidates.remove(best_v)
            
            # Filter candidates: must be connected to best_v
            new_candidates = []
            for w in candidates:
                if adj[w][best_v]:
                    new_candidates.append(w)
            candidates = new_candidates
            
        initial_cliques.append(clique)
        
        # Remove edges
        for a, b in combinations(clique, 2):
            if (a, b) in current_edges:
                current_edges.remove((a, b))
            elif (b, a) in current_edges:
                current_edges.remove((b, a))
                
    # If we ran out of time during greedy, return what we have (plus remaining as edges)
    if not _deadline_ok():
        for e in current_edges:
            initial_cliques.append([e[0], e[1]])
        return initial_cliques

    # Now, Simulated Annealing to improve the partition
    # Representation: List of sets of vertices. 
    # But we need to ensure every edge is covered exactly once.
    # This is tricky to represent directly as "cliques" because a vertex can be in multiple cliques 
    # (e.g., edge (u,v) in C1, edge (u,w) in C2).
    # However, the constraint is on EDGES.
    
    # Alternative Representation for SA:
    # Let's stick to the "Partition of Edges" view? 
    # No, the output is "List of Cliques". 
    # A clique is defined by its vertices. The edges it covers are all pairs in those vertices.
    # Constraint: The union of all edges covered by the cliques must be exactly the set of graph edges.
    # And no edge can be covered by two different cliques.
    
    # This implies that the cliques must be "edge-disjoint".
    # If two cliques share an edge, that edge is covered twice -> Invalid.
    # If two cliques share a vertex but not an edge, that's fine.
    
    # So the state is a list of vertex sets (cliques).
    # Validity: 
    # 1. Each set is a clique in G.
    # 2. For any two distinct cliques C1, C2, E(C1) ∩ E(C2) = ∅.
    # 3. Union of E(Ci) = E(G).
    
    # Moves:
    # 1. Move an edge (u,v) from one clique to another?
    #    If we move edge (u,v) from C1 to C2, C1 must remain a clique (it will, since we remove edges, 
    #    but wait: if C1 was {u,v,w}, removing edge (u,v) leaves {u,w} and {v,w} as edges? 
    #    No, the clique is defined by vertices. If C1 = {u,v,w}, it covers (u,v), (u,w), (v,w).
    #    If we "remove" the edge (u,v) from C1, C1 is no longer a clique covering that edge.
    #    But the vertex set {u,v,w} still implies the edge (u,v) exists in the graph.
    #    The constraint is that the *listed set* must be a clique, and *every edge* must lie in exactly one listed clique.
    #    This means if we list {u,v,w}, we are claiming that edges (u,v), (u,w), (v,w) are covered by this clique.
    #    We cannot "remove" just one edge from a listed clique. We must change the clique's vertex set.
    
    # So a move must involve changing the vertex set of a clique.
    # Example: Change C1 from {u,v,w} to {u,v}. This removes edge (u,w) and (v,w) from coverage by C1.
    # Those edges must then be covered by other cliques.
    # This suggests a complex state space.
    
    # Simpler approach: 
    # Since n is small (32, 64), maybe we can use a different heuristic.
    # The "Greedy Peeling" is actually quite good.
    # The main weakness is the order of selection.
    # Let's try "Multiple Restart Greedy" with different orderings.
    # 1. Random permutation of edges.
    # 2. Greedy: Pick first edge, expand to max clique, remove.
    # 3. Repeat with different seeds/permutations.
    # 4. Keep the best result.
    
    # This is a "Restart Heuristic" which is structurally different from SA (which is local search on a complex state)
    # and different from single-shot greedy.
    
    best_partition = None
    best_count = float('inf')
    
    # Time budget for restarts
    # Each restart takes O(E * n^2) roughly. 
    # E ~ 1000. n=64. n^2=4096. 1000*4096 = 4M ops. Very fast.
    # We can do many restarts.
    
    rng = random.Random(seed)
    
    # Number of restarts depends on time
    # Let's try 50 restarts?
    # Or run until deadline.
    
    # To make it deterministic and robust, let's do a fixed number of restarts with different random seeds derived from the main seed.
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if not _deadline_ok():
            break
            
        # Create a shuffled list of edges for this restart
        shuffled_edges = edges[:]
        rng.shuffle(shuffled_edges)
        
        current_edges = set(shuffled_edges)
        partition = []
        
        # Greedy Peeling with shuffled edge order
        # We pick the first edge in the shuffled list that is still in current_edges?
        # Or just iterate through the shuffled list and if the edge is in current_edges, process it?
        # The latter is better: it defines the order of consideration.
        
        for u, v in shuffled_edges:
            if not _deadline_ok():
                break
                
            if (u, v) not in current_edges:
                continue
                
            # Expand {u, v} to a maximal clique in the *remaining* graph
            # Note: "Maximal" in the remaining graph means we only add vertices w such that
            # (u,w), (v,w) are in current_edges, and w is connected to all other added vertices.
            
            clique = [u, v]
            
            # Candidates must be connected to u and v in the *current* graph (i.e., edges exist in current_edges)
            candidates = []
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check if (u,w) and (v,w) are in current_edges
                # Note: edges are stored as (min, max) in the set? 
                # _random_edges returns (u,v) with u<v.
                # current_edges contains (min, max).
                
                e_uw = (u, w) if u < w else (w, u)
                e_vw = (v, w) if v < w else (w, v)
                
                if e_uw in current_edges and e_vw in current_edges:
                    candidates.append(w)
                    
            # Greedily add candidates
            # To make it "maximal", we should try to add as many as possible.
            # Simple greedy: pick candidate with highest degree in the subgraph of candidates?
            # Or just pick the first one?
            # Let's pick the one with highest degree in the *original* graph among candidates?
            # Or highest degree in the *candidate* subgraph (edges in current_edges)?
            
            while candidates:
                # Pick best candidate
                best_w = -1
                best_score = -1
                for w in candidates:
                    # Score: number of edges from w to other candidates in current_edges
                    score = 0
                    for x in candidates:
                        if w == x: continue
                        e_wx = (w, x) if w < x else (x, w)
                        if e_wx in current_edges:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w == -1:
                    break
                    
                clique.append(best_w)
                candidates.remove(best_w)
                
                # Filter candidates: must be connected to best_w in current_edges
                new_candidates = []
                for w in candidates:
                    e_wbw = (w, best_w) if w < best_w else (best_w, w)
                    if e_wbw in current_edges:
                        new_candidates.append(w)
                candidates = new_candidates
                
            # Add clique to partition
            partition.append(clique)
            
            # Remove all edges in this clique from current_edges
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                if e in current_edges:
                    current_edges.remove(e)
                    
        # After processing all edges in shuffled order, if current_edges is not empty,
        # it means some edges were not covered? 
        # No, the loop iterates through ALL edges in shuffled_edges.
        # If an edge (u,v) is in current_edges when we reach it in the list, we cover it.
        # If it was removed earlier by a larger clique, we skip it.
        # So at the end, current_edges should be empty.
        
        if current_edges:
            # This should not happen if logic is correct, but just in case
            # Add remaining as single edges
            for e in current_edges:
                partition.append([e[0], e[1]])
                
        count = len(partition)
        if count < best_count:
            best_count = count
            best_partition = partition
            
    if best_partition is None:
        return initial_cliques
        
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
