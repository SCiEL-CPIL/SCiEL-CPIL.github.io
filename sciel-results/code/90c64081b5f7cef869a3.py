"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

_START = time.time()
_LIMIT = 110.0  # Safety margin under 120s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    # Reproduce verifier's random_graph exactly
    rng = random.Random(seed)
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
    
    # Build adjacency sets for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Track which edges are covered
    # Using a set of frozensets or tuples for O(1) lookup
    covered_edges = set()
    for u, v in edges:
        covered_edges.add((min(u, v), max(u, v)))
        
    cliques = []
    # We process edges in a specific order. 
    # Strategy: Try to form large cliques.
    # We will iterate through all possible edges. If an edge is not covered, 
    # we try to extend it to a maximal clique within the graph induced by uncovered edges.
    
    # To make this efficient, we can just iterate through the list of edges.
    # If an edge (u, v) is not covered, we attempt to build a clique starting with {u, v}.
    # We want this clique to be as large as possible to cover more edges at once.
    # Greedy extension: 
    # Current clique C = {u, v}
    # Candidates = intersection of neighbors of all nodes in C, restricted to nodes w 
    # such that all edges from w to C are currently UNCOVERED.
    
    # Optimization: Instead of checking "uncovered" dynamically which is expensive,
    # we can just find the maximal clique in the subgraph of *remaining* edges.
    # But finding maximal clique is hard. 
    # Heuristic: 
    # 1. Start with edge (u,v).
    # 2. Find common neighbors of u and v.
    # 3. Filter common neighbors to those that have uncovered edges to u and v.
    # 4. Greedily add the neighbor with the highest degree in the remaining graph? 
    #    Or just the first one?
    #    Let's try: Add the candidate that shares the most uncovered edges with current clique members.
    
    # To ensure we don't miss edges, we iterate through the original edge list.
    # If edge is covered, skip.
    # If not, build a clique.
    
    # This might be slow if we do it for every edge. 
    # However, n is small (32, 64).
    # Number of edges is ~ n^2/2. 
    # If we cover ~5 edges per clique, we do ~ n^2/10 iterations.
    # Each iteration involves set intersections of size <= n.
    # This should be fast enough.
    
    for u, v in edges:
        if not _deadline_ok():
            break
        key = (min(u, v), max(u, v))
        if key in covered_edges:
            continue
            
        # Start clique with u, v
        clique = {u, v}
        # Mark these edges as covered immediately? No, mark at the end to allow 
        # checking "uncovered" status for extension candidates.
        # But we must ensure we don't pick a candidate w if edge (w, u) or (w, v) is already covered.
        
        # Candidates for next vertex:
        # Must be adjacent to all in clique.
        # Must have uncovered edges to all in clique.
        
        candidates = adj[u] & adj[v]
        # Filter candidates: must have uncovered edges to u and v
        # Since (u,v) is uncovered, we check w against u and v.
        valid_candidates = set()
        for w in candidates:
            # Check if (w, u) and (w, v) are uncovered
            key_wu = (min(w, u), max(w, u))
            key_wv = (min(w, v), max(w, v))
            if key_wu not in covered_edges and key_wv not in covered_edges:
                valid_candidates.add(w)
                
        # Greedily expand
        while valid_candidates:
            # Pick the candidate that maximizes the number of *new* uncovered edges covered?
            # Or simply pick the one with highest degree in the remaining graph?
            # Let's pick the one that has the most common neighbors in valid_candidates 
            # that are also adjacent to all current clique members.
            # Actually, simpler: just pick the first one? No, that's bad.
            # Let's pick the one with the highest "potential" to grow.
            # Score: number of neighbors in valid_candidates that are adjacent to all clique members.
            
            best_w = None
            best_score = -1
            
            # Optimization: If we have many candidates, this loop is O(|C| * |Cand| * |Cand|)
            # n=64, so max 64 candidates. 64*64*64 is ~250k ops. Fine.
            
            for w in valid_candidates:
                # Check if w is adjacent to all in clique
                # We already know w is adjacent to u and v (initial step) 
                # But after adding more nodes, we must check against ALL clique members.
                # Wait, `valid_candidates` was computed based on u and v only.
                # If we add w to clique, the new candidates must be adjacent to w too.
                
                # Let's refine the loop structure.
                # We maintain `current_candidates` which are vertices adjacent to ALL current clique members
                # and having uncovered edges to ALL current clique members.
                
                break # Exit inner loop to restructure
                
            # Restructured logic:
            # 1. Initialize clique = {u, v}
            # 2. current_candidates = {w | w adj to all in clique AND edges(w, x) uncovered for all x in clique}
            # 3. While current_candidates is not empty:
            #      Pick w from current_candidates that maximizes |neighbors(w) in current_candidates| 
            #      (This tends to pick central nodes in the remaining graph)
            #      Add w to clique
            #      Update current_candidates: 
            #         new_candidates = {x in current_candidates | x adj to w AND edges(x, w) uncovered}
            #         # Note: we don't need to check x adj to old clique members because 
            #         # they were already in current_candidates.
            #      Set current_candidates = new_candidates
            # 4. Mark all edges in the final clique as covered.
            
            break # Stop previous attempt
            
        # Restart the "while valid_candidates" logic properly inside the for loop
        
        # Re-initialize for this specific edge (u,v)
        clique = {u, v}
        # Candidates must be adjacent to u and v
        cand_set = adj[u] & adj[v]
        
        # Filter for uncovered edges to u and v
        current_cands = set()
        for w in cand_set:
            if (min(w, u), max(w, u)) not in covered_edges and (min(w, v), max(w, v)) not in covered_edges:
                current_cands.add(w)
                
        # Greedy Expansion
        while current_cands:
            # Select w from current_cands
            # Heuristic: Pick w that has the most neighbors in current_cands
            # This promotes forming large cliques.
            best_w = None
            max_common = -1
            for w in current_cands:
                # Count neighbors of w in current_cands
                # adj[w] is a set of all neighbors. 
                # We need intersection with current_cands.
                common_count = len(adj[w] & current_cands)
                if common_count > max_common:
                    max_common = common_count
                    best_w = w
            
            if best_w is None:
                break
                
            # Add best_w to clique
            clique.add(best_w)
            
            # Update current_cands
            # New candidates must be:
            # 1. In current_cands
            # 2. Adjacent to best_w
            # 3. Have uncovered edge to best_w
            # (They are already adjacent to all previous clique members by definition of being in current_cands)
            
            new_cands = set()
            for x in current_cands:
                if x == best_w:
                    continue
                if best_w in adj[x]:
                    key_xw = (min(x, best_w), max(x, best_w))
                    if key_xw not in covered_edges:
                        new_cands.add(x)
                        
            current_cands = new_cands
            
            # Check deadline
            if not _deadline_ok():
                break
        
        # Mark all edges in the clique as covered
        for x, y in itertools.combinations(clique, 2):
            key = (min(x, y), max(x, y))
            covered_edges.add(key)
            
        # Add clique to result (only if size >= 2, which is always true here)
        if len(clique) >= 2:
            cliques.append(list(clique))
            
    return cliques

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if not _deadline_ok():
            # If we run out of time, return what we have or empty?
            # The task requires valid partitions. If we don't finish, we should probably
            # fall back to a safe, albeit suboptimal, solution.
            # However, the loop is designed to be fast. 
            # If deadline is hit, we can't guarantee validity if we stop mid-way.
            # So we should only return if we completed the instance.
            # For safety, if we hit deadline, we will just return the single-edge partition for remaining instances.
            pass
        
        cliques = _solve_instance(n, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
