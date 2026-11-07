"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # 1. Identify Candidate Edges
    # To maintain triangle-freeness (and bipartiteness), we only add edges between opposite parity.
    # Since the base is a path, any such edge creates a cycle of even length.
    # We prioritize "long" edges (large |u-v|) as they provide the most diameter reduction.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 1 and (u, v) not in base_edges:
                candidates.append((u, v))
                
    # Sort candidates by potential impact (descending distance)
    # Edges that skip more nodes are more likely to be part of the optimal solution.
    candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    # 2. Precompute base adjacency for quick updates
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        
    def get_diameter(adj):
        max_dist = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        q.append(v)
            if -1 in dist:
                return float('inf')
        return max_dist

    # 3. Branch and Bound Search
    # We search for the minimum number of edges k such that there exists a subset of size k 
    # that reduces diameter to <= 4.
    # We iterate k from 1 upwards.
    # For each k, we use backtracking to find a valid subset.
    
    # Optimization: We can limit the search space.
    # A path of 24 nodes has diameter 23.
    # Adding an edge (u, v) with distance d reduces the diameter roughly by d/2 in the worst case, 
    # but usually creates a "hub".
    # Let's try to find a solution with small k first.
    
    # Heuristic pruning: 
    # If we pick an edge (u, v), it effectively merges the path segments.
    # We can use a simple bound: if the current best solution size is B, we stop exploring branches 
    # that have already selected >= B edges.
    
    best_solution = None
    best_size = n # Upper bound
    
    # We will search for k=1, 2, 3... up to a reasonable limit.
    # Given n=24, k is likely small (maybe 3-6).
    
    # To make the search efficient, we don't just iterate all combinations of size k (which is C(144, k)).
    # We use recursive backtracking with pruning based on "potential diameter reduction".
    
    # However, a simpler and very effective approach for small n is:
    # 1. Generate a "shortlist" of the top M most impactful edges (e.g., top 30).
    # 2. Perform a complete search over subsets of this shortlist up to size K_max.
    # Why? The optimal solution likely consists of the longest possible chords.
    # Edges with small |u-v| are rarely needed if we have access to large ones.
    
    M = 40 # Shortlist size
    shortlist = candidates[:M]
    
    # Check if a subset of edges achieves diameter <= 4
    def check_subset(edges_subset):
        # Build adjacency
        adj = [set(base_adj[i]) for i in range(n)]
        for u, v in edges_subset:
            adj[u].add(v)
            adj[v].add(u)
        return get_diameter(adj) <= 4

    # Search for minimum k
    # We use a BFS-like or DFS-based combination generation.
    # Since M=40, C(40, 5) is ~658,000. C(40, 6) is ~3.8M. This is feasible in Python if optimized, 
    # but might be tight. 
    # Let's use a recursive generator that prunes early if possible? 
    # Actually, checking diameter is O(N^2) = 576 ops. 
    # 3.8M * 576 is ~2.2B ops. Too slow for Python.
    
    # Strategy Adjustment: 
    # Instead of blind combination, use a Greedy + Local Search or a smarter B&B.
    # Let's stick to the Greedy but improve it with "Look-ahead" or "Re-starting".
    # OR, realize that for a path, the structure is 1D.
    # The diameter is determined by the endpoints of the longest path in the new graph.
    
    # Let's try a different angle: 
    # The problem is equivalent to placing "teleporters".
    # A known result for path augmentation: 
    # To get diameter D, we need to cover the path with segments of length <= D.
    # But edges can connect non-adjacent segments.
    
    # Let's go back to the Greedy but make it smarter:
    # 1. Start with empty set.
    # 2. At each step, pick the edge that minimizes the *maximum* distance among all pairs, 
    #    but with a tie-breaker that maximizes the number of pairs reduced by > 1.
    # 3. Add a "local search" phase: after greedy finishes, try swapping one edge for another 
    #    to see if we can reduce the count (if we allow starting with a slightly worse greedy).
    
    # Actually, the most robust simple method for n=24 is:
    # 1. Find the top K candidates (K=20).
    # 2. Use a recursive search to find the minimum subset size.
    #    To speed up, we can pre-compute the "contribution" of each edge? No, interactions are complex.
    
    # Let's try a hybrid:
    # 1. Greedy to get a good upper bound (current best).
    # 2. Then, try to find a smaller solution by searching subsets of the "promising" edges 
    #    identified by the greedy process + its neighbors.
    
    # Greedy Phase
    adj = [set(base_adj[i]) for i in range(n)]
    current_edges = set()
    current_diam = get_diameter(adj)
    
    # We will run multiple greedy passes with different tie-breaking strategies and keep the best.
    best_greedy_solution = None
    best_greedy_size = float('inf')
    
    for strategy in range(5):
        adj = [set(base_adj[i]) for i in range(n)]
        current_edges = set()
        current_diam = get_diameter(adj)
        
        while current_diam > 4:
            if time.time() > deadline:
                break
                
            best_edge = None
            best_diam = current_diam + 1
            best_tie_breaker = -1 # Number of pairs improved
            
            remaining = [e for e in candidates if e not in current_edges]
            
            # Sort by distance descending
            remaining.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
            
            for u, v in remaining:
                # Simulate
                adj[u].add(v)
                adj[v].add(u)
                
                new_diam = get_diameter(adj)
                
                # Undo
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = (u, v)
                    # Calculate tie breaker: how many pairs got closer?
                    # This is expensive, so we only do it if it's a candidate for best.
                    # For now, just take the first one with min diameter.
                    # To break ties better, we can look at the sum of distances?
                    # Let's just use the edge with max |u-v| among those with min diameter.
                    # Since we iterate sorted by |u-v|, the first one found with min diameter 
                    # is the one with max |u-v| among the min-diameter edges? 
                    # No, we iterate all. So we should track the one with max |u-v| for the same best_diam.
                    if new_diam == best_diam:
                        if best_edge is None or abs(u-v) > abs(best_edge[0]-best_edge[1]):
                            best_edge = (u, v)
                    else:
                        best_edge = (u, v)
                            
                if new_diam <= 4:
                    # We can't get better than 4 in terms of validity, but maybe a different edge 
                    # allows fewer total edges later? Unlikely to matter for the immediate step.
                    # But breaking early might miss a "better" edge for future steps.
                    # Let's not break, let's find the BEST edge for this step.
                    pass
            
            if best_edge is None:
                break
                
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            current_edges.add((u, v))
            current_diam = best_diam
            
            if current_diam <= 4:
                break
                
        if current_diam <= 4:
            if len(current_edges) < best_greedy_size:
                best_greedy_size = len(current_edges)
                best_greedy_solution = list(current_edges)
                
    if best_greedy_solution is None:
        # Fallback: return a valid large solution
        # Just add enough long edges to ensure connectivity and low diameter
        # e.g., connect 0 to 12, 12 to 23, etc.
        # But the greedy should have found something.
        return []

    # Local Search / Refinement
    # Try to remove edges from the greedy solution to see if it's still valid.
    # If valid, we found a smaller solution.
    # Then try swapping?
    
    current_sol = best_greedy_solution
    improved = True
    while improved:
        improved = False
        for i in range(len(current_sol)):
            # Try removing current_sol[i]
            candidate_sol = current_sol[:i] + current_sol[i+1:]
            if check_subset(candidate_sol):
                current_sol = candidate_sol
                improved = True
                break # Restart loop as indices changed
                
        if time.time() > deadline:
            break
            
        # Try swapping one edge with another from the candidate pool
        # Only do this if we have time
        if time.time() < deadline - 10:
            for i in range(len(current_sol)):
                u, v = current_sol[i]
                for alt_u, alt_v in candidates:
                    if (alt_u, alt_v) in current_sol:
                        continue
                    if abs(alt_u - alt_v) < abs(u - v): # Only try similar or better edges
                         continue
                    candidate_sol = current_sol[:i] + [(alt_u, alt_v)] + current_sol[i+1:]
                    if check_subset(candidate_sol):
                         current_sol = candidate_sol
                         improved = True
                         break
                if improved:
                    break
                if time.time() > deadline - 5:
                    break

    # Format output
    final_added = []
    seen = set()
    for u, v in current_sol:
        if u < v:
            key = (u, v)
        else:
            key = (v, u)
        if key not in seen:
            final_added.append([int(key[0]), int(key[1])])
            seen.add(key)
            
    return final_added

# EVOLVE-BLOCK-END
