"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build base adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Identify candidate edges: must be bipartite (even-odd) to guarantee triangle-free
    # Base path is bipartite. Adding even-odd edges keeps it bipartite.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 != 0:
                if (u, v) not in base_edges and (v, u) not in base_edges:
                    candidates.append((u, v))
                    
    # Identify the "universe" of pairs that need to be brought within distance 4
    # A pair (i, j) is "bad" if dist(i, j) > 4 in the current graph.
    # We need to select edges such that for all pairs, dist(i, j) <= 4.
    
    # Pre-compute distances in base graph (path)
    # In a path, dist(i, j) = |i - j|
    bad_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) > 4:
                bad_pairs.add((i, j))
                
    # For each candidate edge, determine which bad pairs it "covers"
    # A pair (i, j) is covered by edge (u, v) if adding (u, v) makes dist(i, j) <= 4.
    # This is equivalent to: there exists a path from i to j of length <= 4 using (u, v) and base edges.
    # Since we only add one edge at a time for coverage calculation, we check if the shortest path
    # in the graph with just this added edge is <= 4.
    
    # Optimization: 
    # dist(i, j) with edge (u, v) is min( dist_base(i, j), dist_base(i, u) + 1 + dist_base(v, j), dist_base(i, v) + 1 + dist_base(u, j) )
    # But this assumes the rest of the graph is the base path. 
    # Wait, when we add multiple edges, they interact. 
    # However, for set cover formulation, we usually assume independence or use a greedy heuristic.
    # Exact set cover with interaction is NP-hard and complex. 
    # Let's refine: The condition is that the *final* diameter is <= 4.
    # This is not a simple set cover because edges can combine to shorten paths.
    # Example: Edge A shortens path from i to k, Edge B shortens k to j. Together they shorten i to j.
    # So standard set cover doesn't directly apply to "distance <= 4".
    
    # Alternative Strategy: 
    # Since n=24 is small, we can use a smarter search.
    # The diameter constraint is global.
    # Let's try a different angle: 
    # The graph must be connected and diameter <= 4.
    # This implies that for every vertex v, there is a "center" within distance 2? No, diameter 4 means eccentricity <= 4.
    
    # Let's stick to the previous heuristic but improve it with a more exhaustive local search or simulated annealing?
    # The current score is 12.1 (10 edges). 
    # Can we do 9 edges?
    # Let's analyze the lower bound.
    # Each added edge (u, v) can reduce the distance between many pairs.
    
    # Let's try a different exact approach: 
    # Branch and Bound on the number of edges.
    # Start with k=1, check if any k-edge subset works.
    # If not, k=2, etc.
    # Given the symmetry, we can prune heavily.
    
    # However, checking if a specific set of edges yields diameter <= 4 is O(n^2) BFS.
    # Number of candidates is ~ 12*12 - 11 = 133.
    # Choosing k from 133 is huge.
    
    # Let's refine the "Cover" concept.
    # If we add an edge (u, v), the new distance between any i, j is at most:
    # min(dist(i, j), dist(i, u)+1+dist(v, j), dist(i, v)+1+dist(u, j))
    # This is valid for a single edge.
    # For multiple edges, the distance can only decrease.
    # So, if a pair (i, j) is NOT covered by ANY single edge in our set (meaning even the best single-edge shortcut doesn't bring it to <=4), 
    # it might still be covered by a combination.
    # But if we require that EVERY bad pair is "covered" by at least one edge in the sense that the single-edge shortcut brings it to <=4, 
    # is that sufficient?
    # No. Consider i -> u -> v -> j. If dist(i, u)=1, dist(v, j)=1, then dist(i, j) becomes 3.
    # If dist(i, u)=2, dist(v, j)=2, then dist(i, j) becomes 5. Not covered by single edge logic.
    # But if we have another edge w, x that helps...
    
    # Actually, the condition "diameter <= 4" is equivalent to:
    # For all i, j, there is a path of length <= 4.
    # Let's look at the structure.
    # A known result for path augmentation:
    # To make P_n have diameter d, we need roughly n/(d+1) edges?
    # For d=4, n=24. 24/5 ~ 5?
    # Let's test small k.
    
    # I will implement a randomized local search with multiple restarts, but with a stronger "repair" mechanism.
    # Or, better: use the fact that the optimal solution likely has a symmetric structure.
    
    # Let's try to find a solution with 9 edges using a guided random search.
    # 1. Generate random sets of 9 edges.
    # 2. Check validity.
    # 3. If invalid, try to fix by swapping edges.
    
    # Even better: Use the previous hub solution as a seed and perform a "swap" local search.
    # Current best: 10 edges.
    # Try to replace 2 edges with 1? No, cardinality must decrease.
    # Try to remove 1 edge and add 0? (Current code does this).
    # Try to remove 2 edges and add 1? (Net -1).
    
    # Let's implement a 2-opt style move: Remove edge A, Remove edge B, Add edge C.
    # If diameter <= 4 and triangle-free, accept.
    
    def compute_diameter_fast(added_edges):
        # Build adjacency
        full_adj = [set() for _ in range(n)]
        for u, v in base_edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        for u, v in added_edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
            
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nb in full_adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            # Check max dist
            for d in dist:
                if d == -1:
                    return float('inf')
                if d > max_d:
                    max_d = d
            if max_d > 4:
                return max_d
        return max_d

    # Initial solution from previous best (10 edges)
    # Hub 12 connected to odd vertices: 1, 3, 5, 7, 9, 15, 17, 19, 21, 23
    # Note: 11 and 13 are neighbors of 12, so they are already dist 1.
    initial_added = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
    
    # Verify
    if compute_diameter_fast(initial_added) > 4:
        # Fallback to empty or simple valid
        initial_added = []
        
    best = initial_added[:]
    best_d = compute_diameter_fast(best)
    
    # Local Search: Swap (remove 2, add 1) to reduce size by 1
    # Also try (remove 1, add 0)
    
    def is_valid(added):
        # Triangle-free check: all edges must be even-odd
        for u, v in added:
            if (u - v) % 2 == 0:
                return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing 1 edge
        for i in range(len(best)):
            if time.time() > deadline: break
            trial = best[:i] + best[i+1:]
            if is_valid(trial) and compute_diameter_fast(trial) <= 4:
                best = trial
                improved = True
                break
                
        if improved: continue
        
        # Try removing 2 edges and adding 1 edge
        # This is O(k^2 * M) where M is num candidates. k~10, M~130. 100*130 = 13k checks.
        # Each check is O(n^2) BFS ~ 24^2 = 576 ops. 13k * 576 ~ 7.5M ops. Feasible.
        
        for i in range(len(best)):
            if time.time() > deadline: break
            for j in range(i+1, len(best)):
                if time.time() > deadline: break
                # Remove i and j
                removed = {best[i], best[j]}
                base_trial = [e for idx, e in enumerate(best) if idx != i and idx != j]
                
                # Try adding each candidate
                for cand in candidates:
                    if cand in removed: continue
                    if cand in base_trial: continue
                    
                    trial = base_trial + [cand]
                    if not is_valid(trial): continue
                    
                    d = compute_diameter_fast(trial)
                    if d <= 4:
                        best = trial
                        improved = True
                        break
                if improved: break
            if improved: break
            
    # Convert to list of lists
    result = [[int(u), int(v)] for u, v in best]
    return result
# EVOLVE-BLOCK-END
