"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # 1. Identify Candidate Edges
    # Triangle-free constraint on a path implies bipartite constraints.
    # We can only add edges between vertices of opposite parity (even/odd).
    # This ensures no odd cycles are formed, preserving triangle-freeness.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 1 and (u, v) not in base_edges:
                candidates.append((u, v))
                
    # Sort candidates by "potential" - longer edges generally reduce diameter more effectively.
    # We also want to prioritize edges that connect the "ends" or "middle" to cover the path span.
    # A simple heuristic: sort by distance descending.
    candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    # 2. Precompute base adjacency
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

    # 3. Exhaustive Search with Pruning
    # We search for the minimum k such that there exists a subset of k edges from a "shortlist"
    # that achieves diameter <= 4.
    
    # Why a shortlist?
    # C(144, 5) is ~1.3 billion. Too slow.
    # C(20, 5) is ~15,504. Very fast.
    # C(30, 5) is ~142,506. Fast.
    # C(40, 5) is ~658,000. Moderate.
    # C(50, 5) is ~2,118,760. Might be tight in Python for 50s if diameter check is slow.
    # Diameter check is O(N^2) = 576 ops. 
    # 2M * 576 ~ 1B ops. This is likely too slow for pure Python in 50s.
    
    # Optimization:
    # We can pre-filter the shortlist more aggressively.
    # The optimal solution for a path P_N to diameter D usually involves edges that "jump" large distances.
    # Let's try increasing shortlist size k from 15 to 25, and k_edges from 1 to 6.
    
    # We will iterate k_edges (number of added edges) from 1 upwards.
    # For each k_edges, we iterate shortlist size M from 10 to 30.
    # If we find a solution, we return it.
    # If we exhaust search for a given M and k, we increase M.
    # If we exhaust k up to some limit, we fall back to a greedy approach.
    
    # Actually, a smarter approach:
    # Just pick a fixed reasonable shortlist size, say M=25.
    # Then iterate k from 1 to 6.
    # If no solution found in C(25, 6) ~ 177k combinations, it's very unlikely a solution exists with
    # edges outside the top 25 longest edges.
    # If no solution found for k up to 6, use greedy as fallback.
    
    M = 25
    shortlist = candidates[:M]
    
    best_solution = None
    best_size = float('inf')
    
    # Iterate k from 1 to 6
    for k in range(1, 7):
        if time.time() > deadline:
            break
            
        # Generate all combinations of size k from shortlist
        # Use itertools.combinations
        found = False
        for combo in combinations(shortlist, k):
            if time.time() > deadline:
                break
                
            # Check if this combo is valid
            adj = [set(base_adj[i]) for i in range(n)]
            for u, v in combo:
                adj[u].add(v)
                adj[v].add(u)
                
            if get_diameter(adj) <= 4:
                best_solution = list(combo)
                best_size = k
                found = True
                break
        
        if found:
            break
            
    if best_solution is not None:
        return [[int(u), int(v)] for u, v in best_solution]
        
    # Fallback: Greedy approach if exhaustive search fails or times out
    # This ensures we always return a valid solution (though maybe not optimal)
    adj = [set(base_adj[i]) for i in range(n)]
    current_edges = []
    current_diam = get_diameter(adj)
    
    while current_diam > 4:
        if time.time() > deadline:
            break
        best_edge = None
        best_diam = current_diam + 1
        
        for u, v in candidates:
            if (u, v) in set(current_edges) or (v, u) in set(current_edges):
                continue
                
            adj[u].add(v)
            adj[v].add(u)
            
            new_diam = get_diameter(adj)
            
            adj[u].remove(v)
            adj[v].remove(u)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        current_edges.append((u, v))
        current_diam = best_diam
        
    # Local search refinement on greedy solution
    if current_diam <= 4:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_edges)):
                u, v = current_edges[i]
                # Try removing
                adj[u].remove(v)
                adj[v].remove(u)
                
                if get_diameter(adj) <= 4:
                    current_edges.pop(i)
                    improved = True
                    break
                else:
                    # Restore
                    adj[u].add(v)
                    adj[v].add(u)
                    
    if current_diam <= 4:
        return [[int(u), int(v)] for u, v in current_edges]
        
    # Emergency fallback: return empty or a known valid large set
    # A complete bipartite graph between evens and odds is always triangle-free and connected.
    # Diameter is at most 2 (if complete) or 3.
    # But we need to minimize edges. The greedy should have worked.
    # If we get here, something is wrong. Return a safe set.
    # Connect 0 to all odds, 2 to all odds, etc? No, just return the greedy result even if diam > 4?
    # The task says "Any broken rule makes the whole result invalid".
    # So we MUST return a valid solution.
    
    # Safe construction: 
    # Add edges (0, 2), (0, 4), ..., (0, 22) ? No, (0,2) is dist 2 (even), forbidden?
    # Wait, base path edges are (i, i+1). Parity differs.
    # So base edges are between even and odd.
    # To be triangle-free, we can only add edges between even and odd.
    # If we add ALL possible even-odd edges, we get K_{12,12}.
    # Diameter of K_{12,12} is 2.
    # Number of edges is 12*12 = 144.
    # This is a valid solution.
    
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (u-v)%2 == 1:
                all_edges.append([u, v])
                
    return all_edges

# EVOLVE-BLOCK-END
