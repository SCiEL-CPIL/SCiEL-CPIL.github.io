# EVOLVE-BLOCK-START
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build initial adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Generate candidate edges
    # Must not be in base, and must not create triangle with base edges initially?
    # Actually, we check dynamically.
    # But we can pre-filter edges that create triangles with BASE edges.
    # Base edges are (i, i+1).
    # Edge (u, v) creates triangle with base if there is w such that w in {u-1, u+1} and w in {v-1, v+1} and w is connected to both.
    # In path, u and v share a neighbor only if |u-v| == 2.
    # So edges with |u-v| == 2 are forbidden.
    # Edges with |u-v| == 1 are base edges (forbidden).
    # So candidates must have |u-v| >= 3.
    
    candidates = []
    for u in range(n):
        for v in range(u + 3, n):
            candidates.append((u, v))
            
    # Sort by length descending to prioritize large jumps
    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    def creates_triangle(u, v, adj):
        return len(adj[u] & adj[v]) > 0
    
    def check_diameter(adj, max_diam=4):
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_diam:
                            return False
                        q.append(nb)
            if any(d == -1 for d in dist):
                return False
        return True
    
    # Backtracking search for minimum k
    best_solution = None
    best_k = float('inf')
    
    # Upper bound: use a greedy heuristic to get a valid solution first
    # Greedy: add edges that reduce diameter most
    adj_greedy = [set(s) for s in adj]
    added_greedy = []
    remaining_greedy = set(candidates)
    
    def compute_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        max_d = max(max_d, dist[nb])
                        q.append(nb)
        return max_d
        
    while True:
        diam = compute_diameter(adj_greedy)
        if diam <= 4:
            break
        if not remaining_greedy:
            break
        best_edge = None
        best_new_diam = float('inf')
        for (u, v) in remaining_greedy:
            if creates_triangle(u, v, adj_greedy):
                continue
            adj_greedy[u].add(v)
            adj_greedy[v].add(u)
            new_diam = compute_diameter(adj_greedy)
            adj_greedy[u].remove(v)
            adj_greedy[v].remove(u)
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (u, v)
        if best_edge is None:
            break
        u, v = best_edge
        adj_greedy[u].add(v)
        adj_greedy[v].add(u)
        added_greedy.append((u, v))
        remaining_greedy.discard((u, v))
        
    if len(added_greedy) < best_k:
        best_k = len(added_greedy)
        best_solution = list(added_greedy)
        
    # Now search for smaller k
    # We search k from 1 to best_k - 1
    
    def backtrack(current_edges, current_adj, idx, k_target, k_current):
        nonlocal best_solution, best_k
        
        if k_current == k_target:
            if check_diameter(current_adj):
                if len(current_edges) < best_k:
                    best_k = len(current_edges)
                    best_solution = list(current_edges)
            return
            
        if idx >= len(candidates):
            return
            
        # Pruning: if even adding all remaining candidates cannot reach k_target? 
        # No, we just need to find ANY set of k_target edges.
        
        u, v = candidates[idx]
        
        # Option 1: Add edge (u, v)
        if not creates_triangle(u, v, current_adj):
            current_edges.append((u, v))
            current_adj[u].add(v)
            current_adj[v].add(u)
            
            backtrack(current_edges, current_adj, idx + 1, k_target, k_current + 1)
            
            current_edges.pop()
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
        # Option 2: Skip edge (u, v)
        # Optimization: if we skip, ensure we have enough candidates left to pick k_target - k_current
        remaining_candidates = len(candidates) - idx - 1
        if remaining_candidates >= k_target - k_current:
            backtrack(current_edges, current_adj, idx + 1, k_target, k_current)
            
    # Try k from 1 to best_k - 1
    for k in range(1, best_k):
        # Reset
        adj_search = [set(s) for s in adj]
        edges_search = []
        backtrack(edges_search, adj_search, 0, k, 0)
        if best_k <= k: # Found a solution with size k or smaller
            break
            
    if best_solution is not None:
        return [[u, v] for u, v in best_solution]
    else:
        # Fallback to greedy
        return [[u, v] for u, v in added_greedy]

# EVOLVE-BLOCK-END
