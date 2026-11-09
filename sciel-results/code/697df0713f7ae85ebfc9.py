"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin for 90s limit
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build base adjacency and distance matrix (Path graph)
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Distances in path: dist(i, j) = |i - j|
    # We only care about pairs with dist > 4.
    # For a pair (i, j) with |i-j| > 4, adding an edge (u, v) (where u,v are even-odd)
    # creates a path of length dist(i,u) + 1 + dist(v,j) or dist(i,v) + 1 + dist(u,j).
    # If either is <= 4, the pair is "covered".
    
    # Generate candidate edges: all even-odd pairs not in base path
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 != 0:
                if (u, v) not in base_edges:
                    candidates.append((u, v))
                    
    m = len(candidates)
    
    # Identify bad pairs
    bad_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) > 4:
                bad_pairs.append((i, j))
                
    p = len(bad_pairs)
    
    # Build coverage matrix: A[p, m] where A[pi, mj] = 1 if edge mj covers pair pi
    # A pair (i, j) is covered by edge (u, v) if:
    # min( |i-u| + 1 + |v-j|, |i-v| + 1 + |u-j| ) <= 4
    
    A = np.zeros((p, m), dtype=int)
    for pi, (i, j) in enumerate(bad_pairs):
        for mj, (u, v) in enumerate(candidates):
            d1 = abs(i - u) + 1 + abs(v - j)
            d2 = abs(i - v) + 1 + abs(u - j)
            if d1 <= 4 or d2 <= 4:
                A[pi, mj] = 1
                
    # Objective: minimize sum(x)
    c = np.ones(m)
    
    # Constraints: A @ x >= 1 for all p pairs
    # LinearConstraint(A.T, lb=np.ones(p), ub=np.inf)
    constraints = LinearConstraint(A.T, lb=np.ones(p), ub=np.inf)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integers
    integrality = np.ones(m)
    
    # Solve ILP
    # Use HiGHS solver (default in scipy)
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0 and res.x is not None:
        # Extract selected edges
        selected = []
        for i in range(m):
            if res.x[i] > 0.5:
                selected.append(candidates[i])
        
        # Verification step: Ensure diameter is actually <= 4 and triangle-free
        # The ILP formulation ensures that for every pair (i,j) with base dist > 4,
        # there is a path of length <= 4 using at most one added edge.
        # This guarantees the final distance is <= 4.
        # Triangle-free is guaranteed by only selecting even-odd edges.
        
        # Let's do a quick BFS verification just to be absolutely safe against numerical issues
        # or logical gaps in the "single edge shortcut" argument (though it is mathematically sound).
        
        def verify(added_edges):
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
                for d in dist:
                    if d == -1:
                        return False
                    if d > max_d:
                        max_d = d
                if max_d > 4:
                    return False
            return True

        if verify(selected):
            # Convert to list of lists
            result = [[int(u), int(v)] for u, v in selected]
            return result
            
    # Fallback: If ILP fails or verification fails, use a known valid heuristic solution
    # Greedy Set Cover or the previous hub solution.
    # Let's try a simple greedy set cover on the same matrix A.
    if time.time() < deadline:
        remaining = set(range(p))
        selected = []
        while remaining:
            best_idx = -1
            best_count = -1
            for mj in range(m):
                if any(candidates[mj] in s for s in selected):
                    continue
                count = sum(1 for pi in remaining if A[pi, mj] == 1)
                if count > best_count:
                    best_count = count
                    best_idx = mj
            if best_idx == -1:
                break
            selected.append(candidates[best_idx])
            for pi in range(p):
                if A[pi, best_idx] == 1:
                    remaining.discard(pi)
            
        if verify(selected):
            result = [[int(u), int(v)] for u, v in selected]
            return result
            
    # Ultimate fallback: The 10-edge hub solution
    initial_added = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
    if verify(initial_added):
        return [[int(u), int(v)] for u, v in initial_added]
        
    # If even that fails (shouldn't happen), return empty (invalid but better than crash)
    return []

# EVOLVE-BLOCK-END
