import time
import sys
from collections import deque
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
m = len(edge_order)

# Precompute arrival index for each edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    arrival[(min(u, v), max(u, v))] = idx

def edge_arrival(u, v):
    if u > v:
        u, v = v, u
    return arrival[(u, v)]

def cycle_score(cycle):
    s = 0
    L = len(cycle)
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        ar = edge_arrival(a, b) + 1
        if ar > s:
            s = ar
    return s

def hungarian(cost):
    """
    Solve the assignment problem using the Hungarian algorithm.
    cost is an n x n matrix. Returns the assignment (list of column indices for each row).
    """
    n = len(cost)
    INF = float('inf')
    
    # u[i] for row i, v[j] for column j
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = row assigned to column j
    way = [0] * (n + 1)
    
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            
            j0 = j1
            if p[j0] == 0:
                break
        
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    
    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    
    return assignment

def find_cycles_from_assignment(assignment):
    """
    Given an assignment (each vertex i is mapped to assignment[i]),
    find the cycles in the resulting permutation.
    """
    n = len(assignment)
    visited = [False] * n
    cycles = []
    
    for start in range(n):
        if visited[start]:
            continue
        cycle = []
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = assignment[cur]
        cycles.append(cycle)
    
    return cycles

def merge_cycles(cycles, arrival_map, deadline):
    """
    Try to merge disjoint cycles into a single Hamiltonian cycle
    using 2-opt style swaps.
    """
    if len(cycles) == 1:
        return cycles[0]
    
    n = sum(len(c) for c in cycles)
    best_cycle = None
    best_score = float('inf')
    
    # Try all pairs of cycles and all pairs of edges to swap
    for _ in range(50):  # Limit iterations
        if time.time() > deadline:
            break
            
        progress = False
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                if len(cycles[i]) == 0 or len(cycles[j]) == 0:
                    continue
                    
                ci = cycles[i]
                cj = cycles[j]
                
                # Try all pairs of edges: one from ci, one from cj
                for ai in range(len(ci)):
                    for bj in range(len(cj)):
                        # Edge in ci: ci[ai] -> ci[(ai+1) % len(ci)]
                        # Edge in cj: cj[bj] -> cj[(bj+1) % len(cj)]
                        u1 = ci[ai]
                        v1 = ci[(ai + 1) % len(ci)]
                        u2 = cj[bj]
                        v2 = cj[(bj + 1) % len(cj)]
                        
                        # Swap: remove (u1,v1) and (u2,v2), add (u1,v2) and (u2,v1)
                        # Check if these new edges exist in our graph (arrival < some bound)
                        # For now, just check if they're valid edges (they always are in complete graph)
                        
                        # Create merged cycle
                        # ci: ... u1 -> v1 ...
                        # cj: ... u2 -> v2 ...
                        # After swap: ... u1 -> v2 ... and ... u2 -> v1 ...
                        
                        # Reconstruct the two cycles
                        # ci becomes: [v1, ..., u1] + [v2, ..., u2] (reversed cj part)
                        # Actually, let's think more carefully
                        
                        # Original ci: c0, c1, ..., c_{k-1} where ci[ai] = u1, ci[(ai+1)%k] = v1
                        # Original cj: d0, d1, ..., d_{l-1} where cj[bj] = u2, cj[(bj+1)%l] = v2
                        
                        # After removing edges (u1,v1) and (u2,v2) and adding (u1,v2) and (u2,v1):
                        # We get one cycle: v1, ..., u1, v2, ..., u2, v1
                        # Wait, that's not right. Let me reconsider.
                        
                        # ci: ... -> u1 -> v1 -> ... -> u1 (cycle)
                        # cj: ... -> u2 -> v2 -> ... -> u2 (cycle)
                        
                        # Remove u1->v1 and u2->v2
                        # Add u1->v2 and u2->v1
                        # New cycle: ... -> u1 -> v2 -> ... -> u2 -> v1 -> ... -> u1
                        
                        # So the merged cycle is:
                        # Start from v1, go around ci to u1, then to v2, go around cj to u2, then to v1
                        # But we need to be careful about the direction
                        
                        # Let's construct it:
                        # ci segment: from v1 to u1 (going forward in ci)
                        # cj segment: from v2 to u2 (going forward in cj)
                        
                        k = len(ci)
                        l = len(cj)
                        
                        # ci: indices ai+1, ai+2, ..., k-1, 0, 1, ..., ai
                        # That's from v1 to u1
                        ci_part = [ci[(ai + 1 + t) % k] for t in range(k)]
                        # ci_part[0] = v1, ci_part[-1] = u1
                        
                        # cj: indices bj+1, bj+2, ..., l-1, 0, 1, ..., bj
                        # That's from v2 to u2
                        cj_part = [cj[(bj + 1 + t) % l] for t in range(l)]
                        # cj_part[0] = v2, cj_part[-1] = u2
                        
                        # Merged cycle: ci_part + cj_part
                        # But we need to close it: last element is u2, first is v1
                        # We need edge u2 -> v1, which is what we added
                        merged = ci_part + cj_part
                        
                        # Compute score of this merged cycle
                        score = 0
                        for idx in range(len(merged)):
                            a = merged[idx]
                            b = merged[(idx + 1) % len(merged)]
                            ar = edge_arrival(a, b) + 1
                            if ar > score:
                                score = ar
                        
                        if score < best_score:
                            best_score = score
                            # Update cycles: replace ci and cj with merged
                            new_cycles = []
                            for idx_c, c in enumerate(cycles):
                                if idx_c == i or idx_c == j:
                                    continue
                                new_cycles.append(c)
                            new_cycles.append(merged)
                            cycles = new_cycles
                            progress = True
                            break
                        
                        if progress:
                            break
                    if progress:
                        break
                if progress:
                    break
            if progress:
                break
        
        if not progress:
            break
    
    if len(cycles) == 1:
        return cycles[0]
    
    # If we couldn't merge into one cycle, return the best single cycle we can find
    # Or return None to indicate failure
    return None

def solve_with_hungarian(K, deadline):
    """
    Find a Hamiltonian cycle using edges with arrival index <= K.
    Uses Hungarian algorithm to find minimum-weight cycle cover, then merges cycles.
    """
    # Build cost matrix: cost[i][j] = arrival index of edge (i,j) if <= K, else INF
    INF = float('inf')
    cost = [[INF] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ar = edge_arrival(i, j)
            if ar <= K:
                cost[i][j] = ar
    
    # Check if each row and column has at least one finite value
    for i in range(n):
        if all(cost[i][j] == INF for j in range(n)):
            return None
        if all(cost[j][i] == INF for j in range(n)):
            return None
    
    # Solve assignment
    assignment = hungarian(cost)
    
    # Verify assignment is valid
    for i in range(n):
        if cost[i][assignment[i]] == INF:
            return None
    
    # Find cycles
    cycles = find_cycles_from_assignment(assignment)
    
    if len(cycles) == 1:
        return cycles[0]
    
    # Try to merge cycles
    merged = merge_cycles(cycles, arrival, deadline)
    if merged is not None:
        return merged
    
    return None

def backtracking_hc(adj, deadline):
    """
    Backtracking search for Hamiltonian cycle in graph defined by adj.
    """
    n = len(adj)
    if n < 3:
        return None
    
    # Degree check
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Connectivity check
    visited = [False] * n
    dq = deque([0])
    visited[0] = True
    count = 0
    while dq:
        u = dq.popleft()
        count += 1
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                dq.append(w)
    if count != n:
        return None
    
    # 2-Connectivity check (Articulation Points)
    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    ap = [False] * n
    time_counter = [0]
    
    def tarjan(u):
        nonlocal time_counter
        disc[u] = low[u] = time_counter[0]
        time_counter[0] += 1
        children = 0
        for v in adj[u]:
            if disc[v] == -1:
                children += 1
                parent[v] = u
                tarjan(v)
                low[u] = min(low[u], low[v])
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap[u] = True
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    
    tarjan(0)
    if any(ap):
        return None
    
    # Backtracking with Warnsdorff's heuristic
    start = 0
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            last = path[-1]
            if start in adj[last]:
                return path[:]
            return None
        
        cur = path[-1]
        candidates = []
        for nb in adj[cur]:
            if not visited[nb]:
                onward = 0
                for w in adj[nb]:
                    if not visited[w]:
                        onward += 1
                if len(path) == n - 1:
                    if start not in adj[nb]:
                        continue
                candidates.append((onward, nb))
        
        candidates.sort()
        
        for _, nb in candidates:
            visited[nb] = True
            path.append(nb)
            
            # Connectivity prune
            unvisited_mask = 0
            for i in range(n):
                if not visited[i]:
                    unvisited_mask |= (1 << i)
            
            if unvisited_mask:
                start_unvisited = -1
                for i in range(n):
                    if (unvisited_mask >> i) & 1:
                        start_unvisited = i
                        break
                
                seen = 0
                dq = deque([start_unvisited])
                seen |= (1 << start_unvisited)
                
                while dq:
                    u = dq.popleft()
                    for w in adj[u]:
                        if (unvisited_mask >> w) & 1:
                            if not (seen >> w) & 1:
                                seen |= (1 << w)
                                dq.append(w)
                                
                if seen != unvisited_mask:
                    path.pop()
                    visited[nb] = False
                    continue
            
            res = backtrack()
            if res:
                return res
            
            path.pop()
            visited[nb] = False
            
        return None
    
    return backtrack()

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    best_cycle = None
    best_score = m + 1
    
    # First, try greedy 2-regular to get an initial upper bound
    degree = [0] * n
    adj_greedy = [set() for _ in range(n)]
    parent = list(range(n))
    size = [1] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]
        return True
    
    for idx in range(m):
        if time.time() > deadline - 5.0:
            break
        u, v = edge_order[idx]
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        ru, rv = find(u), find(v)
        if ru == rv:
            if size[ru] < n:
                continue
        degree[u] += 1
        degree[v] += 1
        adj_greedy[u].add(v)
        adj_greedy[v].add(u)
        union(u, v)
        if all(d == 2 for d in degree):
            root = find(0)
            ok = all(find(i) == root for i in range(n))
            if ok:
                cycle = [0]
                prev = -1
                cur = 0
                for _ in range(n - 1):
                    nxts = list(adj_greedy[cur])
                    nxt = nxts[0] if nxts[0] != prev else nxts[1]
                    prev, cur = cur, nxt
                    cycle.append(cur)
                sc = cycle_score(cycle)
                best_score = sc
                best_cycle = cycle
                break
    
    # Binary search for minimum K using Hungarian + merge
    if best_cycle is not None:
        hi = best_score - 1
    else:
        hi = m - 1
    
    lo = 0
    
    # Try Hungarian approach for various K
    for K in range(0, hi + 1):
        if time.time() > deadline - 10.0:
            break
        
        # Check if graph with edges arrival <= K has minimum degree >= 2
        deg = [0] * n
        for idx in range(K + 1):
            u, v = edge_order[idx]
            deg[u] += 1
            deg[v] += 1
        
        if any(d < 2 for d in deg):
            continue
        
        # Try Hungarian
        cycle = solve_with_hungarian(K, deadline - 5.0)
        if cycle is not None:
            sc = cycle_score(cycle)
            if sc < best_score:
                best_score = sc
                best_cycle = cycle
            break
        
        # If Hungarian failed, try backtracking for smaller K values
        if K < 50 and time.time() < deadline - 20.0:
            adj = [set() for _ in range(n)]
            for idx in range(K + 1):
                u, v = edge_order[idx]
                adj[u].add(v)
                adj[v].add(u)
            
            check_deadline = min(deadline - 10.0, time.time() + 5.0)
            cycle = backtracking_hc(adj, check_deadline)
            if cycle:
                sc = cycle_score(cycle)
                if sc < best_score:
                    best_score = sc
                    best_cycle = cycle
                break
    
    if best_cycle is None:
        return list(range(n))
    
    return best_cycle

# EVOLVE-BLOCK-END
