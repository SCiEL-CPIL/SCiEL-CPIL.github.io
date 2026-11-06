import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']

# Precompute arrival index (0-based) for each edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    key = (min(u, v), max(u, v))
    arrival[key] = idx

def get_arrival(u, v):
    key = (min(u, v), max(u, v))
    return arrival[key]

def compute_score(cycle):
    """Compute the score (max 1-based arrival index) of a Hamilton cycle."""
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = get_arrival(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1  # 1-based

def hungarian(cost_matrix):
    """
    Solve the assignment problem (minimize total cost) using the Hungarian algorithm.
    cost_matrix is an n x n matrix. Returns (assignment, total_cost) where
    assignment[i] = j means row i is assigned to column j.
    """
    n_local = len(cost_matrix)
    INF = float('inf')
    
    # u[i] = potential for row i, v[j] = potential for column j
    u = [0] * (n_local + 1)
    v = [0] * (n_local + 1)
    # p[j] = row assigned to column j (1-indexed columns)
    p = [0] * (n_local + 1)
    # way[j] = previous column in the augmenting path
    way = [0] * (n_local + 1)
    
    for i in range(1, n_local + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n_local + 1)
        used = [False] * (n_local + 1)
        
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            
            for j in range(1, n_local + 1):
                if not used[j]:
                    cur = cost_matrix[i0-1][j-1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            
            for j in range(0, n_local + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            
            j0 = j1
            if p[j0] == 0:
                break
        
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    
    # Build assignment: assignment[i] = j for row i (0-indexed)
    assignment = [0] * n_local
    for j in range(1, n_local + 1):
        if p[j] > 0:
            assignment[p[j]-1] = j - 1
    
    total_cost = sum(cost_matrix[i][assignment[i]] for i in range(n_local))
    return assignment, total_cost

def cycle_cover_bottleneck(threshold):
    """
    Find a cycle cover (permutation with no fixed points) using only edges
    with arrival <= threshold. Returns the permutation or None.
    Uses Hungarian on a cost matrix where edges within threshold have cost 0
    and others have cost 1. We want a permutation with all 0-cost edges.
    """
    # Build cost matrix: 0 if edge within threshold and not self-loop, 1 otherwise
    cost = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                cost[i][j] = 1  # forbid self-loops
            else:
                if get_arrival(i, j) <= threshold:
                    cost[i][j] = 0
                else:
                    cost[i][j] = 1
    
    assignment, total_cost = hungarian(cost)
    
    # Check if all assigned edges are within threshold
    if total_cost == 0:
        # Verify no self-loops
        for i in range(n):
            if assignment[i] == i:
                return None
        return assignment
    
    # If not all zero, try a different approach:
    # Use a large penalty for out-of-threshold edges
    BIG = 1000
    cost2 = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                cost2[i][j] = BIG * 2
            else:
                if get_arrival(i, j) <= threshold:
                    cost2[i][j] = 0
                else:
                    cost2[i][j] = BIG
    
    assignment2, total_cost2 = hungarian(cost2)
    
    # Check if we got a valid cycle cover within threshold
    max_arr = 0
    for i in range(n):
        j = assignment2[i]
        if i == j:
            continue
        a = get_arrival(i, j)
        if a > max_arr:
            max_arr = a
    
    if max_arr <= threshold:
        return assignment2
    
    return None

def extract_cycles(perm):
    """Extract cycles from a permutation."""
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = perm[cur]
            if len(cycle) > 1:
                cycles.append(cycle)
    return cycles

def two_opt_repair(perm, threshold, time_limit=5.0):
    """
    Given a cycle cover (permutation), try to merge subtours into a single
    Hamilton cycle using 2-opt style exchanges, keeping max edge arrival <= threshold.
    Returns the Hamilton cycle as a list of vertices, or None.
    """
    deadline = time.time() + time_limit
    cycles = extract_cycles(perm)
    
    if len(cycles) == 1:
        # Already a Hamilton cycle
        return cycles[0]
    
    # Try to merge pairs of cycles
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cycles = extract_cycles(perm)
        if len(cycles) == 1:
            return cycles[0]
        
        # Try all pairs of cycles
        for ci in range(len(cycles)):
            if time.time() > deadline:
                break
            for cj in range(ci + 1, len(cycles)):
                if time.time() > deadline:
                    break
                cyc_i = cycles[ci]
                cyc_j = cycles[cj]
                
                # Try all 4 possible 2-opt exchanges between the two cycles
                # Cycle i: a -> b (consecutive in cycle i)
                # Cycle j: c -> d (consecutive in cycle j)
                # Exchange: remove a->b and c->d, add a->c and b->d (or a->d and b->c)
                
                for ai in range(len(cyc_i)):
                    if time.time() > deadline:
                        break
                    bi = (ai + 1) % len(cyc_i)
                    a = cyc_i[ai]
                    b = cyc_i[bi]
                    
                    for cj_idx in range(len(cyc_j)):
                        if time.time() > deadline:
                            break
                        di = (cj_idx + 1) % len(cyc_j)
                        c = cyc_j[cj_idx]
                        d = cyc_j[di]
                        
                        # Option 1: a->c, b->d
                        if get_arrival(a, c) <= threshold and get_arrival(b, d) <= threshold:
                            # Perform the exchange
                            new_perm = perm[:]
                            # In cycle i: a was going to b, now a goes to c
                            # In cycle j: c was going to d, now b goes to d
                            # d's predecessor in cycle j is c, now d's predecessor is b
                            # b's predecessor in cycle i is a, now a's successor is c
                            
                            # a -> c (was a -> b)
                            new_perm[a] = c
                            # b -> d (was some other)
                            new_perm[b] = d
                            
                            # Check: this should merge the two cycles
                            new_cycles = extract_cycles(new_perm)
                            if len(new_cycles) < len(cycles):
                                perm = new_perm
                                improved = True
                                break
                        # Option 2: a->d, b->c
                        elif get_arrival(a, d) <= threshold and get_arrival(b, c) <= threshold:
                            new_perm = perm[:]
                            new_perm[a] = d
                            new_perm[b] = c
                            
                            new_cycles = extract_cycles(new_perm)
                            if len(new_cycles) < len(cycles):
                                perm = new_perm
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    cycles = extract_cycles(perm)
    if len(cycles) == 1:
        return cycles[0]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    # Default cycle
    default_cycle = list(range(n))
    best_score = compute_score(default_cycle)
    best_cycle = default_cycle
    
    # Binary search on bottleneck threshold
    # Lower bound: need at least n edges, each vertex degree >= 2 in the subgraph
    # Try from low to high
    lo = 23
    hi = 275
    
    # First, find the minimum threshold where a cycle cover exists
    best_threshold = None
    best_perm = None
    
    # Binary search for minimum threshold with a valid cycle cover
    while lo <= hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        perm = cycle_cover_bottleneck(mid)
        if perm is not None:
            best_threshold = mid
            best_perm = perm
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Now we have the minimum threshold for a cycle cover
    # Try to repair it into a Hamilton cycle
    if best_perm is not None and time.time() < deadline - 5:
        # Try 2-opt repair
        remaining_time = min(30.0, deadline - time.time() - 3)
        ham_cycle = two_opt_repair(best_perm, best_threshold, time_limit=remaining_time)
        
        if ham_cycle is not None:
            score = compute_score(ham_cycle)
            if score < best_score:
                best_score = score
                best_cycle = ham_cycle
        
        # If 2-opt didn't work, try with slightly higher thresholds
        if best_score > best_threshold + 1:
            for extra in range(1, 20):
                if time.time() > deadline - 3:
                    break
                t = best_threshold + extra
                perm = cycle_cover_bottleneck(t)
                if perm is None:
                    continue
                remaining_time = min(10.0, deadline - time.time() - 2)
                ham_cycle = two_opt_repair(perm, t, time_limit=remaining_time)
                if ham_cycle is not None:
                    score = compute_score(ham_cycle)
                    if score < best_score:
                        best_score = score
                        best_cycle = ham_cycle
                    if score <= t + 1:
                        break
    
    # Also try a direct backtracking approach with the best threshold found
    if time.time() < deadline - 5:
        # Try backtracking from the best threshold
        for t in range(best_threshold if best_threshold else 23, min(best_threshold + 5 if best_threshold else 28, 60)):
            if time.time() > deadline - 3:
                break
            cycle = _backtrack_hamiltonian(t, time_limit=3.0)
            if cycle is not None:
                score = compute_score(cycle)
                if score < best_score:
                    best_score = score
                    best_cycle = cycle
    
    return best_cycle

def _backtrack_hamiltonian(max_arrival, time_limit, start_vertex=0):
    """
    Find a Hamiltonian cycle using only edges with arrival <= max_arrival.
    Uses Warnsdorff's heuristic.
    """
    adj = {i: [] for i in range(n)}
    for idx in range(max_arrival + 1):
        u, v = edge_order[idx]
        adj[u].append(v)
        adj[v].append(u)
    
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Check connectivity
    visited = [False] * n
    queue = deque([start_vertex])
    visited[start_vertex] = True
    count = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    if count != n:
        return None
    
    for v in range(n):
        adj[v].sort(key=lambda x: get_arrival(v, x))
    
    deadline = time.time() + time_limit
    result = [None]
    
    def backtrack():
        if time.time() > deadline:
            return False
        if len(path) == n:
            last = path[-1]
            if get_arrival(last, start_vertex) <= max_arrival:
                result[0] = path[:]
                return True
            return False
        
        current = path[-1]
        remaining = n - len(path)
        
        candidates = []
        for nb in adj[current]:
            if visited[nb]:
                continue
            if remaining == 2:
                if get_arrival(nb, start_vertex) > max_arrival:
                    continue
            candidates.append(nb)
        
        if not candidates:
            return False
        
        # Warnsdorff: sort by number of onward options
        candidates.sort(key=lambda x: sum(1 for nb2 in adj[x] if not visited[nb2]))
        
        for nb in candidates:
            visited[nb] = True
            path.append(nb)
            if backtrack():
                return True
            path.pop()
            visited[nb] = False
        
        return False
    
    visited = [False] * n
    path = [start_vertex]
    visited[start_vertex] = True
    
    backtrack()
    return result[0]

# EVOLVE-BLOCK-END
