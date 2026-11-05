"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

_N = 24

_EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# edge_arrival[u][v] = 1-based index
_edge_arrival = np.zeros((_N, _N), dtype=np.int32)
for idx, (u, v) in enumerate(_EDGE_ORDER, 1):
    _edge_arrival[u, v] = idx
    _edge_arrival[v, u] = idx

def _cycle_score(cycle):
    n = len(cycle)
    mx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if _edge_arrival[u, v] > mx:
            mx = _edge_arrival[u, v]
    return mx

def _build_adj_list(k):
    """Build adjacency list using only edges with arrival index <= k."""
    adj = [[] for _ in range(_N)]
    for i in range(_N):
        for j in range(i + 1, _N):
            if _edge_arrival[i, j] <= k:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def _check_hamilton_cycle_dp(k, deadline):
    """
    Check if a Hamilton cycle exists using only edges with arrival index <= k.
    Uses Held-Karp style bitmask DP.
    Returns True if exists, False otherwise.
    """
    n = _N
    start = time.time()
    
    # Quick checks
    adj = _build_adj_list(k)
    for i in range(n):
        if len(adj[i]) < 2:
            return False
    
    # Connectivity check
    visited = [False] * n
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    if count < n:
        return False
    
    # Bitmask DP: dp[mask][v] = True if there's a Hamilton path from 0 to v visiting exactly the vertices in mask
    # mask has bit i set if vertex i is visited
    # v is the last vertex in the path
    # We fix vertex 0 as the start.
    
    full_mask = (1 << n) - 1
    start_bit = 1  # vertex 0
    
    # dp[mask] is a bitset of vertices v such that there's a path 0->...->v visiting mask
    # Use array of uint64 for the vertex bitset (24 bits fit in uint64)
    num_masks = 1 << n
    # dp[mask] = bitset of possible endpoints
    dp = np.zeros(num_masks, dtype=np.uint64)
    dp[start_bit] = start_bit  # path of just vertex 0, endpoint is 0
    
    # Precompute adjacency bitsets
    adj_bits = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)
    
    # Iterate over masks in order of increasing popcount
    # For each mask containing start_bit, and for each endpoint v in dp[mask],
    # extend to neighbors w not in mask
    # dp[mask | (1<<w)] |= (1 << w)
    
    # Process masks by popcount for efficiency
    # But for n=24, we can just iterate all masks
    # Optimization: only process masks that contain start_bit
    start_bit_val = 1
    
    for mask in range(num_masks):
        if time.time() > deadline:
            return False
        if (mask & start_bit_val) == 0:
            continue
        endpoints = dp[mask]
        if endpoints == 0:
            continue
        
        # For each endpoint v in endpoints, try extending to neighbor w
        # We need to iterate over set bits in endpoints
        ep = endpoints
        while ep:
            if time.time() > deadline:
                return False
            v_bit = ep & (-ep)
            v = v_bit.bit_length() - 1
            ep ^= v_bit
            
            # Neighbors of v not in mask
            neighbors = adj_bits[v] & ~mask
            while neighbors:
                w_bit = neighbors & (-neighbors)
                w = w_bit.bit_length() - 1
                neighbors ^= w_bit
                new_mask = mask | w_bit
                dp[new_mask] |= w_bit
    
    # Check if full_mask has any endpoint v such that v is connected to 0
    full_endpoints = dp[full_mask]
    # Need v such that edge (v, 0) exists, i.e., v in adj[0]
    adj0_bits = adj_bits[0]
    if full_endpoints & adj0_bits:
        return True
    return False

def _extract_cycle_backtrack(adj, deadline):
    """Extract a Hamilton cycle from the adjacency using backtracking with good heuristics."""
    n = _N
    sys.setrecursionlimit(10000)
    
    # Sort neighbors by degree (ascending) for MRV heuristic
    adj_sorted = []
    for i in range(n):
        adj_sorted.append(sorted(adj[i], key=lambda x: len(adj[x])))
    
    used = [False] * n
    path = [0]
    used[0] = True
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            if path[0] in adj[path[-1]]:
                return list(path)
            return None
        
        cur = path[-1]
        candidates = []
        for nb in adj_sorted[cur]:
            if not used[nb]:
                # If this is the second-to-last position, nb must connect back to start
                if len(path) == n - 1:
                    if path[0] in adj[nb]:
                        candidates.append(nb)
                else:
                    candidates.append(nb)
        
        if not candidates:
            return None
        
        for nb in candidates:
            used[nb] = True
            path.append(nb)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
            used[nb] = False
            if time.time() > deadline:
                return None
        
        return None
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin within 90s limit
    
    n = _N
    num_edges = len(_EDGE_ORDER)
    
    # Binary search on K: minimum threshold such that Hamilton cycle exists
    lo = n  # minimum possible: need at least n edges
    hi = num_edges  # worst case: all edges
    
    best_K = None
    
    # First, do a quick check at hi to make sure a cycle exists
    if not _check_hamilton_cycle_dp(hi, deadline - 5):
        # Shouldn't happen for complete graph, but fallback
        return list(range(n))
    
    # Binary search
    while lo <= hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        if _check_hamilton_cycle_dp(mid, deadline - 10):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_K is None:
        best_K = hi  # fallback
    
    # Extract the actual cycle
    adj = _build_adj_list(best_K)
    cycle = _extract_cycle_backtrack(adj, deadline - 2)
    
    if cycle is None or len(cycle) != n or set(cycle) != set(range(n)):
        # Fallback: try with slightly larger K
        for delta in range(1, 20):
            if time.time() > deadline - 2:
                break
            k_try = best_K + delta
            if k_try > num_edges:
                break
            adj_try = _build_adj_list(k_try)
            cycle = _extract_cycle_backtrack(adj_try, deadline - 2)
            if cycle is not None and len(cycle) == n and set(cycle) == set(range(n)):
                break
        if cycle is None or len(cycle) != n or set(cycle) != set(range(n)):
            cycle = list(range(n))
    
    return cycle
# EVOLVE-BLOCK-END
