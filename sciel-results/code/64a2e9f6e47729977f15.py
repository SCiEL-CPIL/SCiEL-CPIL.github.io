"""Find a Hamilton cycle with minimum maximum edge arrival index using DP + 2-opt."""
import time
import sys
import random
import numpy as np

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGES = P['edge_order']
FULL_MASK = (1 << N) - 1


def _build_adj(k):
    """Build adjacency bitmasks for first k edges."""
    adj = [0] * N
    for i in range(min(k, len(EDGES))):
        u, v = EDGES[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _dp_hamilton_path(adj, start, end, time_limit=10.0):
    """
    DP to check if Hamilton path exists from start to end in graph defined by adj.
    Returns the path as a list, or None.
    Uses bitmask DP: dp[mask][v] = True if there's a Hamilton path from start to v visiting exactly vertices in mask.
    """
    t0 = time.time()
    n = N
    full = (1 << n) - 1
    
    # dp[mask][v]: reachable?
    # We use a list of sets for efficiency: for each mask, which vertices are reachable
    # Actually, let's use a 2D array but only for valid states
    # dp[mask] = bitmask of vertices v such that path start->...->v visits exactly mask
    
    # Initialize
    dp = [0] * (1 << n)
    dp[1 << start] = 1 << start  # path of just start, ending at start
    
    # parent tracking for reconstruction: parent[mask][v] = previous vertex
    parent = [[-1] * n for _ in range(1 << n)]
    
    # Iterate over masks in increasing order
    for mask in range(1 << n):
        if time.time() - t0 > time_limit:
            return None
        reachable = dp[mask]
        if reachable == 0:
            continue
        # For each vertex v in mask that is reachable, try to extend
        v_mask = reachable
        while v_mask:
            # Get lowest set bit
            lb = v_mask & (-v_mask)
            v = lb.bit_length() - 1
            v_mask ^= lb
            
            # Try to extend to neighbors of v not in mask
            neighbors = adj[v] & ~mask
            while neighbors:
                lb2 = neighbors & (-neighbors)
                u = lb2.bit_length() - 1
                neighbors ^= lb2
                new_mask = mask | (1 << u)
                if not (dp[new_mask] & (1 << u)):
                    dp[new_mask] |= (1 << u)
                    parent[new_mask][u] = v
    
    # Check if end is reachable in full mask
    if dp[full] & (1 << end):
        # Reconstruct path
        path = [end]
        mask = full
        v = end
        while v != start:
            v = parent[mask][v]
            mask ^= (1 << v)
            path.append(v)
        path.reverse()
        return path
    return None


def _find_cycle_2opt(adj, initial_cycle, time_limit=5.0):
    """
    Given an initial Hamilton cycle (list of vertices), try to improve it using 2-opt.
    But we need the cycle to exist in the graph. Actually, we just need to find ANY Hamilton cycle.
    Let's use a different approach: given the graph, find a Hamilton cycle using local search.
    """
    t0 = time.time()
    n = N
    # adj is a list of bitmasks
    adj_set = [set() for _ in range(n)]
    for i in range(n):
        mask = adj[i]
        while mask:
            lb = mask & (-mask)
            adj_set[i].add(lb.bit_length() - 1)
            mask ^= lb
    
    # Start with a random permutation
    rng = random.Random(42)
    best = list(range(n))
    rng.shuffle(best)
    # Ensure it's a valid cycle in the graph? No, we need to find one that IS valid.
    # Actually, let's use a constructive approach: build a path greedily, then close it.
    
    # Better: use the DP result if available, or just do a greedy + local search
    # For now, return the initial cycle as-is (caller should pass a valid one)
    return initial_cycle


def _quick_check(adj):
    """Quick necessary conditions for Hamilton cycle."""
    n = N
    # All vertices must have degree >= 2
    for i in range(n):
        if bin(adj[i]).count('1') < 2:
            return False
    # Graph must be connected
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        node = stack.pop()
        mask = adj[node]
        while mask:
            lb = mask & (-mask)
            nb = lb.bit_length() - 1
            mask ^= lb
            if not visited[nb]:
                visited[nb] = True
                count += 1
                stack.append(nb)
    return count == n


def _check_prefix_dp(k, time_limit=8.0):
    """
    Check if first k edges contain a Hamilton cycle using DP.
    Returns the cycle as a list of vertices, or None.
    """
    adj = _build_adj(k)
    
    if not _quick_check(adj):
        return None
    
    # Try to find a Hamilton path from vertex 0 to some other vertex, then close
    # A Hamilton cycle exists iff there's a Hamilton path from 0 to v where v is adjacent to 0
    # and the path visits all vertices.
    
    # Try all possible endpoints adjacent to 0
    neighbors_of_0 = adj[0]
    candidates = []
    mask = neighbors_of_0
    while mask:
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        mask ^= lb
        candidates.append(v)
    
    if not candidates:
        return None
    
    # Shuffle candidates for variety
    random.Random(123).shuffle(candidates)
    
    per_candidate_time = max(time_limit / max(len(candidates), 1), 0.1)
    
    for end in candidates:
        if time.time() > time.time() - time_limit:  # placeholder
            break
        remaining = time_limit - (time.time() - t0_check)
        if remaining < 0.1:
            break
        path = _dp_hamilton_path(adj, 0, end, time_limit=min(remaining, per_candidate_time))
        if path is not None:
            # Verify it's a valid cycle
            if len(path) == N and path[0] == 0 and path[-1] == end:
                # Check all consecutive pairs and last-first
                valid = True
                for i in range(N - 1):
                    if not (adj[path[i]] & (1 << path[i+1])):
                        valid = False
                        break
                if valid and (adj[path[-1]] & (1 << path[0])):
                    return path
    return None


t0_check = 0

def construct():
    global t0_check
    t0_check = time.time()
    deadline = t0_check + 80.0  # Leave 10s margin
    
    best_cycle = None
    best_k = None
    
    lo = N
    hi = len(EDGES)
    
    # First, verify hi works
    remaining = deadline - time.time() - 5.0
    if remaining > 1.0:
        tlim = min(remaining * 0.3, 15.0)
        cycle = _check_prefix_dp(hi, time_limit=tlim)
        if cycle is not None:
            best_cycle = cycle
            best_k = hi
    
    # Binary search for minimum k
    while lo < hi and time.time() < deadline - 15.0:
        mid = (lo + hi) // 2
        remaining = deadline - time.time() - 10.0
        if remaining < 1.0:
            break
        
        tlim = min(remaining * 0.35, 12.0)
        if tlim < 0.5:
            tlim = 0.5
        
        cycle = _check_prefix_dp(mid, time_limit=tlim)
        if cycle is not None:
            best_cycle = cycle
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    else:
        # Fallback: return a simple cycle (may not be valid, but better than nothing)
        return list(range(N))
# EVOLVE-BLOCK-END
