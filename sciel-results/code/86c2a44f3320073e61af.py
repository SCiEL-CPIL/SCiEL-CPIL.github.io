import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
M = len(P['edge_order'])
START = 0

# Precompute edge position lookup (1-based index)
_edge_pos = {}
for i, (a, b) in enumerate(P['edge_order'], 1):
    _edge_pos[(a, b)] = i
    _edge_pos[(b, a)] = i


def _cycle_score(cycle):
    """Largest 1-based arrival index of edges in the cycle (with implicit closing edge)."""
    mx = 0
    n = len(cycle)
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        p = _edge_pos.get((a, b), 0)
        if p > mx:
            mx = p
    return mx


def _build_adj(T):
    """Adjacency bitmask list for the first T edges."""
    adj = [0] * N
    for i in range(T):
        a, b = P['edge_order'][i]
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    return adj


def _dp_hamilton_cycle(T, deadline):
    """
    Exact bitmask DP to find a Hamiltonian cycle in the first T edges.
    Returns a cycle (list of vertices) or None.
    Fast for small T (sparse graph).
    """
    adj = _build_adj(T)
    n = N
    start = START
    full_mask = (1 << n) - 1
    start_bit = 1 << start

    # dp[mask][last] = previous vertex on the path (or -1 if unreachable)
    # We only need paths that start at `start` and end at `last`, covering `mask`.
    # To save memory, store as list of dicts: dp[mask] = {last: prev}
    dp = [None] * (1 << n)
    dp[start_bit] = {start: -1}

    # Iterate masks in increasing order of popcount
    for mask in range(1 << n):
        if not (mask & start_bit):
            continue
        prev_map = dp[mask]
        if prev_map is None:
            continue
        # For each last vertex in this mask that is reachable
        for last in list(prev_map.keys()):
            # Check deadline
            if time.time() > deadline:
                return None
            # Try extending to neighbors of last not in mask
            avail = adj[last] & ~mask
            while avail:
                lb = avail & (-avail)
                v = lb.bit_length() - 1
                avail ^= lb
                new_mask = mask | lb
                if dp[new_mask] is None:
                    dp[new_mask] = {}
                # Only store if not already set (first found is fine for existence)
                if v not in dp[new_mask]:
                    dp[new_mask][v] = last

    # Check if full mask reachable with a last vertex connected to start
    full_map = dp[full_mask]
    if full_map is None:
        return None
    for last in full_map:
        if adj[last] & start_bit:
            # Reconstruct path
            path = [last]
            mask = full_mask
            prev = full_map[last]
            while prev != -1:
                path.append(prev)
                mask ^= (1 << prev)
                # find prev's predecessor
                pm = dp[mask]
                prev = pm[prev]
            path.append(start)
            path.reverse()
            return path
    return None


def _dfs_hamilton_cycle(T, deadline, start=None):
    """
    Memoized DFS to find a Hamiltonian cycle in the first T edges.
    Uses forced-endpoint pruning and articulation-point pruning.
    Returns a cycle or None.
    """
    if start is None:
        start = START
    adj = _build_adj(T)
    n = N
    start_bit = 1 << start
    full_mask = (1 << n) - 1

    # Memo: (mask, last) -> True if a Hamiltonian path from start covering mask ending at last exists
    memo = {}

    def dfs(mask, last):
        if time.time() > deadline:
            return None
        if mask == full_mask:
            # Check if last connects to start
            if adj[last] & start_bit:
                return [last]
            return None

        key = (mask, last)
        if key in memo:
            return memo[key]

        avail = adj[last] & ~mask
        # Forced endpoint pruning: if only one unvisited vertex remains, it must be connected to start
        if avail and (avail & (avail - 1)) == 0:
            # Only one candidate
            v_bit = avail
            v = v_bit.bit_length() - 1
            if not (adj[v] & start_bit):
                memo[key] = None
                return None

        # Try candidates, ordered by degree (fewest unvisited neighbors first) for faster failure
        candidates = []
        a = avail
        while a:
            lb = a & (-a)
            v = lb.bit_length() - 1
            a ^= lb
            # Count unvisited neighbors of v (excluding last which is already visited)
            unvisited_neighbors = adj[v] & ~mask
            candidates.append((unvisited_neighbors.bit_count(), v))
        candidates.sort()

        for _, v in candidates:
            v_bit = 1 << v
            res = dfs(mask | v_bit, v)
            if res is not None:
                res.append(last)
                memo[key] = res
                return res

        memo[key] = None
        return None

    res = dfs(start_bit, start)
    if res is not None:
        res.append(start)
        res.reverse()
        return res
    return None


def _find_cycle_in_threshold(T, deadline):
    """
    Find a Hamiltonian cycle using only edges from the first T edges.
    Uses DP for small T, DFS for larger T.
    Returns cycle or None.
    """
    # For small T, DP is exact and fast
    if T <= 350:
        cycle = _dp_hamilton_cycle(T, deadline)
        if cycle is not None:
            return cycle
        # DP said no cycle exists (or timed out). If it completed, no cycle.
        # Check if it timed out: if the full mask was never processed, it might have timed out
        # For safety, fall through to DFS
        # Actually, if DP returns None, it either found no cycle or timed out.
        # If T is small enough that DP should complete quickly, None means no cycle.
        # But to be safe, let's also try DFS for small T in case DP had issues.
    
    # For larger T, use DFS
    cycle = _dfs_hamilton_cycle(T, deadline)
    return cycle


def _polish_for_lower_score(cycle, current_score, deadline):
    """
    Try to find a cycle with a lower max edge index than current_score.
    Search thresholds from current_score-1 downward.
    Returns a better cycle or the original.
    """
    best_cycle = cycle
    best_score = current_score
    
    # Try to find a cycle with max edge index < current_score
    # We can binary search or linear search downward
    # Since we're polishing, try a few thresholds below current_score
    for T in range(current_score - 1, max(N, current_score - 50), -1):
        if time.time() > deadline:
            break
        cycle2 = _find_cycle_in_threshold(T, deadline)
        if cycle2 is not None:
            score2 = _cycle_score(cycle2)
            if score2 < best_score:
                best_score = score2
                best_cycle = cycle2
                # Can't do better than T, so break
                break
    
    return best_cycle, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    sys.setrecursionlimit(10000)
    
    # Binary search for the smallest T where a Hamiltonian cycle exists
    lo, hi = N, M  # A cycle needs at least N edges
    
    best_cycle = None
    best_score = float('inf')
    
    # First, verify the full graph has a Hamiltonian cycle
    cycle_full = _find_cycle_in_threshold(M, deadline)
    if cycle_full is not None:
        best_cycle = cycle_full
        best_score = _cycle_score(cycle_full)
    
    # Binary search
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        cycle_mid = _find_cycle_in_threshold(mid, deadline)
        if cycle_mid is not None:
            score_mid = _cycle_score(cycle_mid)
            if score_mid < best_score:
                best_score = score_mid
                best_cycle = cycle_mid
            hi = mid
        else:
            lo = mid + 1
    
    # Check the final lo
    if time.time() < deadline:
        cycle_lo = _find_cycle_in_threshold(lo, deadline)
        if cycle_lo is not None:
            score_lo = _cycle_score(cycle_lo)
            if score_lo < best_score:
                best_score = score_lo
                best_cycle = cycle_lo
    
    # Polish: try to find a cycle with a lower max edge index
    if best_cycle is not None and time.time() < deadline:
        best_cycle, best_score = _polish_for_lower_score(best_cycle, best_score, deadline)
    
    # Fallback
    if best_cycle is None:
        best_cycle = list(range(N))
    
    return list(best_cycle)
# EVOLVE-BLOCK-END
