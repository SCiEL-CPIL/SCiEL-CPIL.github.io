"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
TOTAL = 1 << N  # 512
DEADLINE_MARGIN = 15.0  # seconds to spare before the 150s kill

# EVOLVE-BLOCK-START

# Precompute adjacency: adj[v] = bitmask of neighbors of v in Q_N
_ADJ = [0] * TOTAL
for v in range(TOTAL):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _ADJ[v] = mask


def _is_valid_snake(path):
    """Check if a list of integers forms a valid induced path in Q_N."""
    n = len(path)
    if n < 2:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def _greedy_seed():
    """Generate a greedy snake of length ~104 in Q_9 using bitmask speedup."""
    path = [0]
    used_mask = 1  # bitmask of used vertices
    blocked_mask = 0  # vertices that cannot be added next (adjacent to non-consecutive)
    while True:
        cur = path[-1]
        # Candidates: neighbors of cur not in used and not in blocked
        candidates = _ADJ[cur] & ~used_mask & ~blocked_mask
        if candidates == 0:
            break
        # Pick first available candidate (or use Warnsdorff)
        # Use Warnsdorff: pick candidate with fewest future options
        best_v = -1
        best_degree = TOTAL + 1
        c = candidates
        while c:
            low = c & (-c)
            v = low.bit_length() - 1
            c ^= low
            # Count available neighbors of v
            avail = _ADJ[v] & ~used_mask & ~blocked_mask & ~(1 << v)
            deg = avail.bit_count()
            if deg < best_degree:
                best_degree = deg
                best_v = v
        if best_v < 0:
            break
        path.append(best_v)
        used_mask |= 1 << best_v
        # Update blocked: vertices adjacent to best_v that are not cur and not already used
        # These become blocked for the next step (they'd be adjacent to non-consecutive best_v)
        new_blocked = _ADJ[best_v] & ~(1 << cur) & ~used_mask
        # But we also need to unblock: the old cur was the last, so vertices only blocked by cur
        # Actually, blocked should be: vertices adjacent to any vertex in path[:-1] that aren't used
        # Let's recompute: a vertex x is blocked if x is adjacent to some path[i] where i < len(path)-1
        # and x is not used. When we add best_v, new vertices become blocked if adjacent to best_v
        # and not adjacent to cur (since cur is now non-consecutive).
        # Wait - actually the blocked set is: all vertices that are adjacent to any vertex in path[:-1]
        # When we extend path by best_v, the new path[:-1] includes old path[:-1] plus cur.
        # So new blocked = old_blocked | (adj(cur) & ~used_mask) but we need to be careful.
        # Let me think again:
        # Before: path = [..., prev, cur], blocked = {x : x adj to some path[i], i < len-1, x not used}
        # After: path = [..., prev, cur, best_v], new blocked = {x : x adj to some path[i], i < len-1, x not used}
        #   = old_blocked | {x : x adj to cur, x not used, x != best_v}
        # But also best_v is now used, so remove it from blocked if it was there.
        blocked_mask |= (_ADJ[cur] & ~used_mask & ~(1 << best_v))
        # Remove best_v from blocked (it's now used)
        blocked_mask &= ~(1 << best_v)
        # Also, cur is no longer the last vertex, so its neighbors become blocked
        # But we already added them above.
    return path


def _bitmask_dfs(path, used_mask, blocked_mask, deadline, best_path):
    """
    Bitmask-accelerated DFS backtracking.
    path: current path (list)
    used_mask: bitmask of used vertices
    blocked_mask: bitmask of vertices that are adjacent to non-consecutive vertices (can't be next)
    deadline: time.time() limit
    best_path: list to store the best path found so far
    """
    if len(path) > len(best_path):
        best_path.clear()
        best_path.extend(path)

    if time.time() > deadline:
        return

    cur = path[-1]
    # Candidates: neighbors of cur, not used, not blocked
    candidates = _ADJ[cur] & ~used_mask & ~blocked_mask

    if candidates == 0:
        return

    # Build candidate list and sort by Warnsdorff heuristic (fewest future options)
    cand_list = []
    c = candidates
    while c:
        low = c & (-c)
        v = low.bit_length() - 1
        c ^= low
        # Heuristic: count available neighbors of v (fewer = more constrained = try first)
        avail = _ADJ[v] & ~used_mask & ~blocked_mask
        deg = avail.bit_count()
        cand_list.append((deg, v))

    cand_list.sort()

    for deg, v in cand_list:
        if time.time() > deadline:
            return

        path.append(v)
        new_used = used_mask | (1 << v)
        # New blocked: old blocked + neighbors of cur that aren't v and aren't used
        new_blocked = blocked_mask | (_ADJ[cur] & ~new_used & ~(1 << v))
        # Remove v from blocked (it's now used)
        new_blocked &= ~(1 << v)

        _bitmask_dfs(path, new_used, new_blocked, deadline, best_path)

        path.pop()
        if time.time() > deadline:
            return


def _warnsdorff_greedy(start=0):
    """
    Warnsdorff-like greedy from a given start vertex.
    Returns the path.
    """
    path = [start]
    used_mask = 1 << start
    blocked_mask = 0
    while True:
        cur = path[-1]
        candidates = _ADJ[cur] & ~used_mask & ~blocked_mask
        if candidates == 0:
            break
        # Pick candidate with fewest future options
        best_v = -1
        best_degree = TOTAL + 1
        c = candidates
        while c:
            low = c & (-c)
            v = low.bit_length() - 1
            c ^= low
            avail = _ADJ[v] & ~used_mask & ~blocked_mask
            deg = avail.bit_count()
            if deg < best_degree:
                best_degree = deg
                best_v = v
        if best_v < 0:
            break
        path.append(best_v)
        used_mask |= 1 << best_v
        new_blocked = blocked_mask | (_ADJ[cur] & ~used_mask & ~(1 << best_v))
        new_blocked &= ~(1 << best_v)
        blocked_mask = new_blocked
    return path


def _random_greedy_prefix(start, rng, length=15):
    """Generate a random greedy prefix of given length from start."""
    path = [start]
    used_mask = 1 << start
    blocked_mask = 0
    for _ in range(length):
        cur = path[-1]
        candidates = _ADJ[cur] & ~used_mask & ~blocked_mask
        if candidates == 0:
            break
        # Pick randomly
        c = candidates
        cands = []
        while c:
            low = c & (-c)
            v = low.bit_length() - 1
            cands.append(v)
            c ^= low
        v = rng.choice(cands)
        path.append(v)
        used_mask |= 1 << v
        new_blocked = blocked_mask | (_ADJ[cur] & ~used_mask & ~(1 << v))
        new_blocked &= ~(1 << v)
        blocked_mask = new_blocked
    return path, used_mask, blocked_mask


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Returns a list of distinct integers in [0, 2**N).
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    rng = random.Random(42)

    # Get initial seeds from multiple strategies
    best_path = _greedy_seed()
    best_len = len(best_path)

    # Try Warnsdorff from vertex 0
    ws = _warnsdorff_greedy(0)
    if len(ws) > best_len:
        best_path = list(ws)
        best_len = len(ws)

    # Try Warnsdorff from a few other starts
    for s in [1, 2, 5, 10, 20, 50, 100, 200, 400]:
        ws = _warnsdorff_greedy(s)
        if len(ws) > best_len:
            best_path = list(ws)
            best_len = len(ws)

    # Now do bitmask DFS from the best seed and from random prefixes
    # Start with the best seed
    _bitmask_dfs(list(best_path), 0, 0, deadline, best_path)
    # Rebuild used_mask and blocked_mask for best_path
    used_mask = 0
    for v in best_path:
        used_mask |= 1 << v
    # Compute blocked_mask: vertices adjacent to any non-last vertex
    if len(best_path) >= 2:
        blocked_mask = 0
        for v in best_path[:-1]:
            blocked_mask |= _ADJ[v]
        blocked_mask &= ~used_mask
    else:
        blocked_mask = 0

    _bitmask_dfs(list(best_path), used_mask, blocked_mask, deadline, best_path)

    # Try multiple random starts with DFS
    for start in range(0, TOTAL, 3):  # Every 3rd vertex for diversity
        if time.time() > deadline:
            break
        if time.time() > deadline - 5:
            break  # Leave buffer

        # Generate a random prefix
        prefix, um, bm = _random_greedy_prefix(start, rng, length=12)
        if len(prefix) > best_len:
            best_path = list(prefix)
            best_len = len(prefix)

        _bitmask_dfs(prefix, um, bm, deadline, best_path)

    # Final validation
    if not _is_valid_snake(best_path):
        # Fallback to the greedy seed
        best_path = _greedy_seed()

    return best_path

# EVOLVE-BLOCK-END
