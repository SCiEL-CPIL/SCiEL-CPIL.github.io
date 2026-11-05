"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import math

TOTAL_POINTS = N * N
ALL_MASK = (1 << TOTAL_POINTS) - 1

# Point to index mapping
def point_to_idx(p):
    return p[0] * N + p[1]

def idx_to_point(i):
    return (i // N, i % N)

# Precompute conflict bitmasks at module level
# conflict[i][j] bit k is 1 if (i, j, k) forms an isosceles triangle
_points = [(i // N, i % N) for i in range(TOTAL_POINTS)]

_dist = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
for i in range(TOTAL_POINTS):
    xi, yi = _points[i]
    for j in range(i + 1, TOTAL_POINTS):
        xj, yj = _points[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        _dist[i][j] = d
        _dist[j][i] = d

_conflict = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
for i in range(TOTAL_POINTS):
    for j in range(i + 1, TOTAL_POINTS):
        mask = 0
        dij = _dist[i][j]
        for k in range(TOTAL_POINTS):
            if k == i or k == j:
                continue
            dik = _dist[i][k]
            djk = _dist[j][k]
            if dij == dik or dij == djk or dik == djk:
                mask |= (1 << k)
        _conflict[i][j] = mask
        _conflict[j][i] = mask


def _verify(sol):
    """Verify that sol is a valid maximal no-isosceles set."""
    # Check no isosceles triangles
    for a in range(len(sol)):
        for b in range(a + 1, len(sol)):
            for c in range(b + 1, len(sol)):
                i, j, k = sol[a], sol[b], sol[c]
                if _dist[i][j] == _dist[i][k] or _dist[i][j] == _dist[j][k] or _dist[i][k] == _dist[j][k]:
                    return False
    # Check maximality
    forbidden = 0
    for a in range(len(sol)):
        for b in range(a + 1, len(sol)):
            forbidden |= _conflict[sol[a]][sol[b]]
    for p in sol:
        forbidden |= (1 << p)
    return forbidden == ALL_MASK


def _greedy_build(order, stop_early=True):
    """Build a maximal set by greedily adding points in the given order.
    If stop_early, stop as soon as valid_mask becomes 0."""
    chosen = []
    valid_mask = ALL_MASK
    for p in order:
        if not (valid_mask & (1 << p)):
            continue
        chosen.append(p)
        valid_mask &= ~(1 << p)
        for q in chosen[:-1]:
            valid_mask &= ~_conflict[p][q]
        if valid_mask == 0 and stop_early:
            break
    return chosen


def _remove_point(sol, p):
    """Try to remove point p from sol. Returns new sol or None if removal breaks maximality."""
    new_sol = [x for x in sol if x != p]
    if len(new_sol) < 2:
        return None
    # Check no isosceles triangles in new_sol
    for a in range(len(new_sol)):
        for b in range(a + 1, len(new_sol)):
            for c in range(b + 1, len(new_sol)):
                i, j, k = new_sol[a], new_sol[b], new_sol[c]
                if _dist[i][j] == _dist[i][k] or _dist[i][j] == _dist[j][k] or _dist[i][k] == _dist[j][k]:
                    return None
    # Check maximality
    forbidden = 0
    for a in range(len(new_sol)):
        for b in range(a + 1, len(new_sol)):
            forbidden |= _conflict[new_sol[a]][new_sol[b]]
    for q in new_sol:
        forbidden |= (1 << q)
    if forbidden != ALL_MASK:
        return None
    return new_sol


def _local_search_remove(sol, time_limit):
    """Try to shrink sol by removing points one at a time."""
    current = sol[:]
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try removing each point
        for idx in range(len(current)):
            p = current[idx]
            new_sol = _remove_point(current, p)
            if new_sol is not None:
                current = new_sol
                improved = True
                break
    return current


def _dfs_find_maximal(k, time_limit, rng=None):
    """DFS to find a set of exactly k points that is maximal (forbidden covers all)."""
    if rng is None:
        rng = random
    best = [None]

    def dfs(S, valid_mask, depth):
        if time.time() > time_limit:
            return
        if depth == 0:
            if valid_mask == 0:
                best[0] = S[:]
            return

        if valid_mask.bit_count() < depth:
            return

        # Get candidates
        m = valid_mask
        candidates = []
        while m:
            lsb = m & -m
            candidates.append(lsb.bit_length() - 1)
            m ^= lsb

        # Shuffle with rng
        rng.shuffle(candidates)

        for p in candidates:
            if time.time() > time_limit:
                return
            if best[0] is not None:
                return
            new_valid = valid_mask & ~(1 << p)
            for q in S:
                new_valid &= ~_conflict[p][q]
            if new_valid == 0 and depth > 1:
                continue
            S.append(p)
            dfs(S, new_valid, depth - 1)
            S.pop()
            if best[0] is not None:
                return

    dfs([], ALL_MASK, k)
    return best[0]


def _simulated_annealing(initial_sol, time_limit):
    """SA that moves by removing a point and possibly adding points to restore maximality."""
    if not initial_sol:
        return None

    current = initial_sol[:]
    best = current[:]
    best_len = len(current)

    def cost(sol):
        """Lower is better. Penalize non-maximality heavily."""
        if len(sol) < 2:
            return 10000
        # Check independence
        for a in range(len(sol)):
            for b in range(a + 1, len(sol)):
                for c in range(b + 1, len(sol)):
                    i, j, k = sol[a], sol[b], sol[c]
                    if _dist[i][j] == _dist[i][k] or _dist[i][j] == _dist[j][k] or _dist[i][k] == _dist[j][k]:
                        return 10000
        # Compute uncovered
        forbidden = 0
        for a in range(len(sol)):
            for b in range(a + 1, len(sol)):
                forbidden |= _conflict[sol[a]][sol[b]]
        for p in sol:
            forbidden |= (1 << p)
        uncovered = ALL_MASK & ~forbidden
        return len(sol) + 50 * uncovered.bit_count()

    T = 2.0
    cooling = 0.995
    cur_cost = cost(current)

    while time.time() < time_limit:
        # Proposal: remove a random point
        if len(current) <= 2:
            break
        idx = random.randrange(len(current))
        p = current[idx]
        new_sol = [x for x in current if x != p]
        new_cost = cost(new_sol)

        delta = new_cost - cur_cost
        if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / max(T, 0.01))):
            current = new_sol
            cur_cost = new_cost
            if cur_cost < len(current) + 50:  # possibly maximal or close
                # Check if truly maximal
                forbidden = 0
                for a in range(len(current)):
                    for b in range(a + 1, len(current)):
                        forbidden |= _conflict[current[a]][current[b]]
                for q in current:
                    forbidden |= (1 << q)
                if forbidden == ALL_MASK:
                    if len(current) < best_len:
                        best = current[:]
                        best_len = len(current)
        T *= cooling

    # Try to restore maximality on best
    if _verify(best):
        return best
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    # Phase 1: Greedy constructions to get initial solutions
    best_sol = None
    best_len = TOTAL_POINTS

    # Try several random greedy orders
    for attempt in range(50):
        if time.time() > deadline - 60:
            break
        order = list(range(TOTAL_POINTS))
        random.shuffle(order)
        sol = _greedy_build(order, stop_early=True)
        if _verify(sol) and len(sol) < best_len:
            best_sol = sol
            best_len = len(sol)

    # Phase 2: Local search (removal) on best solution
    if best_sol and time.time() < deadline - 40:
        local_time = min(15.0, deadline - time.time() - 40)
        if local_time > 2:
            improved = _local_search_remove(best_sol, start_time + time.time() - start_time + local_time)
            # Correct time limit
            improved = _local_search_remove(best_sol, time.time() + local_time)
            if improved and len(improved) < best_len:
                best_sol = improved
                best_len = len(best_sol)

    # Phase 3: Try DFS for smaller k values
    # We know best_len, so try k from 2 up to best_len - 1
    max_k_to_try = best_len - 1 if best_sol else 20
    for k in range(2, min(max_k_to_try + 1, 30)):
        if time.time() > deadline - 30:
            break
        # Give more time to smaller k (easier to find), less to larger
        time_budget = min(5.0, max(1.0, (deadline - time.time() - 30) / max(1, max_k_to_try - k)))
        sol = _dfs_find_maximal(k, time.time() + time_budget)
        if sol is not None and _verify(sol):
            best_sol = sol
            best_len = k
            # Found a solution of size k; can stop early if k is very small
            if k <= 4:
                break

    # Phase 4: SA refinement on best solution
    if best_sol and time.time() < deadline - 10:
        sa_time = min(20.0, deadline - time.time() - 10)
        if sa_time > 3:
            sa_sol = _simulated_annealing(best_sol, time.time() + sa_time)
            if sa_sol and _verify(sa_sol) and len(sa_sol) < best_len:
                best_sol = sa_sol
                best_len = len(sa_sol)

    # Final local search on best
    if best_sol and time.time() < deadline - 5:
        ls_time = min(10.0, deadline - time.time() - 5)
        if ls_time > 1:
            final = _local_search_remove(best_sol, time.time() + ls_time)
            if final and _verify(final) and len(final) < best_len:
                best_sol = final
                best_len = len(final)

    # Fallback: should not reach here
    if best_sol is None:
        # Last resort: greedy
        order = list(range(TOTAL_POINTS))
        random.shuffle(order)
        best_sol = _greedy_build(order, stop_early=True)
        if not _verify(best_sol):
            # Try another order
            for _ in range(20):
                order = list(range(TOTAL_POINTS))
                random.shuffle(order)
                best_sol = _greedy_build(order, stop_early=True)
                if _verify(best_sol):
                    break

    return [idx_to_point(i) for i in best_sol]

# EVOLVE-BLOCK-END
