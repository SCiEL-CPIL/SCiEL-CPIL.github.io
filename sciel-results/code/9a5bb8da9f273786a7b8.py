"""Construct a 12-state binary deterministic synchronizing automaton with a long reset word.

Strategy:
1. Fast BFS threshold computation using bitmask representation.
2. Systematic construction of candidate automata (cycle + perturbation families).
3. Steepest-ascent hill climbing on the neighborhood of the best candidate.
4. Random restarts with local search to escape local optima.
"""
import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1


def _popcount(x):
    """Count set bits in an integer."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def _apply_mask(a, b, mask, letter):
    """Apply a transition to a subset represented as a bitmask.
    letter=0 uses 'a', letter=1 uses 'b'.
    Returns the new bitmask."""
    result = 0
    if letter == 0:
        trans = a
    else:
        trans = b
    m = mask
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        result |= (1 << trans[idx])
        m ^= lb
    return result


def _threshold_fast(a, b):
    """BFS on subset automaton using bitmasks. Returns shortest reset word length or -1."""
    start = FULL_MASK
    if _popcount(start) == 1:
        return 0
    # visited: dict mapping bitmask -> depth
    visited = {start: 0}
    q = deque([start])
    while q:
        curr = q.popleft()
        d = visited[curr]
        if _popcount(curr) == 1:
            return d
        na = _apply_mask(a, b, curr, 0)
        if na not in visited:
            visited[na] = d + 1
            q.append(na)
        nb = _apply_mask(a, b, curr, 1)
        if nb not in visited:
            visited[nb] = d + 1
            q.append(nb)
    return -1


def _threshold(a, b):
    """Wrapper for threshold computation."""
    return _threshold_fast(a, b)


def _is_synchronizing(a, b):
    """Quick check if automaton synchronizes. Returns threshold or -1."""
    return _threshold(a, b)


def _hill_climb(a, b, deadline, best_thr, best_a, best_b):
    """Steepest-ascent hill climbing from (a, b). Returns updated best."""
    curr_a = a[:]
    curr_b = b[:]
    curr_thr = _threshold(curr_a, curr_b)
    if curr_thr <= 0:
        return best_a, best_b, best_thr

    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_neighbor_thr = curr_thr
        best_neighbor_a = None
        best_neighbor_b = None

        # Try changing one element of a
        for i in range(N):
            if time.time() > deadline:
                break
            old = curr_a[i]
            for new_val in range(N):
                if new_val == old:
                    continue
                curr_a[i] = new_val
                thr = _threshold(curr_a, curr_b)
                if thr > best_neighbor_thr:
                    best_neighbor_thr = thr
                    best_neighbor_a = curr_a[:]
                    best_neighbor_b = curr_b[:]
                curr_a[i] = old

        # Try changing one element of b
        for i in range(N):
            if time.time() > deadline:
                break
            old = curr_b[i]
            for new_val in range(N):
                if new_val == old:
                    continue
                curr_b[i] = new_val
                thr = _threshold(curr_a, curr_b)
                if thr > best_neighbor_thr:
                    best_neighbor_thr = thr
                    best_neighbor_a = curr_a[:]
                    best_neighbor_b = curr_b[:]
                curr_b[i] = old

        if best_neighbor_a is not None:
            curr_a = best_neighbor_a
            curr_b = best_neighbor_b
            curr_thr = best_neighbor_thr
            improved = True
            if curr_thr > best_thr:
                best_a = curr_a[:]
                best_b = curr_b[:]
                best_thr = curr_thr

    return best_a, best_b, best_thr


def construct():
    """Construct a 12-state binary deterministic synchronizing automaton with long reset word."""
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin below 90s wall-clock limit

    best_a = None
    best_b = None
    best_thr = 0

    a_cycle = [(i + 1) % N for i in range(N)]
    a_rev = [(i - 1) % N for i in range(N)]

    # === Phase 1: Systematic construction families ===

    # Family 1: a = cycle, b = monotone with one merge
    for merge_pos in range(N):
        if time.time() > deadline:
            break
        b_mono = list(range(N))
        b_mono[merge_pos] = merge_pos - 1 if merge_pos > 0 else N - 1
        for a in (a_cycle, a_rev):
            thr = _threshold(a, b_mono)
            if thr > 0 and thr > best_thr:
                best_a, best_b, best_thr = a[:], b_mono[:], thr

    # Family 2: a = cycle, b = identity with one perturbation
    for i in range(N):
        if time.time() > deadline:
            break
        for new_val in range(N):
            if new_val == i:
                continue
            b_pert = list(range(N))
            b_pert[i] = new_val
            for a in (a_cycle, a_rev):
                thr = _threshold(a, b_pert)
                if thr > 0 and thr > best_thr:
                    best_a, best_b, best_thr = a[:], b_pert[:], thr

    # Family 3: a = cycle, b = "almost identity" with two merges
    for i in range(N):
        for j in range(i + 1, N):
            if time.time() > deadline:
                break
            for target in range(N):
                if target == i or target == j:
                    continue
                b_two = list(range(N))
                b_two[i] = target
                b_two[j] = target
                for a in (a_cycle, a_rev):
                    thr = _threshold(a, b_two)
                    if thr > 0 and thr > best_thr:
                        best_a, best_b, best_thr = a[:], b_two[:], thr

    # Family 4: a = cycle, b = "shift with defect"
    for shift in range(N):
        for defect_pos in range(N):
            if time.time() > deadline:
                break
            for defect_val in range(N):
                b_shift = [(i + shift) % N for i in range(N)]
                b_shift[defect_pos] = defect_val
                for a in (a_cycle, a_rev):
                    thr = _threshold(a, b_shift)
                    if thr > 0 and thr > best_thr:
                        best_a, best_b, best_thr = a[:], b_shift[:], thr

    # Family 5: a = cycle, b = "bottleneck" - most states map forward, one maps to sink
    for sink in range(N):
        if time.time() > deadline:
            break
        for num_sink in range(1, 4):
            for positions in _combinations(range(N), num_sink):
                if time.time() > deadline:
                    break
                b_sink = [(i + 1) % N for i in range(N)]
                for p in positions:
                    b_sink[p] = sink
                for a in (a_cycle, a_rev):
                    thr = _threshold(a, b_sink)
                    if thr > 0 and thr > best_thr:
                        best_a, best_b, best_thr = a[:], b_sink[:], thr

    # Family 6: Known "pinwheel" constructions
    # a = cycle, b maps i -> (i+1) % N for most, but a few map to 0
    for num_zero in range(1, 5):
        for positions in _combinations(range(N), num_zero):
            if time.time() > deadline:
                break
            b_pin = [(i + 1) % N for i in range(N)]
            for p in positions:
                b_pin[p] = 0
            for a in (a_cycle, a_rev):
                thr = _threshold(a, b_pin)
                if thr > 0 and thr > best_thr:
                    best_a, best_b, best_thr = a[:], b_pin[:], thr

    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle[:]
            best_b = [0] * N
            best_thr = _threshold(best_a, best_b)
        return best_a, best_b

    # === Phase 2: Random search with local improvement ===
    rng = random.Random(42)

    # Try many random b with a = cycle
    for trial in range(500):
        if time.time() > deadline:
            break
        b_rand = [rng.randint(0, N - 1) for _ in range(N)]
        for a in (a_cycle, a_rev):
            thr = _threshold(a, b_rand)
            if thr > 0 and thr > best_thr:
                best_a, best_b, best_thr = a[:], b_rand[:], thr

    # Try random a (permutations) with random b
    for trial in range(200):
        if time.time() > deadline:
            break
        a_perm = list(range(N))
        rng.shuffle(a_perm)
        b_rand = [rng.randint(0, N - 1) for _ in range(N)]
        thr = _threshold(a_perm, b_rand)
        if thr > 0 and thr > best_thr:
            best_a, best_b, best_thr = a_perm[:], b_rand[:], thr

    # Try "structured random": a = cycle, b = mostly monotone with random perturbations
    for trial in range(300):
        if time.time() > deadline:
            break
        b_struct = list(range(N))
        # Apply 1-3 random perturbations
        num_pert = rng.randint(1, 3)
        for _ in range(num_pert):
            pos = rng.randint(0, N - 1)
            val = rng.randint(0, N - 1)
            b_struct[pos] = val
        for a in (a_cycle, a_rev):
            thr = _threshold(a, b_struct)
            if thr > 0 and thr > best_thr:
                best_a, best_b, best_thr = a[:], b_struct[:], thr

    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle[:]
            best_b = [0] * N
            best_thr = _threshold(best_a, best_b)
        return best_a, best_b

    # === Phase 3: Hill climbing from the best found ===
    if best_a is not None and best_b is not None:
        best_a, best_b, best_thr = _hill_climb(best_a, best_b, deadline, best_thr, best_a, best_b)

    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle[:]
            best_b = [0] * N
            best_thr = _threshold(best_a, best_b)
        return best_a, best_b

    # === Phase 4: More random restarts + hill climbing ===
    for restart in range(10):
        if time.time() > deadline:
            break
        # Generate a random starting point near the best
        a_start = best_a[:]
        b_start = best_b[:]
        # Mutate a few positions
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.5:
                pos = rng.randint(0, N - 1)
                a_start[pos] = rng.randint(0, N - 1)
            else:
                pos = rng.randint(0, N - 1)
                b_start[pos] = rng.randint(0, N - 1)

        thr = _threshold(a_start, b_start)
        if thr > 0:
            # Local search from here
            a_local = a_start[:]
            b_local = b_start[:]
            local_thr = thr
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try single mutations
                for i in range(N):
                    if time.time() > deadline:
                        break
                    old_a = a_local[i]
                    for new_val in range(N):
                        if new_val == old_a:
                            continue
                        a_local[i] = new_val
                        t = _threshold(a_local, b_local)
                        if t > local_thr:
                            local_thr = t
                            improved = True
                        a_local[i] = old_a

                    old_b = b_local[i]
                    for new_val in range(N):
                        if new_val == old_b:
                            continue
                        b_local[i] = new_val
                        t = _threshold(a_local, b_local)
                        if t > local_thr:
                            local_thr = t
                            improved = True
                        b_local[i] = old_b

            if local_thr > best_thr:
                best_a, best_b, best_thr = a_local[:], b_local[:], local_thr

    # Fallback: ensure we have a valid synchronizing automaton
    if best_a is None or best_b is None or best_thr <= 0:
        best_a = a_cycle[:]
        best_b = [0] * N
        best_thr = _threshold(best_a, best_b)

    return best_a, best_b


def _combinations(iterable, r):
    """Generate combinations of iterable taken r at a time."""
    pool = list(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield [pool[i] for i in indices]
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield [pool[i] for i in indices]
