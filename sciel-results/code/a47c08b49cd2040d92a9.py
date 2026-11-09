import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 82  # safety margin under 90s kill


def _build_seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _verify(comps):
    """Vectorized zero-one verification over all 2^N boolean inputs.

    Returns True iff the comparator network sorts every boolean input.
    Uses NumPy uint32 arrays so all 131072 inputs are processed at once.
    """
    total = 1 << N
    masks = np.arange(total, dtype=np.uint32)
    for (i, j) in comps:
        bi = np.uint32(1 << i)
        bj = np.uint32(1 << j)
        # swap iff bit i is 1 AND bit j is 0
        cond = (masks & bi) & ~(masks & bj)
        if np.any(cond):
            swap_mask = np.uint32((1 << i) | (1 << j))
            masks = np.where(cond, masks ^ swap_mask, masks)
    # sorted iff mask & (mask+1) == 0  (i.e. mask is of form (1<<k)-1)
    return bool(np.all((masks & (masks + 1)) == 0))


def _depth_of_network(comps):
    """Compute parallel depth (longest path) of the network."""
    if not comps:
        return 0
    depth = [0] * N
    for (i, j) in comps:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)


def _greedy_remove(comps, deadline):
    """Repeatedly try to remove each comparator until a full pass removes none.
    Returns the reduced list (in-place modification of a copy)."""
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(comps)
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return comps
            new_comps = comps[:idx] + comps[idx + 1:]
            if _verify(new_comps):
                comps = new_comps
                improved = True
                break  # restart scan from the end
    return comps


def _local_search(comps, rng, deadline):
    """Local search with swaps, insertions, and re-removal to escape minima."""
    comps = list(comps)
    n0 = len(comps)
    while time.time() < deadline:
        moved = False
        # Try a few random swaps (reorder) and re-remove
        for _ in range(3):
            if time.time() > deadline:
                break
            n = len(comps)
            if n < 2:
                break
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            cand = comps[:]
            cand[i], cand[j] = cand[j], cand[i]
            if _verify(cand):
                reduced = _greedy_remove(cand, deadline)
                if len(reduced) < len(comps):
                    comps = reduced
                    moved = True
                    break
                elif len(reduced) == len(comps) and rng.random() < 0.3:
                    # accept a same-size reorder to change the landscape
                    comps = reduced
                    moved = True
                    break
        # Try inserting a comparator then re-removing (escape move)
        if not moved and time.time() < deadline:
            for _ in range(2):
                if time.time() > deadline:
                    break
                a = rng.randint(0, N - 1)
                b = rng.randint(a + 1, N)
                pos = rng.randint(0, len(comps) + 1)
                cand = comps[:pos] + [(a, b)] + comps[pos:]
                if _verify(cand):
                    reduced = _greedy_remove(cand, deadline)
                    if len(reduced) < len(comps):
                        comps = reduced
                        moved = True
                        break
        if not moved:
            # no improvement this round; a couple more rounds then stop
            break
    return comps


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    comps = _build_seed_network()

    # Verify seed is correct
    if not _verify(comps):
        return comps

    best_comps = list(comps)
    best_len = len(comps)

    # Phase 1: initial greedy removal
    comps = _greedy_remove(comps, deadline)
    if len(comps) < best_len:
        best_len = len(comps)
        best_comps = list(comps)

    # Phase 2: multi-start local search with restarts
    rng = np.random.default_rng(42)
    while time.time() < deadline - 1:
        # restart from the current best to explore a fresh neighborhood
        comps = _local_search(best_comps, rng, deadline)
        if len(comps) < best_len:
            best_len = len(comps)
            best_comps = list(comps)
        # If we've hit a reasonable size and are burning time, stop
        if best_len <= 75 and time.time() > start_time + DEADLINE_SECONDS * 0.6:
            break

    return best_comps
