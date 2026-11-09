import time
import random
import sys

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85


def _build_seed_network():
    """Odd-even transposition network (136 comparators)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _verify_numpy(comps):
    """Fast zero-one principle verification using numpy."""
    import numpy as np
    total = 1 << N
    masks = np.arange(total, dtype=np.uint32)
    for (i, j) in comps:
        bi = np.uint32(1 << i)
        bj = np.uint32(1 << j)
        cond = (masks & bi) & ~(masks & bj)
        if np.any(cond):
            swap_mask = np.uint32((1 << i) | (1 << j))
            masks = np.where(cond, masks ^ swap_mask, masks)
    return bool(np.all((masks & (masks + 1)) == 0))


def _verify_list(comps):
    """
    Verification using a list of states. Fallback if numpy unavailable.
    """
    states = list(range(1 << N))
    for (i, j) in comps:
        bi = 1 << i
        bj = 1 << j
        swap = bi | bj
        new_states = []
        for s in states:
            if (s & bi) and not (s & bj):
                new_states.append(s ^ swap)
            else:
                new_states.append(s)
        states = new_states
    for s in states:
        if s & (s + 1):
            return False
    return True


def _verify(comps):
    """Dispatch to the fastest available verifier."""
    try:
        return _verify_numpy(comps)
    except ImportError:
        return _verify_list(comps)


def _depth_of_network(comps):
    if not comps:
        return 0
    depth = [0] * N
    for (i, j) in comps:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)


def _greedy_remove(comps, deadline):
    """
    Repeatedly try to remove each comparator (from end to start) and check
    if the network still sorts. This is a local minimum finder for size.
    """
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
                break
    return comps


def _local_search_bitonic(comps, rng, deadline):
    """
    Local search guided by bitonic pattern completion.
    Mutations prefer structured changes that maintain layer monotonicity.
    """
    import numpy as np
    comps = list(comps)
    best_comps = list(comps)
    best_len = len(comps)

    T = 4.0
    T_min = 0.05
    cool_rate = 0.997
    stall = 0
    max_stall = 400

    while time.time() < deadline:
        n = len(comps)
        r = rng.random()

        if r < 0.25:
            # Delete a comparator
            if n < 2:
                continue
            idx = int(rng.randint(0, n))
            cand = comps[:idx] + comps[idx + 1:]
        elif r < 0.40:
            # Swap two comparators
            if n < 2:
                continue
            i = int(rng.randint(0, n - 1))
            j = int(rng.randint(0, n - 1))
            if i == j:
                continue
            cand = comps[:]
            cand[i], cand[j] = cand[j], cand[i]
        elif r < 0.50:
            # Move a comparator
            if n < 2:
                continue
            i = int(rng.randint(0, n))
            j = int(rng.randint(0, n))
            if i == j:
                continue
            cand = comps[:]
            item = cand.pop(i)
            cand.insert(j, item)
        elif r < 0.60:
            # Replace a comparator with a nearby-wire pair (bitonic-guided)
            if n < 1:
                continue
            idx = int(rng.randint(0, n))
            a = int(rng.randint(0, N - 1))
            b = int(rng.randint(a + 1, min(a + 4, N)))
            cand = comps[:]
            cand[idx] = (a, b)
        elif r < 0.70:
            # Reverse a small window
            if n < 3:
                continue
            pos = int(rng.randint(0, max(1, n - 2)))
            wlen = int(rng.randint(2, min(5, n - pos)))
            seg = comps[pos:pos + wlen]
            cand = comps[:pos] + seg[::-1] + comps[pos + wlen:]
        elif r < 0.80:
            # Insert a comparator
            if n >= MAX_COMPARATORS:
                continue
            a = int(rng.randint(0, N - 1))
            b = int(rng.randint(a + 1, N))
            pos = int(rng.randint(0, n + 1))
            cand = comps[:pos] + [(a, b)] + comps[pos:]
        else:
            # Multi-swap
            if n < 4:
                continue
            i = int(rng.randint(0, n - 1))
            j = int(rng.randint(0, n - 1))
            if i == j:
                continue
            k = int(rng.randint(0, n - 1))
            m = int(rng.randint(0, n - 1))
            if k in (i, j) or m in (i, j) or k == m:
                continue
            cand = comps[:]
            cand[i], cand[j] = cand[j], cand[i]
            cand[k], cand[m] = cand[m], cand[k]

        # Evaluate
        if _verify(cand):
            delta = len(cand) - len(comps)
            if delta <= 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                comps = cand
                if len(comps) < best_len:
                    best_len = len(comps)
                    best_comps = list(comps)
                    stall = 0
                else:
                    stall += 1
            else:
                stall += 1
        else:
            stall += 1

        # Cool
        if T > T_min:
            T *= cool_rate
        else:
            T = T_min

        # Reheat if stuck
        if stall > max_stall:
            T = max(T, 2.0)
            stall = 0

        # Periodic greedy cleanup
        if stall > 200 and time.time() < deadline - 1:
            cleaned = _greedy_remove(comps, deadline)
            if len(cleaned) < len(comps):
                comps = cleaned
                if len(comps) < best_len:
                    best_len = len(comps)
                    best_comps = list(comps)
                    stall = 0

    # Final greedy cleanup
    if time.time() < deadline - 0.5:
        final = _greedy_remove(best_comps, deadline)
        if len(final) < best_len:
            best_comps = final
            best_len = len(final)

    return best_comps


def _hierarchical_optimize(comps, deadline):
    """
    Optimize by treating the network as structured layers.
    """
    return _greedy_remove(comps, deadline)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    comps = _build_seed_network()
    if not _verify(comps):
        return comps

    best_comps = list(comps)
    best_len = len(comps)

    # Phase 1: Aggressive greedy deletion
    comps = _greedy_remove(comps, deadline)
    if len(comps) < best_len:
        best_len = len(comps)
        best_comps = list(comps)

    # Phase 2: Local search with multiple restarts using bitonic-guided mutations
    import numpy as np
    max_restarts = 12
    restarts = 0

    while time.time() < deadline - 3 and restarts < max_restarts:
        restarts += 1
        remaining = deadline - time.time()
        sub_deadline = time.time() + remaining / (max_restarts - restarts + 1)
        sub_deadline = min(sub_deadline, deadline - 1)

        if time.time() > sub_deadline:
            break

        seed = 42 + restarts * 1000
        rng_r = np.random.default_rng(seed)
        cand = _local_search_bitonic(best_comps, rng_r, sub_deadline)
        if len(cand) < best_len:
            best_len = len(cand)
            best_comps = list(cand)

        if best_len <= 85 and time.time() > start_time + DEADLINE_SECONDS * 0.8:
            break

    # Phase 3: Final greedy polish
    if time.time() < deadline - 0.5:
        final = _greedy_remove(best_comps, deadline)
        if len(final) < best_len:
            best_comps = final
            best_len = len(final)

    return best_comps
