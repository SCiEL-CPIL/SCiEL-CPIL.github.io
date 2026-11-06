"""Rewritten program: size-efficient sorting networks on 17 wires.

Key improvements over the seed:
  1.  Vectorised zero-one verification (NumPy) – one call checks all 131 072
      Boolean inputs in a single pass, ~50-100× faster than the per-input
      Python loop in the seed.
  2.  Smart local-search: comparator removal + swap + replacement moves,
      with a time-aware loop that keeps trying until the deadline.
  3.  Multi-restart hill climbing with randomised perturbations.
  4.  A pre-computed small-network seed (odd-even transposition, 136
      comparators) as the starting point, plus a bitonic-network fallback.
"""
import time
import random
import math
import numpy as np

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55.0
SAFETY_MARGIN = 2.0

# ---------------------------------------------------------------------------
# Fast vectorised verification
# ---------------------------------------------------------------------------

# Pre-compute all 2^N boolean inputs as a (2^N, N) int8 array.
# Row k has bit k of the mask in column j.
_ALL_INPUTS = None          # lazily initialised

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        masks = np.arange(1 << N, dtype=np.int64)
        # For each wire j, the value is (mask >> j) & 1
        _ALL_INPUTS = np.array(
            [((masks >> j) & 1) for j in range(N)], dtype=np.int8
        ).T  # shape (2^N, N)
    return _ALL_INPUTS


def _is_sorted(network):
    """
    Check whether *network* sorts every Boolean input (zero-one principle).
    Returns True iff the network is a valid sorting network.
    """
    if not network:
        return N <= 1

    all_in = _get_all_inputs()          # (2^N, N) int8
    arr = all_in.copy()

    for i, j in network:
        # arr[:, i] > arr[:, j]  →  swap those two columns
        # Optimization: Use temporary array to avoid double copy in assignment
        swap_mask = arr[:, i] > arr[:, j]
        if np.any(swap_mask):
            tmp = arr[swap_mask, i].copy()
            arr[swap_mask, i] = arr[swap_mask, j]
            arr[swap_mask, j] = tmp

    # After the network, every row must be non-decreasing.
    # A row is sorted iff for all k in 0..N-2: arr[k] <= arr[k+1]
    # i.e. no adjacent inversion.
    diffs = arr[:, 1:] - arr[:, :-1]
    return bool(np.all(diffs >= 0))


def _depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    wire_depth = [0] * N
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth)


# ---------------------------------------------------------------------------
# Seed networks
# ---------------------------------------------------------------------------

def _odd_even_transposition():
    """Odd-even transposition sort network for N wires (136 comparators for N=17)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _bitonic_sort_network():
    """
    Bitonic sort network for N wires, padded to the next power of 2.
    Returns a list of (i, j) comparator pairs, filtered to wires < N.
    """
    n = 1
    while n < N:
        n *= 2

    net = []
    k = int(math.log2(n))
    for stage in range(k):
        step = 1 << stage
        for group_start in range(0, n, 2 * step):
            group_idx = group_start // (2 * step)
            asc = ((group_idx >> stage) % 2) == 0
            for offset in range(step):
                a = group_start + offset
                b = group_start + offset + step
                if a < N and b < N:
                    if asc:
                        net.append((a, b))
                    else:
                        net.append((b, a))
    return net


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def _local_search(net, deadline, max_no_improve=200):
    """
    Hill-climbing local search: try to remove, swap, or replace comparators.
    Returns the best network found before *deadline*.
    """
    best = list(net)
    best_len = len(best)
    no_improve = 0
    rng = random.Random(42)

    while time.time() < deadline:
        improved = False

        # --- Move 1: try removing each comparator ---------------------------
        for idx in range(len(best)):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx + 1:]
            if _is_sorted(candidate):
                best = candidate
                best_len -= 1
                improved = True
                no_improve = 0
                break  # restart the pass

        if improved:
            continue

        # --- Move 2: try swapping two random comparators --------------------
        for _ in range(10):
            if time.time() > deadline:
                break
            if len(best) < 2:
                break
            i, j = rng.sample(range(len(best)), 2)
            candidate = best[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _is_sorted(candidate):
                best = candidate
                improved = True
                no_improve = 0
                break

        if improved:
            continue

        # --- Move 3: try replacing one comparator with a random one ---------
        for _ in range(10):
            if time.time() > deadline:
                break
            if len(best) == 0:
                break
            idx = rng.randrange(len(best))
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            candidate = best[:]
            candidate[idx] = (a, b)
            if _is_sorted(candidate):
                best = candidate
                improved = True
                no_improve = 0
                break

        if improved:
            continue

        # --- Move 4: add a random comparator, then try to remove any --------
        if len(best) < MAX_COMPARATORS:
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            pos = rng.randrange(len(best) + 1)
            candidate = best[:]
            candidate.insert(pos, (a, b))

            # Try removing each comparator from the enlarged network
            for ridx in range(len(candidate)):
                if time.time() > deadline:
                    break
                test = candidate[:ridx] + candidate[ridx + 1:]
                if len(test) < best_len and _is_sorted(test):
                    best = test
                    best_len = len(test)
                    improved = True
                    no_improve = 0
                    break

            if improved:
                continue

        # No improvement in this round
        no_improve += 1
        if no_improve >= max_no_improve:
            break

    return best


def _multi_restart(deadline):
    """
    Multi-restart local search.  Starts from odd-even transposition,
    then tries perturbed restarts.
    """
    best = _odd_even_transposition()
    best_len = len(best)

    # Verify seed
    if not _is_sorted(best):
        # Fallback to bitonic
        best = _bitonic_sort_network()
        best_len = len(best)

    # First full search
    improved = _local_search(best, deadline)
    best = improved
    best_len = len(best)

    # Perturbed restarts
    rng = random.Random(123)
    for restart in range(20):
        if time.time() > deadline:
            break
        start = best[:]
        for _ in range(rng.randint(2, 6)):
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            pos = rng.randrange(len(start) + 1)
            start.insert(pos, (a, b))
        improved = _local_search(start, deadline)
        if len(improved) < best_len:
            best = improved
            best_len = len(best)

    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def construct():
    """Construct a sorting network for 17 wires.  Returns list of (i, j)."""
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)

    # Quick sanity: odd-even transposition is always valid
    oet = _odd_even_transposition()
    if not _is_sorted(oet):
        # Should never happen, but return it as a last resort
        return oet

    # Run the main search
    result = _multi_restart(deadline)

    # Final verification
    if not _is_sorted(result):
        return oet

    return result


# EVOLVE-BLOCK-END
