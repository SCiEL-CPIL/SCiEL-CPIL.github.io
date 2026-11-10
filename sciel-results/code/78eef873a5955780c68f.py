"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill


def _compute_max_discrepancy(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in [1, n]. Returns the max absolute value.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons always give |sum| = 1

    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                m = np.max(np.abs(csum))
                if m > max_disc:
                    max_disc = m
    return max_disc


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (mapped to ±1)."""
    signs = np.ones(n, dtype=np.int32)
    for i in range(n):
        # Count number of 1-bits in i
        v = i
        bits = 0
        while v:
            bits += v & 1
            v >>= 1
        if bits % 2 == 1:
            signs[i] = -1
    return signs


def _greedy_construct():
    """
    Greedy construction: assign signs one by one, choosing the sign that
    minimizes the current maximum AP discrepancy.
    """
    n = _N
    signs = np.zeros(n, dtype=np.int32)
    signs[0] = 1  # Fix first sign to break symmetry

    # Precompute all APs that end at each position
    # For efficiency, we'll track partial sums incrementally
    # Actually, let's use a simpler approach: for each new position,
    # check all APs that include this position and update the max

    # For a greedy approach, we need to evaluate the impact of each choice.
    # This is expensive, so let's use a heuristic:
    # Choose the sign that minimizes the sum of squares of all AP sums
    # that include the current position, or more simply,
    # balance the local neighborhood.

    # Better approach: use the Thue-Morse sequence as a strong base,
    # then do local search.

    return _thue_morse(n)


def _local_search_vectorized(signs_list, max_time, rng):
    """
    Vectorized local search: flip random positions, accept if it improves
    max AP discrepancy. Uses NumPy for speed.
    """
    n = _N
    signs = np.array(signs_list, dtype=np.int32)
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()

    start_time = time.time()
    no_improve_count = 0

    while time.time() - start_time < max_time:
        if time.time() > _DEADLINE - 3:
            break

        # Pick a random position to flip
        i = rng.randint(0, n - 1)
        old_val = signs[i]
        signs[i] = -signs[i]

        new_disc = _compute_max_discrepancy(signs)

        if new_disc < current_disc:
            current_disc = new_disc
            no_improve_count = 0
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        elif new_disc == current_disc:
            # Accept with some probability to escape local minima
            if rng.random() < 0.3:
                current_disc = new_disc
                no_improve_count = 0
            else:
                signs[i] = old_val
        else:
            signs[i] = old_val
            no_improve_count += 1

        # If stuck for too long, do a random restart of the local search
        if no_improve_count > 200:
            # Try a few random flips to escape
            for _ in range(5):
                j = rng.randint(0, n - 1)
                signs[j] = -signs[j]
            new_disc = _compute_max_discrepancy(signs)
            if new_disc <= current_disc:
                current_disc = new_disc
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs.copy()
            no_improve_count = 0

    return best_signs.tolist(), best_disc


def _multi_start_local_search(max_time, rng):
    """
    Run local search from multiple starting points.
    """
    start_time = time.time()
    remaining = max_time - (time.time() - start_time)

    # Starting points
    starts = []

    # 1. Thue-Morse
    tm = _thue_morse(_N)
    starts.append(tm.tolist())

    # 2. Random starts
    for _ in range(5):
        if time.time() > _DEADLINE - 5:
            break
        rand_signs = [rng.choice((-1, 1)) for _ in range(_N)]
        starts.append(rand_signs)

    best_signs = None
    best_disc = float('inf')

    for s in starts:
        if time.time() > _DEADLINE - 5:
            break

        per_start_time = min(remaining / max(1, len(starts)), 15.0)
        per_start_time = max(per_start_time, 2.0)

        ls_signs, ls_disc = _local_search_vectorized(s, per_start_time, rng)

        if ls_disc < best_disc:
            best_disc = ls_disc
            best_signs = ls_signs

        remaining = max_time - (time.time() - start_time)
        if remaining < 5:
            break

    if best_signs is None:
        # Fallback
        best_signs = tm.tolist()
        best_disc = _compute_max_discrepancy(np.array(best_signs, dtype=np.int32))

    return best_signs, best_disc


def construct():
    """
    Construct 256 signs minimizing max |sum| over all arithmetic progressions.
    Strategy: Thue-Morse initialization + vectorized local search with multiple starts.
    """
    start_time = time.time()
    rng = random.Random(42)
    np.random.seed(42)

    best_signs, best_disc = _multi_start_local_search(_DEADLINE - 5, rng)

    # Final check
    if time.time() > _DEADLINE - 2:
        return best_signs

    # One more round of local search on the best found
    remaining = _DEADLINE - 5 - (time.time() - start_time)
    if remaining > 3:
        ls_signs, ls_disc = _local_search_vectorized(best_signs, remaining, rng)
        if ls_disc <= best_disc:
            best_signs = ls_signs
            best_disc = ls_disc

    return best_signs
# EVOLVE-BLOCK-END
