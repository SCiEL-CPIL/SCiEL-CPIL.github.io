"""
Minimum maximal no-isosceles set on [0,16)^2.

Combined strategy:
- Bitset-based conflict representation (inspiration 2's `_iso_bits` / blocked mask).
- Fast NumPy precomputation of the full isosceles third-point table.
- Greedy selection that picks the point minimizing remaining candidates (current program).
- Incremental `blocked` mask so maximality is an O(1) bit op.
- Local search with both shrink (remove) and swap moves to escape local minima.
- Multiple randomized restarts with a hard wall-clock deadline guard.
"""

import random
import time
import itertools
import numpy as np

N = 16
TOTAL = N * N
ALL_POINTS = list(itertools.product(range(N), repeat=2))

# ---------- Precomputation ----------
# iso_third[i, j, k] is True iff {i, j, k} form an isosceles triangle (distinct points).
# Stored as a flat int8 array for fast indexing.
_pts = np.array(ALL_POINTS, dtype=np.int32)  # (TOTAL, 2)
_diff = _pts[:, np.newaxis, :] - _pts[np.newaxis, :, :]  # (TOTAL, TOTAL, 2)
_dist2 = np.sum(_diff ** 2, axis=2)  # (TOTAL, TOTAL)

# For each pair (i, j), the set of k such that {i,j,k} isosceles.
# iso_third_flat: (TOTAL, TOTAL, TOTAL) bool array.
iso_third_flat = np.zeros((TOTAL, TOTAL, TOTAL), dtype=bool)
# d_ij = dist2[i, j], d_ik = dist2[i, k], d_jk = dist2[j, k]
# isosceles iff d_ik == d_jk (apex k) OR d_ik == d_ij (apex i) OR d_jk == d_ij (apex j)
_dij = _dist2[:, :, np.newaxis]      # (TOTAL, TOTAL, 1)
_dik = _dist2[:, np.newaxis, :]      # (TOTAL, 1, TOTAL)
_djk = _dist2[np.newaxis, :, :]      # (TOTAL, TOTAL, 1) -> need (1, TOTAL, TOTAL)
_djk = _dist2.T[np.newaxis, :, :]    # (1, TOTAL, TOTAL)
iso_third_flat = (
    (_dik == _djk) |
    (_dik == _dij) |
    (_djk == _dij)
)
# Exclude k == i and k == j (degenerate "triangle")
for _i in range(TOTAL):
    iso_third_flat[_i, :, _i] = False
    iso_third_flat[_i, _i, :] = False

# Also make a bitset version for fast OR operations:
# iso_bits[i, j] is a Python int where bit k is set iff iso_third_flat[i, j, k].
# Build incrementally row by row to keep memory/time reasonable.
iso_bits = [[0] * TOTAL for _ in range(TOTAL)]
for _i in range(TOTAL):
    for _j in range(TOTAL):
        if _i == _j:
            continue
        # iso_third_flat[_i, _j, :] is a bool array of length TOTAL
        _row = iso_third_flat[_i, _j]
        # Convert to int bitset
        _bits = 0
        # Use numpy to find set bits
        _set_indices = np.nonzero(_row)[0]
        for _k in _set_indices:
            _bits |= (1 << int(_k))
        iso_bits[_i][_j] = _bits

# ---------- Helpers ----------

def _idx_point(i):
    return (i // N, i % N)


def _is_maximal(chosen, blocked_mask, chosen_mask):
    """Check maximality using blocked mask. O(1) bit op."""
    all_mask = (1 << TOTAL) - 1
    unchosen = all_mask & ~chosen_mask
    return (blocked_mask & unchosen) == unchosen


def _recompute_blocked(chosen):
    """Recompute blocked mask and chosen mask from a chosen list."""
    blocked = 0
    chosen_mask = 0
    for p in chosen:
        chosen_mask |= (1 << p)
    for idx_a in range(len(chosen)):
        a = chosen[idx_a]
        for idx_b in range(idx_a + 1, len(chosen)):
            b = chosen[idx_b]
            blocked |= iso_bits[a][b]
    blocked |= chosen_mask
    return blocked, chosen_mask


def _greedy_blocking_power(deadline, rng):
    """
    Greedy: at each step, pick the point p (from unblocked candidates) that
    minimizes the number of remaining unblocked candidates after adding p.
    Returns (chosen_list, blocked_mask, chosen_mask).
    """
    chosen = []
    blocked = 0
    chosen_mask = 0

    # Track unblocked candidates as a set of indices
    all_mask = (1 << TOTAL) - 1
    unblocked = all_mask  # all points initially unblocked

    while True:
        if time.time() > deadline:
            break

        # Find unblocked point indices
        # Extract set bits from unblocked (excluding already chosen)
        # unblocked already excludes chosen since we set blocked |= (1<<p) when choosing
        candidates = []
        tmp = unblocked
        while tmp:
            # lowest set bit
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            tmp ^= lsb

        if not candidates:
            break

        # For each candidate p, compute how many candidates remain unblocked
        # after adding p.
        # New blocked = blocked | (OR over q in chosen of iso_bits[p][q]) | (1<<p)
        # A candidate q (q != p, q in unblocked) remains if not in new_blocked.
        best_p = -1
        best_remaining = TOTAL + 1
        best_new_blocked = 0
        best_new_chosen_mask = 0

        # Precompute the OR of iso_bits[p][q] for q in chosen
        # For each p, new_block_from_p = OR_{q in chosen} iso_bits[p][q]
        # We can compute this once per p.
        for p in candidates:
            # new_block_from_p
            nbp = 0
            for q in chosen:
                nbp |= iso_bits[p][q]
            new_blocked = blocked | nbp | (1 << p)
            # Count candidates q (q != p, q in unblocked) that are NOT in new_blocked
            # remaining = unblocked & ~new_blocked & ~(1<<p)  (p is in new_blocked already)
            remaining_mask = unblocked & ~new_blocked
            remaining_count = remaining_mask.bit_count()
            if remaining_count < best_remaining:
                best_remaining = remaining_count
                best_p = p
                best_new_blocked = new_blocked
                best_new_chosen_mask = chosen_mask | (1 << p)

        if best_p == -1:
            break

        chosen.append(best_p)
        blocked = best_new_blocked
        chosen_mask = best_new_chosen_mask
        unblocked = all_mask & ~blocked

    return chosen, blocked, chosen_mask


def _greedy_random(order, deadline, rng):
    """
    Greedy in a given order: add p if not blocked.
    Returns (chosen, blocked, chosen_mask).
    """
    chosen = []
    blocked = 0
    chosen_mask = 0
    for p in order:
        if time.time() > deadline:
            break
        if (blocked >> p) & 1:
            continue
        # Add p
        nbp = 0
        for q in chosen:
            nbp |= iso_bits[p][q]
        blocked |= nbp | (1 << p)
        chosen_mask |= (1 << p)
        chosen.append(p)
    return chosen, blocked, chosen_mask


def _local_search(chosen, blocked, chosen_mask, deadline, rng):
    """
    Local search: try removing points and swapping points to reduce size
    while maintaining maximality.
    Returns improved (chosen, blocked, chosen_mask).
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(chosen)

        # --- Shrink: try removing each point ---
        for idx in range(n):
            if time.time() > deadline:
                break
            p = chosen[idx]
            new_chosen = chosen[:idx] + chosen[idx + 1:]
            new_blocked, new_chosen_mask = _recompute_blocked(new_chosen)
            if _is_maximal(new_chosen, new_blocked, new_chosen_mask):
                chosen = new_chosen
                blocked = new_blocked
                chosen_mask = new_chosen_mask
                improved = True
                break

        if improved:
            continue

        # --- Swap: try replacing p with q (q not in chosen, q not blocked) ---
        # Only try if it might reduce size? No, swap keeps same size.
        # But swap can enable further shrinkage. Try a few random swaps.
        # For efficiency, limit swaps per iteration.
        n = len(chosen)
        if n == 0:
            break

        all_mask = (1 << TOTAL) - 1
        unblocked = all_mask & ~blocked

        # Try a limited number of random swaps
        swap_attempts = 0
        max_swap_attempts = max(10, n * 2)
        while swap_attempts < max_swap_attempts and time.time() < deadline:
            swap_attempts += 1
            # Pick a random point in chosen to remove
            idx = rng.randrange(n)
            p = chosen[idx]
            # Pick a random unblocked point to add
            # Extract a random set bit from unblocked
            tmp = unblocked
            if tmp == 0:
                break
            # Random bit selection
            # Use a random number to pick a bit
            # Simple: iterate to a random position
            # For efficiency, just pick the lowest set bit with some randomization
            # Shift by a random amount
            shift = rng.randrange(TOTAL)
            # We'll just pick a random candidate from the unblocked set
            # Build a small list of candidates (up to 50)
            candidates = []
            tmp2 = unblocked
            count = 0
            while tmp2 and count < 200:
                lsb = tmp2 & (-tmp2)
                idx2 = lsb.bit_length() - 1
                candidates.append(idx2)
                tmp2 ^= lsb
                count += 1
            if not candidates:
                break
            q = rng.choice(candidates)

            # Try swap: remove p, add q
            new_chosen = chosen[:idx] + [q] + chosen[idx + 1:]
            new_blocked, new_chosen_mask = _recompute_blocked(new_chosen)
            if _is_maximal(new_chosen, new_blocked, new_chosen_mask):
                # Check if this swap enables further shrinkage
                # Try to shrink the new set
                shrunken = _try_shrink(new_chosen, new_blocked, new_chosen_mask, deadline, rng)
                if shrunken is not None:
                    chosen, blocked, chosen_mask = shrunken
                    improved = True
                    break
                # Even without immediate shrink, accept the swap if it leads
                # to a set where we can shrink
                # Actually, just accept if the new set is maximal (same size)
                # but we only want to accept if it helps. Let's accept and
                # continue the local search loop.
                chosen = new_chosen
                blocked = new_blocked
                chosen_mask = new_chosen_mask
                improved = True
                break

    return chosen, blocked, chosen_mask


def _try_shrink(chosen, blocked, chosen_mask, deadline, rng):
    """Try to shrink the set by removing points. Returns (chosen, blocked, chosen_mask) or None."""
    n = len(chosen)
    for idx in range(n):
        if time.time() > deadline:
            break
        p = chosen[idx]
        new_chosen = chosen[:idx] + chosen[idx + 1:]
        new_blocked, new_chosen_mask = _recompute_blocked(new_chosen)
        if _is_maximal(new_chosen, new_blocked, new_chosen_mask):
            return new_chosen, new_blocked, new_chosen_mask
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 310.0  # safety margin before 330s kill

    rng = random.Random(42)
    all_idx = list(range(TOTAL))

    best_chosen = None
    best_blocked = 0
    best_chosen_mask = 0
    best_size = TOTAL + 1

    # --- Phase 1: Greedy blocking power restarts ---
    restart_count = 0
    while time.time() < deadline - 30.0:
        restart_count += 1
        chosen, blocked, chosen_mask = _greedy_blocking_power(deadline - 20.0, rng)

        if not chosen:
            continue

        # Check maximality
        if not _is_maximal(chosen, blocked, chosen_mask):
            # Shouldn't happen if greedy is correct, but just in case
            # Try to fix by adding points? For now, skip.
            continue

        # Local search
        chosen, blocked, chosen_mask = _local_search(chosen, blocked, chosen_mask, deadline - 10.0, rng)

        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = chosen[:]
            best_blocked = blocked
            best_chosen_mask = chosen_mask

        if best_size <= 10:
            break

    # --- Phase 2: Random-order greedy restarts (for diversity) ---
    while time.time() < deadline - 20.0:
        order = all_idx[:]
        rng.shuffle(order)
        chosen, blocked, chosen_mask = _greedy_random(order, deadline - 15.0, rng)

        if not chosen:
            continue

        if not _is_maximal(chosen, blocked, chosen_mask):
            continue

        chosen, blocked, chosen_mask = _local_search(chosen, blocked, chosen_mask, deadline - 10.0, rng)

        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = chosen[:]
            best_blocked = blocked
            best_chosen_mask = chosen_mask

        if best_size <= 10:
            break

    # --- Fallback ---
    if best_chosen is None:
        # Simple row-major greedy
        chosen, blocked, chosen_mask = _greedy_random(all_idx, deadline - 5.0, rng)
        best_chosen = chosen

    return [_idx_point(i) for i in best_chosen]
