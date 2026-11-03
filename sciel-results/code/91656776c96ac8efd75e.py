import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    total = q ** n
    # radius=1 ball size for q-ary length n
    ball_size = 1 + n * (q - 1)

    # --- Precompute the full ball table: balls[w] = indices of words within dist<=radius of w ---
    # Use vectorized construction
    digits_all = np.zeros((total, n), dtype=np.int32)
    for pos in range(n):
        digits_all[:, pos] = (np.arange(total, dtype=np.int64) // (q ** (n - 1 - pos))) % q

    balls = np.empty((total, ball_size), dtype=np.int32)
    balls[:, 0] = np.arange(total, dtype=np.int32)
    col = 1
    for pos in range(n):
        orig = digits_all[:, pos].astype(np.int64)
        for nv in range(q):
            if nv == 0:
                continue
            # we need all words that differ at pos: for each nv in 0..q-1, nv != orig
            # Build mask of nv values
        # Redo: for each position, for each replacement value
        # We'll just loop nv
        for nv in range(q):
            # mask: positions where orig != nv
            mask = (orig != nv)
            if not mask.any():
                continue
            # For those words, new value at pos = nv
            new_digits = digits_all[mask].copy()
            new_digits[:, pos] = nv
            # convert back to index
            vals = np.zeros(len(new_digits), dtype=np.int64)
            for p2 in range(n):
                vals = vals * q + new_digits[:, p2]
            balls[mask, col:col + mask.sum()] = vals.astype(np.int32)
            col += mask.sum()

    # --- Greedy initialization (fast, vectorized gain) ---
    covered = np.zeros(total, dtype=np.bool_)
    covered_count = 0
    codewords = []

    # Precompute: for greedy, gain of word w = ball_size - sum(covered[balls[w]])
    # We'll recompute gains periodically or each step (expensive but okay for ~500 iters)
    # Actually 6561 words * 17 ball = 111k per full gain pass, 500 passes = 55M, fine.

    while covered_count < total:
        if time.time() > deadline - 60:
            break
        # Compute gains
        covered_in_balls = covered[balls]  # (total, ball_size)
        covered_per_ball = covered_in_balls.sum(axis=1)
        gains = ball_size - covered_per_ball
        best_word = int(np.argmax(gains))
        if gains[best_word] == 0:
            break
        codewords.append(best_word)
        bp = balls[best_word]
        newly = ~covered[bp]
        if newly.any():
            covered[bp] = True
            covered_count += int(newly.sum())

    if not codewords:
        return []

    # Build coverage count
    cov_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        cov_count[balls[cw]] += 1

    cw_set = set(codewords)

    # --- Helper: check if removing cw keeps feasibility ---
    def can_remove(cw):
        bp = balls[cw]
        return (cov_count[bp] > 1).all()

    # --- Phase 1: remove all redundant codewords ---
    changed = True
    while changed and time.time() < deadline - 55:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline - 55:
                break
            cw = codewords[i]
            if can_remove(cw):
                cov_count[balls[cw]] -= 1
                cw_set.discard(cw)
                codewords.pop(i)
                changed = True
                break

    # --- Precompute neighbor lists for fast candidate generation ---
    # neighbors[w] = list of words at Hamming distance 1 from w (not including w)
    neighbors = np.empty((total, n * (q - 1)), dtype=np.int32)
    col = 0
    for pos in range(n):
        orig = digits_all[:, pos].astype(np.int64)
        for nv in range(q):
            mask = (orig != nv)
            if not mask.any():
                continue
            new_digits = digits_all[mask].copy()
            new_digits[:, pos] = nv
            vals = np.zeros(len(new_digits), dtype=np.int64)
            for p2 in range(n):
                vals = vals * q + new_digits[:, p2]
            neighbors[mask, col:col + mask.sum()] = vals.astype(np.int32)
            col += mask.sum()

    # --- Simulated Annealing with pair-swap moves ---
    rng = random.Random(12345)
    current_size = len(codewords)
    best_size = current_size
    best_codewords = list(codewords)
    best_cov = cov_count.copy()

    T0 = 10.0
    Tend = 0.01
    # Total SA iterations budget
    sa_start = time.time()
    sa_deadline = deadline - 10

    # We'll do many iterations; each iteration: pick a random cw to remove, find a candidate to add
    # Accept if size decreases; accept size-increase with prob exp(-delta/T)

    iteration = 0
    while time.time() < sa_deadline:
        iteration += 1
        # Update temperature (geometric cooling)
        # Estimate total iterations: let's do ~50000 iters over the time budget
        # For now, use a fixed schedule based on elapsed fraction
        elapsed_frac = (time.time() - sa_start) / max(1, sa_deadline - sa_start)
        T = T0 * (Tend / T0) ** elapsed_frac

        if len(codewords) == 0:
            break

        # Pick random codeword to remove
        i = rng.randrange(len(codewords))
        cw_i = codewords[i]

        if cw_i not in cw_set:
            continue

        bp_i = balls[cw_i]

        # Try to remove cw_i
        # Check feasibility after removal
        cov_after_remove = cov_count[bp_i] - 1
        if (cov_after_remove < 0).any():
            # Can't remove without breaking coverage; skip
            continue

        # Points that become uncovered after removal: those where cov_count == 1 and in bp_i
        unique_pts = bp_i[cov_count[bp_i] == 1]

        if len(unique_pts) == 0:
            # cw_i is redundant; just remove it
            cov_count[bp_i] -= 1
            cw_set.discard(cw_i)
            codewords.pop(i)
            current_size = len(codewords)
            if current_size < best_size:
                best_size = current_size
                best_codewords = list(codewords)
                best_cov = cov_count.copy()
            continue

        # Need to find a candidate that covers all unique_pts
        # Candidate pool: neighbors of cw_i + unique_pts themselves + neighbors of unique_pts
        # But limit pool size for speed
        candidates = set()
        # Add cw_i's neighbors
        nb_i = neighbors[cw_i]
        for nb in nb_i:
            candidates.add(int(nb))
        # Add unique_pts and their neighbors
        for up in unique_pts:
            up_int = int(up)
            candidates.add(up_int)
            nb_up = neighbors[up_int]
            for nb in nb_up:
                candidates.add(int(nb))
            if len(candidates) > 200:
                break

        if not candidates:
            continue

        # Filter: not in cw_set, and covers all unique_pts
        best_cand = None
        best_score = -1

        # For speed, limit candidates checked
        cand_list = list(candidates)
        if len(cand_list) > 150:
            rng.shuffle(cand_list)
            cand_list = cand_list[:150]

        for cand in cand_list:
            if cand in cw_set:
                continue
            # Check if cand covers all unique_pts
            # dist(cand, up) <= 1 for all up in unique_pts
            covers_all = True
            for up in unique_pts:
                # Compute Hamming distance
                # Use precomputed digits
                d = np.sum(digits_all[cand] != digits_all[up])
                if d > 1:
                    covers_all = False
                    break
            if not covers_all:
                continue

            # Score: how many points in ball(cand) would be uncovered after removing cw_i
            # i.e., cov_count[p] - (1 if p in bp_i else 0) == 0
            bp_cand = balls[cand]
            # Effective coverage after removing cw_i:
            # For p in bp_cand: eff = cov_count[p] - (1 if p in bp_i else 0)
            # We need to count how many p in bp_cand have eff == 0
            # Create a mask for bp_i membership
            in_bp_i = np.zeros(total, dtype=np.bool_)
            in_bp_i[bp_i] = True
            eff = cov_count[bp_cand] - in_bp_i[bp_cand].astype(np.int32)
            score = int((eff == 0).sum())

            if score > best_score:
                best_score = score
                best_cand = cand

        if best_cand is None:
            # No valid replacement; skip this move
            continue

        # Perform the swap
        cov_count[bp_i] -= 1
        cw_set.discard(cw_i)
        codewords.pop(i)

        bp_new = balls[best_cand]
        cov_count[bp_new] += 1
        cw_set.add(best_cand)
        codewords.append(best_cand)
        current_size = len(codewords)

        # Accept/reject based on size change
        delta_size = current_size - best_size  # we track vs best, but for SA we should track vs previous
        # Actually for SA, we compare to the previous state size
        # Let's track prev_size
        # Since we always do exactly one swap (remove 1, add 1), size stays the same!
        # Wait: remove 1, add 1 => size unchanged.
        # To change size, we need to sometimes just remove (if redundant) or not add.
        # The redundancy case above reduces size.
        # For the swap case, size is constant. So SA here is really about finding better
        # configurations at the same size, which enables future removals.

        if current_size < best_size:
            best_size = current_size
            best_codewords = list(codewords)
            best_cov = cov_count.copy()

    # --- Restore best state ---
    if best_size < len(codewords):
        codewords = best_codewords
        cov_count = best_cov
        cw_set = set(codewords)

    # --- Final cleanup: remove redundant codewords ---
    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline - 3:
                break
            cw = codewords[i]
            if cw in cw_set and can_remove(cw):
                cov_count[balls[cw]] -= 1
                cw_set.discard(cw)
                codewords.pop(i)
                changed = True
                break

    # Convert to word lists
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    result = [to_word(w) for w in codewords]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
