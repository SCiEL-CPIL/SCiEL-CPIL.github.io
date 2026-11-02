"""Portfolio SEARCH approach for K_3(6,1): greedy + local search + perturbation."""
import itertools
import random
import time
import numpy as np

# Precompute for the small space 3^6 = 729.
_SPACE = [3] * 6  # length n placeholder

def _build_space(q, n, radius):
    # Return:
    #   words: list of tuples, length q**n, in lex order
    #   sphere: list of lists; sphere[i] = list of indices within Hamming distance radius of words[i]
    #   sphere_arr: numpy array of shape (total, sphere_size) with word indices
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    sphere = []
    for w in words:
        s = [word_to_idx[w]]
        # radius 1 neighbors: change one coordinate to a different value
        for pos in range(n):
            cur = w[pos]
            for v in range(q):
                if v != cur:
                    w2 = list(w)
                    w2[pos] = v
                    s.append(word_to_idx[tuple(w2)])
        sphere.append(s)
    # Build numpy array for fast coverage counting
    sphere_size = len(sphere[0])
    sphere_arr = np.array(sphere, dtype=np.int32)
    return words, sphere, sphere_arr, total

def _verify(code_idx, sphere_arr, total):
    # returns True iff every word index is covered by at least one codeword
    if not code_idx:
        return total == 0
    counts = np.zeros(total, dtype=np.int32)
    for ci in code_idx:
        counts[sphere_arr[ci]] += 1
    return np.all(counts > 0)

def _coverage_counts(code_idx, sphere_arr, total):
    counts = np.zeros(total, dtype=np.int32)
    for ci in code_idx:
        counts[sphere_arr[ci]] += 1
    return counts

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 115.0  # safety margin under 120/150s limits

    words, sphere, sphere_arr, total = _build_space(q, n, radius)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Seed: tetracode direct sum -> 81 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_idx = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            seed_idx.append(word_to_idx[tuple(w)])
    best_code = sorted(set(seed_idx))
    best_size = len(best_code)

    def local_search(code_idx):
        # Repeatedly delete any redundant codeword until fixed point.
        counts = _coverage_counts(list(code_idx), sphere_arr, total)
        code = sorted(set(code_idx))
        changed = True
        while changed:
            changed = False
            new_code = []
            for ci in code:
                # check redundancy: after removing ci, every word in sphere[ci]
                # must still have count > 1 (i.e., covered by some other)
                if np.all(counts[sphere_arr[ci]] > 1):
                    counts[sphere_arr[ci]] -= 1
                    changed = True
                else:
                    new_code.append(ci)
            code = new_code
        return code

    def greedy_from_uncovered(uncovered_set):
        # uncovered_set: set of word indices currently uncovered
        # returns list of centers that cover all of them (additions)
        chosen = []
        uncov_mask = np.zeros(total, dtype=bool)
        for w in uncovered_set:
            uncov_mask[w] = True
        uncov_count = len(uncovered_set)
        while uncov_count > 0:
            # Vectorized: for each center, count how many uncovered words it covers
            # sphere_arr has shape (total, sphere_size)
            # For each center c, count = sum of uncov_mask[sphere_arr[c]]
            counts_per_center = np.array([uncov_mask[sphere_arr[c]].sum() for c in range(total)])
            best_c = int(np.argmax(counts_per_center))
            if counts_per_center[best_c] <= 0:
                break
            chosen.append(best_c)
            # Update uncov_mask
            newly_covered = sphere_arr[best_c]
            still_uncov = uncov_mask[newly_covered]
            uncov_count -= int(still_uncov.sum())
            uncov_mask[newly_covered] = False
        return chosen

    def repair_and_local_search(code_idx, uncovered):
        # add centers to cover uncovered, then local-search
        added = greedy_from_uncovered(uncovered)
        combined = list(code_idx) + added
        return local_search(combined)

    def swap_local_search(code_idx):
        # Try to replace one codeword with another to reduce size or maintain coverage
        # This is a limited swap: try removing each codeword, then try adding back
        # a different one that covers the gap with fewer additional words needed
        code = sorted(set(code_idx))
        counts = _coverage_counts(code, sphere_arr, total)
        improved = True
        while improved:
            improved = False
            for i in range(len(code)):
                ci = code[i]
                # What words become uncovered if we remove ci?
                uncovered_words = sphere_arr[ci][counts[sphere_arr[ci]] <= 1]
                if len(uncovered_words) == 0:
                    # Redundant - just remove
                    counts[sphere_arr[ci]] -= 1
                    code = [c for j, c in enumerate(code) if j != i]
                    improved = True
                    break
                # Try to find a single center that covers all uncovered_words
                # and is not already in code
                in_code = set(code)
                best_c = -1
                best_extra = 10**9
                for c in range(total):
                    if c in in_code:
                        continue
                    # How many of uncovered_words does c cover?
                    covered = sphere_arr[c]
                    cnt = int(np.sum(np.isin(uncovered_words, covered)))
                    if cnt == len(uncovered_words):
                        # c covers all uncovered words
                        # Check: after adding c and removing ci, is everything still covered?
                        # The words in sphere[ci] that had count==1 are covered by c
                        # But we need to make sure no other word loses coverage
                        # Actually we're just swapping, so check net effect
                        # Words in sphere[ci] with count==1: covered by c (by assumption)
                        # Words in sphere[c] that weren't covered: now covered (fine)
                        # Words in sphere[ci] with count>1: still covered (fine)
                        # So this swap is valid!
                        counts[sphere_arr[ci]] -= 1
                        counts[sphere_arr[c]] += 1
                        code[i] = c
                        improved = True
                        break
                if improved:
                    break
        return code

    rng = random.Random(20240517)

    # Pass 1: local search on the 81-word seed
    cand = local_search(best_code)
    if len(cand) < best_size:
        best_size = len(cand)
        best_code = cand

    # Pass 2: full greedy from scratch (random tie-breaks) + local search, multiple times
    for trial in range(100):
        if time.time() > DEADLINE - 40:
            break
        # Randomized greedy: pick centers in random order, but choose the one
        # that covers the most uncovered words
        uncov_mask = np.ones(total, dtype=bool)
        used_centers = set()
        code = []
        while uncov_mask.any() and len(code) < 95:
            # Compute for each unused center how many uncovered words it covers
            # Use vectorized approach
            best_c = -1
            best_cnt = -1
            # Sample a subset of candidates for speed, or use all
            candidates = [c for c in range(total) if c not in used_centers]
            if not candidates:
                break
            # For speed, if too many candidates, sample
            if len(candidates) > 200:
                sampled = rng.sample(candidates, 200)
            else:
                sampled = candidates
            for c in sampled:
                cnt = int(np.sum(uncov_mask[sphere_arr[c]]))
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt <= 0:
                break
            code.append(best_c)
            used_centers.add(best_c)
            uncov_mask[sphere_arr[best_c]] = False
        if not uncov_mask.all() == False and not uncov_mask.any():
            code2 = local_search(code)
            if len(code2) < best_size:
                best_size = len(code2)
                best_code = code2

    # Pass 3: perturbation from best_code with swap-based local search
    iterations = 0
    while time.time() < DEADLINE - 10:
        iterations += 1
        # copy
        code = list(best_code)
        k = len(code)
        if k < 2:
            break
        # pick a random center to drop (with some probability pick two)
        drop_idx = rng.randrange(k)
        dropped = code[drop_idx]
        new_code = [c for j, c in enumerate(code) if j != drop_idx]
        # find uncovered after drop
        counts = _coverage_counts(new_code, sphere_arr, total)
        uncovered = np.where(counts == 0)[0].tolist()
        if not uncovered:
            # Still valid without it!
            if len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code
            continue
        repaired = repair_and_local_search(new_code, uncovered)
        if len(repaired) < best_size:
            best_size = len(repaired)
            best_code = repaired

        # every few iterations try dropping two centers
        if iterations % 4 == 0 and k >= 2:
            d1 = rng.randrange(k)
            d2 = rng.randrange(k)
            if d1 == d2:
                d2 = (d2 + 1) % k
            new_code = [c for j, c in enumerate(code) if j != d1 and j != d2]
            counts = _coverage_counts(new_code, sphere_arr, total)
            uncovered = np.where(counts == 0)[0].tolist()
            if not uncovered:
                if len(new_code) < best_size:
                    best_size = len(new_code)
                    best_code = new_code
                continue
            repaired = repair_and_local_search(new_code, uncovered)
            if len(repaired) < best_size:
                best_size = len(repaired)
                best_code = repaired

        # Every 8th iteration: try swap-based local search on current best
        if iterations % 8 == 0:
            swapped = swap_local_search(best_code)
            if len(swapped) < best_size:
                best_size = len(swapped)
                best_code = swapped

    # Final: ensure best_code is valid (it should be by construction), return it
    if not _verify(best_code, sphere_arr, total):
        # fallback to seed
        best_code = sorted(set(seed_idx))

    # Convert indices -> lists of ints
    return [list(words[i]) for i in sorted(set(best_code))]
