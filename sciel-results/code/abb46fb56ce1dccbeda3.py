"""Portfolio SEARCH approach for K_3(6,1): greedy + local search + perturbation."""
import itertools
import random
import time
import numpy as np

def _build_space(q, n, radius):
    # Return:
    #   words: list of tuples, length q**n, in lex order
    #   sphere: list of lists; sphere[i] = list of indices within Hamming distance radius of words[i]
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
    return words, sphere, total

def _verify(code_idx, sphere, total):
    # returns True iff every word index is covered by at least one codeword
    counts = [0] * total
    for ci in code_idx:
        for w in sphere[ci]:
            counts[w] += 1
    for c in counts:
        if c == 0:
            return False
    return True

def _coverage_counts(code_idx, sphere, total):
    counts = [0] * total
    for ci in code_idx:
        for w in sphere[ci]:
            counts[w] += 1
    return counts

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 115.0  # safety margin under 120/150s limits

    words, sphere, total = _build_space(q, n, radius)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Precompute sphere as numpy arrays for faster operations
    sphere_np = [np.array(s, dtype=np.int32) for s in sphere]

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
        counts = np.zeros(total, dtype=np.int32)
        code = sorted(set(code_idx))
        for ci in code:
            counts[sphere_np[ci]] += 1
        changed = True
        while changed:
            changed = False
            new_code = []
            for ci in code:
                # check redundancy: after removing ci, every word in sphere[ci]
                # must still have count > 1 (i.e., covered by some other)
                redundant = bool(np.all(counts[sphere_np[ci]] > 1))
                if redundant:
                    counts[sphere_np[ci]] -= 1
                    changed = True
                else:
                    new_code.append(ci)
            code = new_code
        return code

    def greedy_from_uncovered(uncovered_set):
        # uncovered_set: set of word indices currently uncovered
        # returns list of centers that cover all of them (additions)
        chosen = []
        uncov = set(uncovered_set)
        while uncov:
            best_c = -1
            best_cnt = -1
            # Use numpy for faster counting
            uncov_arr = np.array(list(uncov), dtype=np.int32)
            for c in range(total):
                cnt = np.isin(sphere_np[c], uncov_arr).sum()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_cnt <= 0:
                break  # should not happen
            chosen.append(best_c)
            for w in sphere_np[best_c]:
                uncov.discard(int(w))
        return chosen

    def repair_and_local_search(code_idx, uncovered):
        # add centers to cover uncovered, then local-search
        added = greedy_from_uncovered(uncovered)
        combined = list(code_idx) + added
        return local_search(combined)

    rng = random.Random(20240517)

    # Pass 1: local search on the 81-word seed
    cand = local_search(best_code)
    if len(cand) < best_size:
        best_size = len(cand)
        best_code = cand

    # Pass 2: full greedy from scratch (random tie-breaks) + local search, multiple times
    for trial in range(80):
        if time.time() > DEADLINE - 35:
            break
        # build greedy order over centers by random weight on coverage
        centers = list(range(total))
        rng.shuffle(centers)
        uncov = set(range(total))
        code = []
        # greedy: repeatedly pick center covering most uncovered (tie by random shuffle already)
        while uncov and len(code) < 95:
            best_c = -1
            best_cnt = -1
            uncov_arr = np.array(list(uncov), dtype=np.int32)
            for c in centers:
                if not uncov:
                    break
                cnt = np.isin(sphere_np[c], uncov_arr).sum()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt <= 0:
                break
            code.append(best_c)
            for w in sphere_np[best_c]:
                uncov.discard(int(w))
            # rebuild centers order? cheaper to just re-shuffle remaining cheaply by re-listing
            centers = [c for c in centers if c != best_c]
            rng.shuffle(centers)
        if uncov:
            continue
        code2 = local_search(code)
        if len(code2) < best_size:
            best_size = len(code2)
            best_code = code2

    # Pass 3: perturbation from best_code: randomly drop one center, repair, local search
    iterations = 0
    while time.time() < DEADLINE - 20:
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
        counts = np.zeros(total, dtype=np.int32)
        for ci in new_code:
            counts[sphere_np[ci]] += 1
        uncovered = set()
        zero_mask = counts == 0
        if np.any(zero_mask):
            uncovered = set(np.where(zero_mask)[0].tolist())
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
            counts = np.zeros(total, dtype=np.int32)
            for ci in new_code:
                counts[sphere_np[ci]] += 1
            uncovered = set()
            zero_mask = counts == 0
            if np.any(zero_mask):
                uncovered = set(np.where(zero_mask)[0].tolist())
            repaired = repair_and_local_search(new_code, uncovered)
            if len(repaired) < best_size:
                best_size = len(repaired)
                best_code = repaired

    # Final: ensure best_code is valid (it should be by construction), return it
    if not _verify(best_code, sphere, total):
        # fallback to seed
        best_code = sorted(set(seed_idx))

    # Convert indices -> lists of ints
    return [list(words[i]) for i in sorted(set(best_code))]
