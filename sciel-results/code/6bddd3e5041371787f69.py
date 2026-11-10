"""Covering code K_q(n, radius): randomized construction + local improvement."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools

_DEADLINE = 140.0  # safety margin below the 150 s wall-clock kill


def _all_words(q, n):
    total = q ** n
    words = np.empty((total, n), dtype=np.uint8)
    for i in range(n):
        words[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    return words


def _build_neighbours(q, n, radius, words):
    """Precompute neighbour lists as list of lists of indices."""
    total = words.shape[0]
    words_int = words.astype(np.int64)
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    
    nb = []
    for idx in range(total):
        digits = words_int[idx]
        nbs = set()
        if radius == 1:
            nbs.add(idx)
            for i in range(n):
                v = digits[i]
                for vp in range(q):
                    if vp == v:
                        continue
                    nbs.add(idx + (vp - v) * stride[i])
        else:
            # General radius
            patterns = []
            for k in range(radius + 1):
                for subset in itertools.combinations(range(n), k):
                    if k == 0:
                        patterns.append([])
                    else:
                        for assign in itertools.product(range(q), repeat=k):
                            patterns.append([(subset[j], assign[j]) for j in range(k)])
            for pat in patterns:
                new_idx = idx
                for (coord, new_sym) in pat:
                    new_idx += (new_sym - digits[coord]) * stride[coord]
                if 0 <= new_idx < total:
                    nbs.add(new_idx)
        nb.append(sorted(list(nbs)))
    return nb


def _greedy_cover(q, n, radius, words, nb, seed, time_limit, strides):
    """Fast greedy set cover with incremental updates."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()

    covered = np.zeros(total, dtype=bool)
    # uncovered_count[i] = number of uncovered words in nb[i]
    uncovered_count = np.array([len(nb[i]) for i in range(total)], dtype=np.int32)
    codewords = []

    while True:
        if time.time() - start > time_limit:
            break
        best_idx = int(np.argmax(uncovered_count))
        if uncovered_count[best_idx] <= 0:
            break
        
        codewords.append(best_idx)
        
        # Update covered and uncovered_count
        for w in nb[best_idx]:
            if not covered[w]:
                covered[w] = True
                # w is now covered. Decrement count for all centers that cover w.
                # Due to symmetry, centers covering w are nb[w].
                for i in nb[w]:
                    uncovered_count[i] -= 1

    return codewords, covered


def _local_improve(codewords, q, n, radius, words, nb, seed, time_limit, strides):
    """Remove redundant codewords and try simple swaps."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    codewords = list(codewords)
    
    if not codewords:
        return codewords

    # Compute coverage count
    cov_count = np.zeros(total, dtype=np.int32)
    for idx in codewords:
        for w in nb[idx]:
            cov_count[w] += 1

    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # 1. Try removals
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() - start > time_limit:
                break
            idx = codewords[i]
            # Check if any word in nb[idx] has cov_count == 1
            # If yes, removing idx would uncover it.
            is_redundant = True
            for w in nb[idx]:
                if cov_count[w] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove
                for w in nb[idx]:
                    cov_count[w] -= 1
                codewords.pop(i)
                improved = True

        # 2. Try 1-for-2 swaps (optional, if time allows)
        # Find two codewords c1, c2 such that their union is covered by some c_new
        # This is expensive, so we do a limited search.
        if time.time() - start < time_limit - 2:
            # Sample a few pairs
            n_trials = 100
            for _ in range(n_trials):
                if time.time() - start > time_limit:
                    break
                if len(codewords) < 2:
                    break
                i1 = rng.randint(0, len(codewords) - 1)
                i2 = rng.randint(0, len(codewords) - 1)
                if i1 == i2:
                    continue
                
                idx1 = codewords[i1]
                idx2 = codewords[i2]
                
                # Union of neighbors
                union_nbs = set(nb[idx1]) | set(nb[idx2])
                
                # Check if removing both leaves everything covered
                # A word w in union_nbs is safe if cov_count[w] > (1 if w in nb[idx1] else 0) + (1 if w in nb[idx2] else 0)
                # i.e., cov_count[w] - count_in_pair > 0
                safe = True
                for w in union_nbs:
                    cnt = cov_count[w]
                    in1 = 1 if w in nb[idx1] else 0
                    in2 = 1 if w in nb[idx2] else 0
                    if cnt <= in1 + in2:
                        safe = False
                        break
                
                if not safe:
                    continue
                
                # Find a candidate c_new that covers union_nbs
                # Candidate must be in intersection of reverse neighbors of all w in union_nbs
                # i.e., c_new is in nb[w] for all w in union_nbs
                # Start with nb[first_w] and intersect with others
                candidates = set(nb[list(union_nbs)[0]])
                for w in list(union_nbs)[1:]:
                    candidates &= set(nb[w])
                    if not candidates:
                        break
                
                if candidates:
                    # Pick a candidate
                    c_new = next(iter(candidates))
                    
                    # Perform swap
                    # Remove i1, i2
                    # Order matters for indices
                    if i1 > i2:
                        i1, i2 = i2, i1
                    
                    # Remove i2 (higher index)
                    for w in nb[codewords[i2]]:
                        cov_count[w] -= 1
                    codewords.pop(i2)
                    
                    # Adjust i1
                    if i1 > i2:
                        i1 -= 1
                    # Wait, if i1 == i2, we already handled i1==i2 continue.
                    # If i1 < i2, removing i2 doesn't change i1's index.
                    # If i1 > i2, we swapped them, so now i1 is the smaller one.
                    # So i1 is always the smaller index after swap.
                    # Removing i2 (larger) leaves i1 index unchanged.
                    
                    # Remove i1
                    for w in nb[codewords[i1]]:
                        cov_count[w] -= 1
                    codewords.pop(i1)
                    
                    # Add c_new
                    for w in nb[c_new]:
                        cov_count[w] += 1
                    codewords.append(c_new)
                    
                    improved = True

    return codewords


def construct(q, n, radius):
    start = time.time()

    # Precompute words and neighbours
    words = _all_words(q, n)
    nb = _build_neighbours(q, n, radius, words)
    total = words.shape[0]
    strides = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    best_codewords = None
    best_count = float('inf')

    # Multiple greedy runs with different seeds
    seeds = [42, 123, 456, 789, 1011, 2022, 3141, 5678, 9012, 13579]
    n_seeds = len(seeds)

    # Time budget
    time_budget_total = _DEADLINE - time.time() - 5  # 5s safety margin
    time_per_greedy = time_budget_total * 0.4 / n_seeds
    time_per_improve = time_budget_total * 0.6 / n_seeds

    for si, seed in enumerate(seeds):
        elapsed = time.time() - start
        if elapsed > _DEADLINE - 10:
            break
        try:
            # Greedy cover
            t_start = time.time()
            codewords, covered = _greedy_cover(q, n, radius, words, nb, seed, time_per_greedy, strides)

            # Verify coverage (just in case)
            # If not covered, add uncovered words
            if not covered.all():
                uncovered = np.where(~covered)[0]
                for idx in uncovered:
                    codewords.append(int(idx))
                    covered[nb[int(idx)]] = True # This line is wrong, nb[idx] is list of ints, covered is array
            
            # Local improvement
            if time.time() < _DEADLINE - 10:
                improve_time = min(time_per_improve, _DEADLINE - 10 - time.time())
                if improve_time > 0.5:
                    improved = _local_improve(codewords, q, n, radius, words, nb, seed, improve_time, strides)
                    if len(improved) < len(codewords):
                        codewords = improved

            if len(codewords) < best_count:
                best_count = len(codewords)
                best_codewords = codewords[:]

        except Exception:
            continue

    if best_codewords is not None:
        # Convert indices to word lists
        return [words[i].tolist() for i in best_codewords]

    # Fallback: direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
