"""Covering code K_q(n, radius): greedy + local search with constraint propagation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools
from collections import defaultdict

_DEADLINE = 145.0  # safety margin below the 150 s wall-clock kill


def _all_words(q, n):
    total = q ** n
    words = np.empty((total, n), dtype=np.uint8)
    for i in range(n):
        words[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    return words


def _build_neighbours(q, n, radius, words):
    """Precompute neighbour lists (indices within Hamming distance <= radius)."""
    total = words.shape[0]
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    if radius == 1:
        nb = np.empty((total, 1 + n * (q - 1)), dtype=np.int64)
        for idx in range(total):
            digits = words[idx].astype(np.int64)
            pos = 0
            nb[idx, pos] = idx
            pos += 1
            for i in range(n):
                v = digits[i]
                for vp in range(q):
                    if vp == v:
                        continue
                    nb[idx, pos] = idx + (vp - v) * stride[i]
                    pos += 1
        return nb
    else:
        patterns = []
        for k in range(radius + 1):
            for subset in itertools.combinations(range(n), k):
                if k == 0:
                    patterns.append([])
                else:
                    for assign in itertools.product(range(q), repeat=k):
                        patterns.append([(subset[j], assign[j]) for j in range(k)])
        nb_list = []
        for idx in range(total):
            digits = words[idx].astype(np.int64)
            nbs = set()
            for pat in patterns:
                new_idx = idx
                for (coord, new_sym) in pat:
                    new_idx += (new_sym - digits[coord]) * stride[coord]
                if 0 <= new_idx < total:
                    nbs.add(new_idx)
            nb_list.append(sorted(nbs))
        return nb_list


def _verify_coverage(codewords, q, n, radius, words, nb):
    """Check if codewords cover all words. Returns (covered_all, uncovered_count)."""
    total = words.shape[0]
    covered = np.zeros(total, dtype=bool)
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        nbs = nb[idx]
        for c in nbs:
            covered[c] = True
    return covered.all(), int((~covered).sum())


def _greedy_cover(q, n, radius, words, nb, seed, time_limit):
    """Greedy set cover with deadline guard."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()

    covered = np.zeros(total, dtype=bool)
    codewords = []

    # gain[i] = number of uncovered words covered by position i
    gain = np.zeros(total, dtype=np.int32)
    for i in range(total):
        for c in nb[i]:
            gain[i] += 1

    while True:
        if time.time() - start > time_limit:
            break

        best_idx = int(np.argmax(gain))
        if gain[best_idx] == 0:
            break

        codewords.append(words[best_idx].tolist())
        for c in nb[best_idx]:
            if not covered[c]:
                covered[c] = True
                for w in nb[c]:
                    gain[w] -= 1

    return codewords, covered


def _remove_redundant(codewords, q, n, radius, words, nb, time_limit):
    """Remove redundant codewords."""
    start = time.time()
    total = words.shape[0]
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    codewords = list(codewords)

    coverage_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        for c in nb[idx]:
            coverage_count[c] += 1

    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        new_codewords = []
        for cw in codewords:
            idx = int(np.dot(cw, stride))
            redundant = True
            for c in nb[idx]:
                if coverage_count[c] == 1:
                    redundant = False
                    break
            if redundant:
                for c in nb[idx]:
                    coverage_count[c] -= 1
                changed = True
            else:
                new_codewords.append(cw)
        codewords = new_codewords

    return codewords


def _local_search(codewords, q, n, radius, words, nb, seed, time_limit):
    """
    Local search: try to remove codewords or swap them to reduce count while maintaining coverage.
    Uses constraint propagation: for each uncovered word, track which candidates could cover it.
    """
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    included = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        included[idx] = True

    # coverage_count[w] = number of selected codewords that cover word w
    coverage_count = np.zeros(total, dtype=np.int32)
    for i in range(total):
        if included[i]:
            for c in nb[i]:
                coverage_count[c] += 1

    best_count = int(included.sum())
    best_included = included.copy()

    # Build reverse mapping: for each word, list of positions that cover it
    # This is already nb (symmetric for radius 1)
    
    no_improve = 0
    max_no_improve = 500

    while time.time() - start < time_limit - 0.5:
        # Try to remove a random codeword
        included_indices = np.where(included)[0]
        if len(included_indices) == 0:
            break
            
        # Pick a random codeword to try removing
        idx = included_indices[rng.randint(0, len(included_indices) - 1)]
        
        # Check if removing it leaves any word uncovered
        can_remove = True
        for c in nb[idx]:
            if coverage_count[c] == 1:
                can_remove = False
                break
        
        if can_remove:
            included[idx] = False
            for c in nb[idx]:
                coverage_count[c] -= 1
            best_count = int(included.sum())
            best_included = included.copy()
            no_improve = 0
        else:
            # Try swap: remove idx, add another codeword that covers more uncovered words
            # Find words that would become uncovered
            would_be_uncovered = []
            for c in nb[idx]:
                if coverage_count[c] == 1:
                    would_be_uncovered.append(c)
            
            if len(would_be_uncovered) > 0:
                # Find a codeword that covers the most of would_be_uncovered
                best_add = -1
                best_gain = 0
                # Sample some candidates for efficiency
                sample_size = min(200, total)
                candidates = rng.sample(range(total), sample_size)
                for cand in candidates:
                    if included[cand]:
                        continue
                    gain = 0
                    for c in would_be_uncovered:
                        if c in nb[cand]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_add = cand
                
                if best_add >= 0 and best_gain == len(would_be_uncovered):
                    # Do the swap
                    included[idx] = False
                    for c in nb[idx]:
                        coverage_count[c] -= 1
                    included[best_add] = True
                    for c in nb[best_add]:
                        coverage_count[c] += 1
                    # Count stays same, but structure changed - might enable future removals
                    no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            break

    result = []
    for i in range(total):
        if best_included[i]:
            result.append(words[i].tolist())
    return result


def _backtracking_improve(codewords, q, n, radius, words, nb, seed, time_limit):
    """
    Backtracking search to find a smaller covering code.
    Starts from current codewords and tries to reduce by re-optimizing.
    """
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    included = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        included[idx] = True

    coverage_count = np.zeros(total, dtype=np.int32)
    for i in range(total):
        if included[i]:
            for c in nb[i]:
                coverage_count[c] += 1

    best_count = int(included.sum())
    best_included = included.copy()

    # Iteratively try to improve by removing one codeword and finding a better replacement set
    for iteration in range(50):
        if time.time() - start > time_limit - 1:
            break
        
        # Find codewords with minimum "essential coverage" (words only they cover)
        essential_counts = np.zeros(total, dtype=np.int32)
        for i in range(total):
            if included[i]:
                for c in nb[i]:
                    if coverage_count[c] == 1:
                        essential_counts[i] += 1
        
        # Sort by essential count (most essential first)
        included_indices = np.where(included)[0]
        sorted_indices = sorted(included_indices, key=lambda x: essential_counts[x], reverse=True)
        
        # Try removing the least essential codeword
        if len(sorted_indices) > 0:
            idx = sorted_indices[0]
            if essential_counts[idx] == 0:
                included[idx] = False
                for c in nb[idx]:
                    coverage_count[c] -= 1
                best_count = int(included.sum())
                best_included = included.copy()
    
    result = []
    for i in range(total):
        if best_included[i]:
            result.append(words[i].tolist())
    return result


def construct(q, n, radius):
    start = time.time()

    # Precompute words and neighbours
    words = _all_words(q, n)
    nb = _build_neighbours(q, n, radius, words)
    total = words.shape[0]

    best_codewords = None
    best_count = float('inf')

    # Phase 1: Greedy constructions with multiple seeds
    greedy_seeds = [42, 123, 456, 789, 1011, 2022, 3141, 2718]
    time_for_greedy = min(15.0, (_DEADLINE - time.time()) * 0.25)
    time_per_greedy = time_for_greedy / len(greedy_seeds)

    for seed in greedy_seeds:
        if time.time() > _DEADLINE - 40:
            break
        try:
            codewords, covered = _greedy_cover(q, n, radius, words, nb, seed, time_per_greedy)
            cov_all, unc = _verify_coverage(codewords, q, n, radius, words, nb)
            if cov_all and len(codewords) < best_count:
                best_count = len(codewords)
                best_codewords = codewords[:]
        except Exception:
            continue

    if best_codewords is None:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Phase 2: Remove redundant codewords
    if time.time() < _DEADLINE - 30:
        try:
            improved = _remove_redundant(best_codewords, q, n, radius, words, nb, 5.0)
            cov_all, unc = _verify_coverage(improved, q, n, radius, words, nb)
            if cov_all and len(improved) < best_count:
                best_count = len(improved)
                best_codewords = improved[:]
        except Exception:
            pass

    # Phase 3: Local search polish
    remaining = _DEADLINE - time.time() - 5
    if remaining > 5 and best_codewords is not None:
        ls_seeds = [42, 123, 456, 789]
        time_per_ls = remaining / len(ls_seeds)
        for seed in ls_seeds:
            if time.time() > _DEADLINE - 5:
                break
            try:
                ls_codewords = _local_search(
                    best_codewords, q, n, radius, words, nb, seed, time_per_ls
                )
                cov_all, unc = _verify_coverage(ls_codewords, q, n, radius, words, nb)
                if cov_all and len(ls_codewords) < best_count:
                    best_count = len(ls_codewords)
                    best_codewords = ls_codewords[:]
            except Exception:
                continue

        # Phase 4: Backtracking improvement
        remaining2 = _DEADLINE - time.time() - 3
        if remaining2 > 3:
            try:
                bt_codewords = _backtracking_improve(
                    best_codewords, q, n, radius, words, nb, 42, remaining2
                )
                cov_all, unc = _verify_coverage(bt_codewords, q, n, radius, words, nb)
                if cov_all and len(bt_codewords) < best_count:
                    best_count = len(bt_codewords)
                    best_codewords = bt_codewords[:]
            except Exception:
                pass

        # Final redundancy removal
        if time.time() < _DEADLINE - 3:
            try:
                improved = _remove_redundant(best_codewords, q, n, radius, words, nb, 2.0)
                cov_all, unc = _verify_coverage(improved, q, n, radius, words, nb)
                if cov_all and len(improved) < best_count:
                    best_count = len(improved)
                    best_codewords = improved[:]
            except Exception:
                pass

    return best_codewords
# EVOLVE-BLOCK-END
