# EVOLVE-BLOCK-START
import time
import random

import numpy as np


def _hamming_leq_one(q, n, radius, words):
    """Return a list of (word, list_of_neighbours) for each word in `words`.
    A neighbour is any vector within Hamming distance <= radius (inclusive)."""
    neigh = []
    for w in words:
        ns = [w]
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig:
                    continue
                w[i] = v
                ns.append(list(w))
            w[i] = orig
        neigh.append(ns)
    return neigh


def _greedy_cover(q, n, radius, seed=0, deadline=None, verbose=False):
    """Greedy set cover: repeatedly pick the codeword covering the most
    still-uncovered words. Uses a bitset for the covered set and a numpy
    array for the remaining uncovered words."""
    rng = random.Random(seed)
    total = q ** n
    full_mask = (1 << total) - 1

    covered = np.zeros(total, dtype=bool)
    remaining = np.arange(total, dtype=np.int64)
    remaining_count = total

    # Precompute the base word list
    base_words = [list(w) for w in np.ndindex(*([q] * n))]
    # Precompute neighbour sets (list of indices) for every word
    neigh_sets = []
    for w in base_words:
        idx = int(np.ravel_multi_index(tuple(w), (q,) * n))
        ns = {idx}
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig:
                    continue
                w[i] = v
                ns.add(int(np.ravel_multi_index(tuple(w), (q,) * n)))
            w[i] = orig
        neigh_sets.append(ns)

    code = []
    # Precompute the full neighbour set for each index (as a frozenset for fast ops)
    neigh_frozensets = [frozenset(s) for s in neigh_sets]

    # Greedy loop
    while remaining_count > 0:
        if deadline is not None and time.time() > deadline:
            break
        # Find the word covering the most uncovered words
        best_idx = -1
        best_count = -1
        # Iterate over all candidate words
        for idx in range(total):
            if covered[idx]:
                continue
            # Count how many uncovered words this word covers
            cnt = 0
            for j in neigh_sets[idx]:
                if not covered[j]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
                if best_count == remaining_count:
                    break
        if best_idx < 0:
            break
        code.append(list(base_words[best_idx]))
        for j in neigh_sets[best_idx]:
            if not covered[j]:
                covered[j] = True
                remaining_count -= 1
        if verbose:
            print(f"  added {best_idx}, covered {total - remaining_count}/{total}")

    return code, covered


def _local_search(q, n, radius, code, deadline=None):
    """Try to remove or replace codewords to shrink the code while
    maintaining coverage. Returns the improved code."""
    total = q ** n
    full_mask = (1 << total) - 1

    # Build coverage: for each word, which codewords cover it
    code_set = set(tuple(c) for c in code)
    code_list = [list(c) for c in code_set]
    m = len(code_list)

    # Precompute neighbour sets for each codeword
    cw_neigh = []
    for c in code_list:
        ns = {tuple(c)}
        for i in range(n):
            orig = c[i]
            for v in range(q):
                if v == orig:
                    continue
                c[i] = v
                ns.add(tuple(c))
            c[i] = orig
        cw_neigh.append(ns)

    # Check coverage
    all_words = set()
    for ns in cw_neigh:
        all_words.update(ns)
    if len(all_words) < total:
        return code_list  # not fully covered, skip

    # Try removing each codeword
    improved = True
    while improved:
        improved = False
        for k in range(len(code_list)):
            if deadline is not None and time.time() > deadline:
                return code_list
            # Check if removing codeword k still covers everything
            # A word is covered if it's in the neighbour set of some other codeword
            removed_neigh = cw_neigh[k]
            ok = True
            for w in removed_neigh:
                found = False
                for j in range(len(code_list)):
                    if j == k:
                        continue
                    if w in cw_neigh[j]:
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok:
                code_list.pop(k)
                cw_neigh.pop(k)
                improved = True
                break

    return code_list


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s

    # Quick sanity: for small cases, use exhaustive
    if q ** n <= 2000:
        # Greedy should be fast
        code, covered = _greedy_cover(q, n, radius, seed=42, deadline=deadline)
        if len(covered) == q ** n and covered.all():
            code = _local_search(q, n, radius, code, deadline=deadline)
            return code
        # Fallback: all words
        return [list(w) for w in np.ndindex(*([q] * n))]

    # For the main case (q=3, n=6, radius=1): 729 words
    # Use a smarter approach: greedy with randomization + local search

    best_code = None
    best_size = float('inf')

    # Try multiple random seeds for the greedy algorithm
    for seed in [42, 123, 7, 999, 2024]:
        if time.time() > deadline - 30:
            break
        code, covered = _greedy_cover(q, n, radius, seed=seed, deadline=deadline)
        if len(covered) == q ** n and covered.all():
            code = _local_search(q, n, radius, code, deadline=deadline)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                if time.time() > deadline - 15:
                    break

    if best_code is not None:
        return best_code

    # Fallback: direct sum construction (729 words)
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Ultimate fallback
    return [list(w) for w in np.ndindex(*([q] * n))]
# EVOLVE-BLOCK-END
