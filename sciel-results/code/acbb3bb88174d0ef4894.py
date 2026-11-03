"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Combines a fast greedy construction (candidate sampling from uncovered neighborhoods)
    with randomized local search (swap + redundancy removal), followed by a final
    high-precision polish pass.
    """
    if n == 0:
        return [[]]

    total_words = q ** n

    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors (fast path for radius==1, general otherwise)
    neighbors = [None] * total_words
    if radius == 1:
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_int(word, q))
                word[pos] = cur
            neighbors[w] = set(neigh)
    else:
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_int(word, q))
                    word[pos] = cur
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_int(word, q))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = set(neigh)

    def greedy_init():
        """Fast greedy: pick the word covering the most uncovered words,
        using a candidate pool built from neighborhoods of sampled uncovered words."""
        covered = [False] * total_words
        codewords = []
        uncovered = set(range(total_words))

        while uncovered:
            if len(uncovered) > 500:
                sample_uncovered = random.sample(list(uncovered), min(100, len(uncovered)))
            else:
                sample_uncovered = list(uncovered)

            candidates = set()
            for u in sample_uncovered:
                candidates.update(neighbors[u])

            if not candidates:
                break

            best_word = -1
            best_new_cov = -1
            for w in candidates:
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_word = w
                    if new_cov == len(uncovered):
                        break

            if best_word == -1:
                u = next(iter(uncovered))
                best_word = next(iter(neighbors[u]))

            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered.discard(u)

        return codewords

    def optimize(codewords, time_limit):
        """Local search: remove redundant codewords and swap codewords to improve coverage."""
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1

        cw_set = set(codewords)

        def remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any

        remove_redundant()

        start_time = time.time()
        last_redundancy_check = start_time

        while time.time() - start_time < time_limit:
            if time.time() - last_redundancy_check > 0.3:
                remove_redundant()
                last_redundancy_check = time.time()
                continue

            if not codewords:
                break

            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]

            would_uncover = [u for u in neighbors[old_cw] if coverage_count[u] == 1]

            if not would_uncover:
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue

            # Find a replacement covering all uniquely-covered words
            candidates = set(neighbors[would_uncover[0]])
            for u in would_uncover[1:]:
                candidates.intersection_update(neighbors[u])
                if not candidates:
                    break

            if not candidates:
                continue
            candidates.discard(old_cw)
            if not candidates:
                continue

            new_cw = random.choice(list(candidates))

            for u in neighbors[old_cw]:
                coverage_count[u] -= 1
            for u in neighbors[new_cw]:
                coverage_count[u] += 1
            codewords[idx] = new_cw
            cw_set.discard(old_cw)
            cw_set.add(new_cw)

            remove_redundant()

        remove_redundant()
        return codewords

    def polish(codewords, time_limit):
        """Final high-precision polish: exhaustive redundancy removal plus a full
        deterministic single-elimination sweep (tighter tolerance, more iterations)."""
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1

        cw_set = set(codewords)

        def remove_redundant():
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        break

        start_time = time.time()
        # Repeat full sweeps while time allows (more iterations = tighter result)
        while time.time() - start_time < time_limit:
            remove_redundant()
            # Deterministic single-elimination sweep over all codewords
            removed = True
            while removed and time.time() - start_time < time_limit:
                removed = False
                for i in range(len(codewords)):
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        removed = True
                        break
            if time.time() - start_time > time_limit - 1:
                break

        remove_redundant()
        return codewords

    # Multiple restarts with an explicit deadline guard (150s kill -> 140s budget)
    start_time = time.time()
    time_limit = 140
    best_codewords = []
    best_len = float('inf')

    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining <= 1:
            break

        # Allocate time for this iteration: greedy + optimize
        iter_time = min(20, remaining - 1)
        if iter_time <= 0:
            break

        cw = greedy_init()
        cw = optimize(cw, max(0.5, iter_time - 2))

        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw

        # Reserve final time for a polish pass on the best found so far
        if time.time() - start_time > time_limit - 12:
            break

    # Final high-precision polish pass using any remaining time
    if best_codewords and (time_limit - (time.time() - start_time)) > 0.5:
        best_codewords = polish(best_codewords, max(0.5, time_limit - (time.time() - start_time) - 1))

    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
