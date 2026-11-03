"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Fast greedy init + local search (1-swap, 2-swap) + perturbation/re-greedy escape.
    """
    if n == 0:
        return [[]]

    total_words = q ** n

    def int_to_word(val, n, q):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # ---- Precompute neighbors (fast path for radius=1) ----
    neighbors = [None] * total_words
    if radius == 1:
        for w in range(total_words):
            word = int_to_word(w, n, q)
            neigh = [w]
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_int(word, q))
                word[pos] = cur
            neighbors[w] = neigh  # list, no duplicates for radius=1
    else:
        for w in range(total_words):
            word = int_to_word(w, n, q)
            neigh = {w}
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.add(word_to_int(word, q))
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
                                neigh.add(word_to_int(word, q))
                            word[pos2] = c2
                        word[pos1] = c1
            neighbors[w] = list(neigh)

    # ---- Greedy init (sampling-based) ----
    def greedy_init():
        covered = [False] * total_words
        uncovered = set(range(total_words))
        codewords = []
        while uncovered:
            if len(uncovered) > 300:
                sample = random.sample(list(uncovered), 100)
            else:
                sample = list(uncovered)
            candidates = set()
            for u in sample:
                candidates.update(neighbors[u])
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
                best_word = neighbors[u][0]
            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered.discard(u)
        return codewords

    # ---- Coverage bookkeeping ----
    def build_coverage(codewords):
        cc = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                cc[u] += 1
        return cc

    def remove_redundant(codewords, cc, cw_set):
        changed = True
        while changed:
            changed = False
            idxs = list(range(len(codewords)))
            random.shuffle(idxs)
            for i in idxs:
                cw = codewords[i]
                ok = True
                for u in neighbors[cw]:
                    if cc[u] == 1:
                        ok = False
                        break
                if ok:
                    for u in neighbors[cw]:
                        cc[u] -= 1
                    codewords.pop(i)
                    cw_set.discard(cw)
                    changed = True
                    break

    # ---- 1-swap: move old_cw to a better location (must cover its unique words) ----
    def try_1swap(codewords, cc, cw_set, idx):
        old_cw = codewords[idx]
        would_uncover = [u for u in neighbors[old_cw] if cc[u] == 1]
        if not would_uncover:
            for u in neighbors[old_cw]:
                cc[u] -= 1
            codewords.pop(idx)
            cw_set.discard(old_cw)
            return True
        # candidate pool: intersection of neighbors of would_uncover words
        first = would_uncover[0]
        cand = [w for w in neighbors[first] if w not in cw_set and w != old_cw]
        for u in would_uncover[1:]:
            su = set(neighbors[u])
            cand = [w for w in cand if w in su]
        if not cand:
            return False
        # pick the candidate that maximizes additional (redundant) coverage
        best = -1
        best_extra = -1
        for w in cand:
            extra = 0
            for u in neighbors[w]:
                extra += cc[u]  # higher redundancy = more removable later
            if extra > best_extra:
                best_extra = extra
                best = w
        if best == -1:
            return False
        for u in neighbors[old_cw]:
            cc[u] -= 1
        for u in neighbors[best]:
            cc[u] += 1
        codewords[idx] = best
        cw_set.discard(old_cw)
        cw_set.add(best)
        return True

    # ---- 2-swap: relocate two codewords simultaneously ----
    def try_2swap(codewords, cc, cw_set):
        m = len(codewords)
        if m < 2:
            return False
        i, j = random.sample(range(m), 2)
        a, b = codewords[i], codewords[j]
        ua = [u for u in neighbors[a] if cc[u] == 1]
        ub = [u for u in neighbors[b] if cc[u] == 1]
        # candidate for a: covers ua, not b, not in set
        if ua:
            cand_a = [w for w in neighbors[ua[0]] if w not in cw_set and w != a and w != b]
            for u in ua[1:]:
                su = set(neighbors[u])
                cand_a = [w for w in cand_a if w in su]
        else:
            cand_a = [w for w in neighbors[a] if w not in cw_set and w != a and w != b]
        if ub:
            cand_b = [w for w in neighbors[ub[0]] if w not in cw_set and w != a and w != b]
            for u in ub[1:]:
                su = set(neighbors[u])
                cand_b = [w for w in cand_b if w in su]
        else:
            cand_b = [w for w in neighbors[b] if w not in cw_set and w != a and w != b]
        if not cand_a or not cand_b:
            return False
        na = random.choice(cand_a)
        nb = random.choice(cand_b)
        if na == nb or na in (a, b) or nb in (a, b):
            return False
        # apply
        for u in neighbors[a]:
            cc[u] -= 1
        for u in neighbors[b]:
            cc[u] -= 1
        for u in neighbors[na]:
            cc[u] += 1
        for u in neighbors[nb]:
            cc[u] += 1
        codewords[i] = na
        codewords[j] = nb
        cw_set.discard(a)
        cw_set.discard(b)
        cw_set.add(na)
        cw_set.add(nb)
        return True

    # ---- Perturbation + re-greedy fill for uncovered gaps ----
    def perturb_and_fill(codewords, cc, cw_set):
        # remove a few random codewords to create gaps, then re-greedy
        m = len(codewords)
        k = max(1, m // 20)
        removed = random.sample(range(m), k)
        for i in sorted(removed, reverse=True):
            cw = codewords[i]
            for u in neighbors[cw]:
                cc[u] -= 1
            codewords.pop(i)
            cw_set.discard(cw)
        # find uncovered
        uncovered = [u for u in range(total_words) if cc[u] == 0]
        if not uncovered:
            return
        while uncovered:
            if len(uncovered) > 200:
                sample = random.sample(uncovered, 60)
            else:
                sample = uncovered
            candidates = set()
            for u in sample:
                candidates.update(neighbors[u])
            best_word = -1
            best_new_cov = -1
            for w in candidates:
                new_cov = 0
                for u in neighbors[w]:
                    if cc[u] == 0:
                        new_cov += 1
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_word = w
            if best_word == -1:
                best_word = neighbors[uncovered[0]][0]
            codewords.append(best_word)
            cw_set.add(best_word)
            for u in neighbors[best_word]:
                cc[u] += 1
            uncovered = [u for u in uncovered if cc[u] == 0]

    # ---- Main driver ----
    start_time = time.time()
    time_limit = 115  # safety margin under 150s kill

    codewords = greedy_init()
    cc = build_coverage(codewords)
    cw_set = set(codewords)
    remove_redundant(codewords, cc, cw_set)

    best_size = len(codewords)
    best_snapshot = list(codewords)

    stagnation = 0
    last_improve = start_time

    while time.time() - start_time < time_limit:
        # periodic redundancy removal
        if remove_redundant(codewords, cc, cw_set):
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_snapshot = list(codewords)
                last_improve = time.time()
                stagnation = 0
            continue

        if not codewords:
            break

        moved = False
        r = random.random()
        if r < 0.55:
            idx = random.randrange(len(codewords))
            moved = try_1swap(codewords, cc, cw_set, idx)
        elif r < 0.80:
            moved = try_2swap(codewords, cc, cw_set)
        else:
            # perturbation escape
            perturb_and_fill(codewords, cc, cw_set)
            remove_redundant(codewords, cc, cw_set)
            moved = True

        if len(codewords) < best_size:
            best_size = len(codewords)
            best_snapshot = list(codewords)
            last_improve = time.time()
            stagnation = 0
        else:
            stagnation += 1

        # if long stagnation, force a perturbation to escape local optimum
        if stagnation > 400 and time.time() - last_improve > 8:
            perturb_and_fill(codewords, cc, cw_set)
            remove_redundant(codewords, cc, cw_set)
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_snapshot = list(codewords)
            stagnation = 0
            last_improve = time.time()

    # Final polish: one more redundancy sweep
    if time.time() - start_time < time_limit - 2:
        remove_redundant(codewords, cc, cw_set)
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_snapshot = list(codewords)

    # Return the best (smallest) feasible solution found
    result = best_snapshot if best_snapshot else codewords
    return [int_to_word(cw, n, q) for cw in result]
# EVOLVE-BLOCK-END
