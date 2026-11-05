"""Ternary covering code K_3(7,1): vectorized greedy + count-based local search."""
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 112.0  # safety margin before 120s wall-clock

    # Fallback for non-(3,7,1): simple product construction (valid, not optimal)
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q]
                     for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]

    N = 3 ** n  # 2187
    # all_words[i] = digits of base-3 representation of i
    all_words = np.zeros((N, n), dtype=np.int8)
    for idx in range(N):
        t = idx
        for pos in range(n - 1, -1, -1):
            all_words[idx, pos] = t % 3
            t //= 3

    # Ball matrix B[i, j] = 1 iff dist(i, j) <= radius
    B = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        B[i, i] = 1
        w = all_words[i]
        for pos in range(n):
            base = 3 ** (n - 1 - pos)
            for val in range(3):
                if val == w[pos]:
                    continue
                B[i, i + (val - int(w[pos])) * base] = 1

    def is_valid(code):
        if not code:
            return False
        return bool((B[np.asarray(code, dtype=np.intp)].any(axis=0)).all())

    def greedy_fill(code, covered):
        """Add codewords until every word is covered. Returns (code, covered)."""
        code = list(code)
        covered = covered.copy()
        while not covered.all():
            if time.time() > deadline - 6:
                break
            gain_vec = (~covered).astype(np.int32)
            gains = B @ gain_vec
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            code.append(best)
            covered |= B[best].astype(bool)
        return code, covered

    def local_search(code):
        """Count-based deletion + swap passes. Returns the (possibly smaller) code."""
        code = list(code)
        cs = set(code)
        counts = B[np.asarray(code, dtype=np.intp)].sum(axis=0)
        
        # Precompute ball indices for faster access
        ball_indices = {c: B[c].nonzero()[0] for c in code}

        improved = True
        while improved and time.time() < deadline - 6:
            improved = False

            # ---- Deletion pass: drop any codeword whose ball is fully covered ----
            to_remove = [c for c in code
                         if counts[ball_indices[c]].min() > 1]
            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    cs.discard(c)
                    counts[ball_indices[c]] -= 1
                    del ball_indices[c]

            # ---- Swap pass: relocate codewords to reduce critical points ----
            if time.time() > deadline - 6:
                break
            shuffled = list(code)
            random.shuffle(shuffled)
            for c in shuffled:
                if time.time() > deadline - 6:
                    break
                if c not in cs:
                    continue
                c_ball = ball_indices[c]
                crit = [p for p in c_ball if counts[p] == 1]
                if not crit:
                    continue  # would be removed by deletion
                if len(crit) > 3:
                    continue  # too hard to cover all with one new position
                cand_set = set(B[crit[0]].nonzero()[0])
                for p in crit[1:]:
                    cand_set &= set(B[p].nonzero()[0])
                candidates = [x for x in cand_set if x not in cs]
                if not candidates:
                    continue
                best_w, best_score = -1, 0
                for w in candidates:
                    w_ball = B[w].nonzero()[0]
                    score = 0
                    for p in w_ball:
                        if counts[p] == 1 and B[c, p] == 0:
                            score += 1
                    for p in c_ball:
                        if counts[p] == 2 and B[w, p] == 0:
                            score -= 1
                    if score > best_score:
                        best_score, best_w = score, w
                if best_w >= 0:
                    counts[c_ball] -= 1
                    w_ball = B[best_w].nonzero()[0]
                    counts[w_ball] += 1
                    code.remove(c)
                    cs.discard(c)
                    code.append(best_w)
                    cs.add(best_w)
                    ball_indices[best_w] = w_ball
                    improved = True
        return code

    # ---- Initial greedy construction ----
    covered = np.zeros(N, dtype=bool)
    code = greedy_fill([], covered)[0]
    code = local_search(code)
    if not is_valid(code):
        code, _ = greedy_fill(code, np.zeros(N, dtype=bool))
        code = local_search(code)
    best = code if is_valid(code) else []

    # ---- Kick / re-greedy / local search, keep best-so-far ----
    while time.time() < deadline - 8:
        cur = list(best)
        # Use a slightly larger kick size to escape local minima more effectively
        k = max(1, min(5, len(cur) // 5))
        to_remove = set(random.sample(cur, k))
        kicked = [c for c in cur if c not in to_remove]
        covered = np.zeros(N, dtype=bool)
        if kicked:
            covered |= B[np.asarray(kicked, dtype=np.intp)].any(axis=0)
        new_code, _ = greedy_fill(kicked, covered)
        new_code = local_search(new_code)
        if is_valid(new_code) and (not best or len(new_code) < len(best)):
            best = new_code

    if not best or not is_valid(best):
        # Safety fallback: full greedy then local search
        covered = np.zeros(N, dtype=bool)
        code = greedy_fill([], covered)[0]
        code = local_search(code)
        best = code
    
    # Final polish pass with more aggressive local search
    if time.time() < deadline - 2:
        final_code = local_search(best)
        if is_valid(final_code) and len(final_code) < len(best):
            best = final_code

    if not is_valid(best):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(all_words[c]) for c in best]
