"""Ternary covering code K_3(6,1): greedy removal + add + polish, time-safe."""
import itertools
import time


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 110.0  # 10s safety margin under 120s

    # ---- Initial code: direct sum of ternary tetracode (9 words, radius-1 perfect) ----
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail)
            for h in tetracode
            for tail in itertools.product(range(3), repeat=n - 4)]
    code = [tuple(c) for c in code]
    m0 = len(code)
    assert m0 == 81

    # ---- Per-position coverage counts: cov[pos][value] = # codewords with value at pos ----
    def build_cov(code):
        cov = [[0] * q for _ in range(n)]
        for cw in code:
            for pos in range(n):
                cov[pos][cw[pos]] += 1
        return cov

    def covers(cw, other, cov):
        # True iff removing `cw` leaves every word within radius of some remaining codeword.
        # Only words at distance exactly 1 from cw need checking (distance-0 word is covered
        # by cw itself and, after removal, must be within 1 of another codeword).
        for pos in range(n):
            v = cw[pos]
            if cov[pos][v] <= 1:
                # cw is the only codeword with value v at pos. The word with v at pos and
                # matching cw elsewhere is at distance 1 from cw; after removal it must be
                # within 1 of some other codeword.
                found = False
                for other in other:
                    if other[pos] == v:
                        continue  # can't help (would need to differ elsewhere but matches v)
                    # other differs at pos; for the target word (cw with pos=v) to be within
                    # 1 of other, other must equal cw at all other positions.
                    ok = True
                    for p2 in range(n):
                        if p2 == pos:
                            continue
                        if other[p2] != cw[p2]:
                            ok = False
                            break
                    if ok:
                        found = True
                        break
                if not found:
                    return False
        return True

    def try_remove(cw, code, cov):
        others = [c for c in code if c is not cw]
        if covers(cw, others, cov):
            return True
        return False

    # ---- Greedy removal loop ----
    def greedy_remove(code, cov, max_rounds=200):
        for _ in range(max_rounds):
            if time.time() > DEADLINE:
                break
            removed_any = False
            for i, cw in enumerate(list(code)):
                if time.time() > DEADLINE:
                    break
                others = [c for c in code if c is not cw]
                if covers(cw, others, cov):
                    code.remove(cw)
                    for pos in range(n):
                        cov[pos][cw[pos]] -= 1
                    removed_any = True
                    break  # restart scan with updated cov
            if not removed_any:
                break
        return code, cov

    code, cov = greedy_remove(code, cov)

    # ---- Add step: add a word that covers many under-covered positions, then remove ----
    def add_one(code, cov):
        # Score candidate words by how many positions they place a value with low coverage.
        best = None
        best_score = -1
        # Candidate: for each position, the least-covered value; combine.
        # Try a handful of heuristic candidates.
        cands = []
        # Candidate 1: least-covered value per position (may not be a valid word but is in space)
        cand = [min(range(q), key=lambda v: cov[pos][v]) for pos in range(n)]
        cands.append(tuple(cand))
        # Candidate 2: most-covered value per position (to enable removal of others)
        cand = [max(range(q), key=lambda v: cov[pos][v]) for pos in range(n)]
        cands.append(tuple(cand))
        # Candidate 3: a few random-ish
        import random
        rng = random.Random(12345)
        for _ in range(50):
            cands.append(tuple(rng.randrange(q) for _ in range(n)))
        for c in cands:
            if c in code:
                continue
            # score = sum over positions of (q - cov[pos][c[pos]])  (more under-covered = better)
            score = 0
            for pos in range(n):
                score += (q - cov[pos][c[pos]])
            if score > best_score:
                best_score = score
                best = c
        if best is None or best in code:
            return code, cov
        new_code = code + [best]
        new_cov = [[c2 for c2 in row] for row in cov]
        for pos in range(n):
            new_cov[pos][best[pos]] += 1
        return new_code, new_cov

    # Try a few add-then-remove cycles
    for _ in range(5):
        if time.time() > DEADLINE:
            break
        if len(code) <= 73:
            break
        new_code, new_cov = add_one(code, cov)
        if new_code is code:
            break
        new_code, new_cov = greedy_remove(new_code, new_cov, max_rounds=50)
        if len(new_code) < len(code):
            code, cov = new_code, new_cov
        else:
            # revert
            break

    # ---- Final polish: one more removal pass ----
    code, cov = greedy_remove(code, cov, max_rounds=100)

    # ---- Return as list of lists (distinct guaranteed) ----
    return [list(c) for c in code]
