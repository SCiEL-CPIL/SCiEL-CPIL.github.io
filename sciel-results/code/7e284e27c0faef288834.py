# EVOLVE-BLOCK-START
import time
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140  # 10s safety margin out of 150s

    # ------------------------------------------------------------------
    # Precompute the full ball-membership matrix once.
    # For q=3, n=6, radius=1: total = 729, matrix is 729x729 bool (~0.5MB).
    # balls[i, j] == True  iff  word j is within Hamming distance `radius` of word i.
    # ------------------------------------------------------------------
    total = q ** n
    shapes = (q,) * n

    # Convert all words to index space: word index -> coordinate tuple
    all_words = [list(np.unravel_index(i, shapes)) for i in range(total)]

    # Build ball matrix
    balls = np.zeros((total, total), dtype=bool)
    for i in range(total):
        w = all_words[i]
        balls[i, i] = True
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    w2 = w.copy()
                    w2[pos] = val
                    j = int(np.ravel_multi_index(tuple(w2), shapes))
                    balls[i, j] = True

    # ------------------------------------------------------------------
    # Phase 0: initial seed from tetracode product construction (if q==3, n>=4)
    # ------------------------------------------------------------------
    def _tetracode_seed():
        if q != 3 or n < 4:
            return None
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        code = [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]
        # Convert to indices
        idxs = [int(np.ravel_multi_index(tuple(cw), shapes)) for cw in code]
        return idxs

    # ------------------------------------------------------------------
    # Phase 1: bitset greedy set cover (pick word covering most uncovered)
    # ------------------------------------------------------------------
    def _greedy_cover(seed_idxs):
        if seed_idxs is not None and len(seed_idxs) > 0:
            # Start from the seed, verify it's a cover, then greedily add if needed
            covered = np.zeros(total, dtype=bool)
            chosen = list(seed_idxs)
            for idx in chosen:
                covered |= balls[idx]
            # If seed doesn't cover everything, finish with greedy
            while not covered.all() and time.time() < deadline - 30:
                uncovered_mask = ~covered
                counts = balls[:, uncovered_mask].sum(axis=1)
                best = int(np.argmax(counts))
                if counts[best] == 0:
                    break
                chosen.append(best)
                covered |= balls[best]
            return chosen
        else:
            # Pure greedy from scratch
            covered = np.zeros(total, dtype=bool)
            chosen = []
            while not covered.all() and time.time() < deadline - 30:
                uncovered_mask = ~covered
                counts = balls[:, uncovered_mask].sum(axis=1)
                best = int(np.argmax(counts))
                if counts[best] == 0:
                    break
                chosen.append(best)
                covered |= balls[best]
            return chosen

    seed = _tetracode_seed()
    best_code = _greedy_cover(seed)

    # ------------------------------------------------------------------
    # Phase 2: add-then-remove local search
    #   - Try removing each codeword; if coverage breaks, add back the best
    #     replacement, then retry removals. This escapes greedy local minima.
    # ------------------------------------------------------------------
    def _local_search(code):
        code = list(code)
        covered = np.zeros(total, dtype=bool)
        for idx in code:
            covered |= balls[idx]
        if not covered.all():
            return code

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(code)):
                if time.time() > deadline - 5:
                    break
                # Tentatively remove code[i]
                old = code[i]
                code.pop(i)
                # Recompute coverage of remaining
                rem_covered = np.zeros(total, dtype=bool)
                for idx in code:
                    rem_covered |= balls[idx]
                if rem_covered.all():
                    # Removal is safe: keep it
                    improved = True
                    covered = rem_covered
                    break
                else:
                    # Restore; try to find a better replacement for old
                    code.insert(i, old)
                    # Points only covered by old
                    only_old = covered & ~rem_covered
                    # Find candidate that covers all of only_old and adds least overlap
                    best_alt = -1
                    best_overlap = total + 1
                    # Only consider candidates that cover only_old
                    for cand in range(total):
                        if time.time() > deadline - 5:
                            break
                        b = balls[cand]
                        if not (b[only_old].all()):
                            continue
                        overlap = int((b & rem_covered).sum())
                        if overlap < best_overlap:
                            best_overlap = overlap
                            best_alt = cand
                    if best_alt >= 0 and best_alt != old:
                        code[i] = best_alt
                        covered = rem_covered | balls[best_alt]
                        improved = True
                        break
                    # else: no improvement, continue scanning

        return code

    best_code = _local_search(best_code)

    # ------------------------------------------------------------------
    # Phase 3: final removal pass (pure removal, no add)
    # ------------------------------------------------------------------
    def _final_remove(code):
        code = list(code)
        covered = np.zeros(total, dtype=bool)
        for idx in code:
            covered |= balls[idx]
        if not covered.all():
            return code
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(len(code)):
                if time.time() > deadline - 1:
                    break
                old = code[i]
                code.pop(i)
                rem = np.zeros(total, dtype=bool)
                for idx in code:
                    rem |= balls[idx]
                if rem.all():
                    improved = True
                    covered = rem
                    break
                else:
                    code.insert(i, old)
        return code

    best_code = _final_remove(best_code)

    # ------------------------------------------------------------------
    # Convert indices back to coordinate lists
    # ------------------------------------------------------------------
    result = [all_words[i] for i in best_code]
    return result
# EVOLVE-BLOCK-END
