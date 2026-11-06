"""Combined program: global greedy set cover over bitsets + fast local-search polish."""
import time
import random

_BASE = 51975
# divisors > 1 of 51975 = 3^3 * 5^2 * 7 * 11 (all odd)
_DIVS = [d for d in range(3, _BASE + 1, 2) if _BASE % d == 0]
_DIVS.sort()
# per-modulus precomputed residue masks (int bitset over _BASE residues)
_MASKS = [[0] * m for m in _DIVS]
for m in _DIVS:
    mm = _MASKS[_DIVS.index(m)]
    for r in range(m):
        bits = 0
        x = r
        while x < _BASE:
            bits |= (1 << x)
            x += m
        mm[r] = bits
_FULL = (1 << _BASE) - 1


def construct():
    deadline = time.time() + 115.0
    n = len(_DIVS)

    def greedy():
        """Global greedy set cover: repeatedly pick the class covering the most
        currently-uncovered residues. Returns (classes, uncovered_bits, score)."""
        unc = _FULL
        chosen = [None] * n
        for _ in range(n):
            best_i, best_r, best_c = -1, -1, -1
            for i in range(n):
                if chosen[i] is not None:
                    continue
                m = _DIVS[i]
                for r in range(m):
                    c = (unc & _MASKS[i][r]).bit_count()
                    if c > best_c:
                        best_c, best_i, best_r = c, i, r
            chosen[best_i] = best_r
            unc &= ~_MASKS[best_i][best_r]
        return chosen, unc, unc.bit_count()

    def polish(chosen, unc):
        """Fast local-search polish: repeatedly set each modulus's residue to the
        one maximizing coverage of residues not covered by the other classes.
        Returns (chosen, unc, score)."""
        score = unc.bit_count()
        for _ in range(80):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                m = _DIVS[i]
                # residues not covered by any other chosen class
                wo = _FULL
                for j in range(n):
                    if j != i and chosen[j] is not None:
                        wo &= ~_MASKS[j][chosen[j]]
                cur = chosen[i]
                best_r, best_c = cur, (wo & _MASKS[i][cur]).bit_count()
                for r in range(m):
                    c = (wo & _MASKS[i][r]).bit_count()
                    if c > best_c:
                        best_c, best_r = c, r
                if best_r != cur:
                    chosen[i] = best_r
                    improved = True
            if not improved:
                break
            # recompute exact uncovered + score
            unc = _FULL
            for i in range(n):
                if chosen[i] is not None:
                    unc &= ~_MASKS[i][chosen[i]]
            ns = unc.bit_count()
            if ns >= score:
                break
            score = ns
            if score == 0:
                break
        return chosen, unc, score

    best_classes, best_score = None, _BASE
    chosen, unc, sc = greedy()
    chosen, unc, sc = polish(chosen, unc)
    best_classes, best_score = chosen[:], sc
    if best_score == 0:
        return [[_DIVS[i], best_classes[i]] for i in range(n) if best_classes[i] is not None]

    # random restarts with polish to escape greedy local minima
    random.seed(7)
    while time.time() < deadline - 3.0:
        chosen, unc, sc = greedy()
        chosen, unc, sc = polish(chosen, unc)
        if sc < best_score:
            best_classes, best_score = chosen[:], sc
            if best_score == 0:
                break

    return [[_DIVS[i], best_classes[i]] for i in range(n) if best_classes[i] is not None]
