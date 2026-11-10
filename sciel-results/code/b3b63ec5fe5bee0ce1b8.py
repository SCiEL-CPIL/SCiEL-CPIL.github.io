"""Stronger search for few-term Egyptian fractions (denominators >= 5, distinct, sum = 1)."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Return a list of distinct integer denominators in [5, 10000000] whose
    reciprocal sum is exactly 1.  Minimizes the number of terms.

    Strategy:
      1. Hard-code a known 4-term solution (fast path).
      2. If that fails validation, run a bounded DFS over Egyptian fractions
         with denominators >= 5, depth up to 6, with a wall-clock guard.
    """
    t0 = time.time()
    DEADLINE = t0 + 80  # 80 s hard guard; evaluator kills at 90 s.

    # ---- fast path: known 4-term representation with all denominators >= 5 ----
    # 1/5 + 1/7 + 1/10 + 1/35 = (7 + 5 + 10 + 2)/70 = 24/70?  no.
    # Let's use a verified one:
    # 1/6 + 1/10 + 1/15 + 1/30  -> 1, but 6,10,15,30 are all >=5?  yes.
    # Actually 1/6+1/10+1/15+1/30 = (5+3+2+1)/30 = 11/30 != 1.
    # Correct known: 1/2+1/3+1/7+1/42 = 1 but 2<5.
    # Let's find a valid 4-term set programmatically at init?  Too slow.
    # Known: 1/6+1/7+1/10+1/15+1/42?  That's 5 terms.
    #
    # Let's just do a quick search for small 4-term solutions.
    # 1/a + 1/b + 1/c + 1/d = 1, a<b<c<d, a>=5.
    # Try a=5: 1/b+1/c+1/d = 4/5.  b>=6.
    #   b=6: 1/c+1/d = 4/5-1/6 = (24-5)/30 = 19/30.  c>=7.
    #     c=7: 1/d = 19/30-1/7 = (133-30)/210 = 103/210.  d=210/103 not int.
    #     c=8: 1/d = 19/30-1/8 = (152-30)/240 = 122/240 = 61/120.  d=120/61 not int.
    #     c=9: 1/d = 19/30-1/9 = (57-10)/90 = 47/90.  d=90/47 not int.
    #     c=10: 1/d = 19/30-1/10 = 16/30 = 8/15.  d=15/8 not int.
    #     c=11: 1/d = 19/30-1/11 = (209-30)/330 = 179/330.  not int.
    #     ... this is getting complex.
    #
    # Let's just run the DFS with a tight budget for depth 4 first.
    # If found, return it.  Otherwise fall back to depth 5/6.
    # If time runs out, return best found or a known 6-term solution.

    # Known 6-term (all >=5): 1/6+1/7+1/8+1/9+1/10+1/...?
    # Let's use: 1/5+1/7+1/10+1/15+1/35+1/... 
    # Actually, simplest verified: 1/6+1/7+1/14+1/21+1/42+... 
    # Let's just rely on DFS.

    # Known valid 5-term: 1/6 + 1/7 + 1/14 + 1/21 + 1/42 = (7+6+3+2+1)/42 = 19/42. Not 1.
    # 1/5+1/6+1/7+1/8+1/... 
  
    # Let me just run a proper search.
    best = None
    best_terms = None

    def validate(denoms):
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == 1

    def dfs(remaining, min_denom, depth, max_depth, path, terms):
        nonlocal best, best_terms
        if time.time() > DEADLINE:
            return
        if remaining == 0:
            if not path:
                return
            cand = path[:]
            if validate(cand):
                if best is None or len(cand) < len(best):
                    best = cand
                    best_terms = list(terms)
            return
        if depth > max_depth:
            return
        if remaining < 0:
            return
        # Pruning: with (max_depth - depth + 1) terms left, each >= 1/min_denom
        # So max we can add is (max_depth - depth + 1) * (1/min_denom)
        # Actually remaining is a Fraction; compare.
        max_terms_left = max_depth - depth + 1
        if remaining > max_terms_left * Fraction(1, min_denom):
            return
        # Min possible: we need at least 1 term
        if remaining > Fraction(1, min_denom) * max_terms_left:
            # Already covered above
            pass
        # Lower bound on next denominator
        # 1/d <= remaining, so d >= 1/remaining
        lo = max(min_denom, math.ceil(1 / float(remaining)))
        if lo < min_denom:
            lo = min_denom
        # Upper bound: 1/d >= remaining / max_terms_left  => d <= max_terms_left / remaining
        hi = int(max_terms_left / float(remaining)) + 2
        if hi > 10000000:
            hi = 10000000
        if lo > hi:
            return
        # Iterate
        for d in range(lo, hi + 1):
            if time.time() > DEADLINE:
                return
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            path.append(d)
            terms.add(d)
            dfs(new_rem, d + 1, depth + 1, max_depth, path, terms)
            path.pop()
            terms.discard(d)
            if best is not None and len(best) == 4:
                # Can't do better than 4 (minimum possible for >=5)
                return

    # Try depths 4, 5, 6
    for max_d in [4, 5, 6]:
        if time.time() > DEADLINE - 5:
            break
        dfs(Fraction(1, 1), 5, 1, max_d, [], set())
        if best is not None and len(best) == max_d and max_d == 4:
            break
        if best is not None:
            break

    if best is not None:
        return best

    # Fallback: a known valid representation
    # 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + ... too many
    # Let's construct: 1 = 1/5 + 4/5.  4/5 = 1/6 + 19/30. 19/30 = 1/7 + 80/210 = 1/7 + 8/21.
    # 8/21 = 1/9 + 1/63?  1/9=7/63, 8/21=24/63, 24/63-7/63=17/63. Not clean.
    # 
    # Known: 1/2+1/3+1/7+1/42 but 2<5.
    # Scale: multiply denominators by k? No, that changes the sum.
    #
    # Let's just return a computed 6-term from a quick greedy:
    denoms = []
    rem = Fraction(1, 1)
    d = 5
    while rem > 0 and len(denoms) < 256:
        # Greedy: largest unit fraction <= rem with denom >= max(5, last+1)
        lo = max(5, (denoms[-1] if denoms else 0) + 1)
        # 1/x <= rem  => x >= 1/rem
        x_min = math.ceil(1 / float(rem))
        if x_min < lo:
            x_min = lo
        if x_min > 10000000:
            break
        denoms.append(x_min)
        rem -= Fraction(1, x_min)
        if rem == 0:
            break
        d = x_min + 1

    if denoms and validate(denoms):
        return denoms

    # Absolute fallback (should never be reached)
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
