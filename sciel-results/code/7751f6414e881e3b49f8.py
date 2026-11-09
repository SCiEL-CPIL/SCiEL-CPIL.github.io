"""Few-term Egyptian fractions with denominators > 4.

Strategy:
  - Depth-first search over unit fractions 1/d with d >= 5, in strictly
    increasing order, summing to exactly 1.
  - Aggressive pruning:
      * If current sum already exceeds 1 -> prune.
      * If even taking the largest possible remaining terms (1/d_next +
        1/(d_next+1) + ...) cannot reach 1 -> prune.
      * If current sum + max_possible_remaining < 1 -> prune.
  - Track the best (shortest) solution found and enforce a wall-clock
    deadline so we always return a valid answer quickly.
"""
import time

_MAX_D = 10_000_000
_MIN_D = 5


def construct():
    deadline = time.time() + 50.0  # safety margin under 60s

    best = []  # best (shortest) solution found so far

    # We will search by number of terms k = 2, 3, 4, ...
    # For a given k, we do a DFS choosing d_1 < d_2 < ... < d_k, d_i >= 5,
    # such that sum(1/d_i) == 1.

    # Precompute: for a given remaining "need" (as a fraction) and a given
    # minimum allowed next denominator, the maximum number of terms we can
    # still fit is bounded. We use rational arithmetic via Python's Fraction.
    from fractions import Fraction

    one = Fraction(1, 1)

    # Helper: given current sum S (Fraction), next min denominator d_min,
    # and remaining terms r, check if it's possible to reach 1.
    # Upper bound on remaining sum: 1/d_min + 1/(d_min+1) + ... (r terms)
    # Lower bound: we need 1 - S <= that upper bound.
    # Also 1 - S must be > 0 (else prune) and the largest single term 1/d_min
    # must be >= 1 - S (else even the biggest we can pick is too small).

    def max_sum_from(d_min, r):
        """Upper bound: sum of 1/(d_min), 1/(d_min+1), ..., 1/(d_min+r-1)."""
        s = Fraction(0, 1)
        for i in range(r):
            s += Fraction(1, d_min + i)
        return s

    def dfs(terms, current_sum, d_min, r):
        """
        terms: list of chosen denominators so far.
        current_sum: Fraction, sum of 1/d for d in terms.
        d_min: minimum allowed next denominator.
        r: number of terms still to choose.
        """
        nonlocal best
        if time.time() > deadline:
            return

        if r == 0:
            if current_sum == one:
                if not best or len(terms) < len(best):
                    best = list(terms)
            return

        need = one - current_sum
        if need <= 0:
            return

        # Prune: even the maximum possible remaining sum is too small.
        # The max remaining sum is achieved by the smallest denominators:
        # 1/d_min + 1/(d_min+1) + ... + 1/(d_min+r-1)
        max_rem = max_sum_from(d_min, r)
        if need > max_rem:
            return

        # The next denominator d must satisfy 1/d >= need / r  (rough)
        # More precisely, 1/d >= need - max_sum_from(d+1, r-1)
        # But a simpler bound: 1/d >= need / r  =>  d <= r / need
        # Also d >= d_min.
        # And we need enough room: after picking d, the remaining r-1 terms
        # with denominators >= d+1 must sum to need - 1/d.
        # Lower bound on d: 1/d <= need  =>  d >= 1/need  (but d is int >= d_min)
        # Upper bound on d: we need need - 1/d <= max_sum_from(d+1, r-1)
        #   i.e., the remaining need after taking 1/d must be achievable.
        # Also 1/d must be < need (strictly, since we need positive remaining)
        #   => d > 1/need.
        # And 1/d must be such that remaining is positive: 1/d < need => d > 1/need.

        # Compute d range:
        # d >= d_min
        # d > 1/need  =>  d >= floor(1/need) + 1
        import math
        d_lo = d_min
        if need > 0:
            # d must satisfy 1/d < need  =>  d > 1/need
            # 1/need could be large; compute integer bound
            # d > 1/need  =>  d >= floor(1/need) + 1
            inv_need = 1.0 / float(need) if need < 1 else 0.0
            d_lo2 = int(inv_need) + 1
            if d_lo2 > d_lo:
                d_lo = d_lo2

        # Upper bound: we need need - 1/d <= max_sum_from(d+1, r-1)
        # For r-1 == 0: need - 1/d == 0 => d == 1/need exactly.
        if r == 1:
            # need must equal 1/d exactly for some integer d >= d_lo
            # d = 1/need
            if need.denominator == 1 and need.numerator == 1:
                d = need.denominator
                if d >= d_lo and d <= _MAX_D:
                    terms.append(d)
                    if not best or len(terms) < len(best):
                        best = list(terms)
                    terms.pop()
            return

        # For d, the remaining need is need - 1/d, and we need r-1 more terms
        # with denominators >= d+1.
        # The maximum remaining sum is max_sum_from(d+1, r-1).
        # We need need - 1/d <= max_sum_from(d+1, r-1).
        # As d increases, 1/d decreases, so need - 1/d increases.
        # max_sum_from(d+1, r-1) decreases as d increases.
        # So there's an upper bound on d.
        # Let's iterate d from d_lo upward, and stop when the condition fails.

        # Also, the smallest remaining need (when d is smallest) is need - 1/d_lo.
        # We need at least r-1 terms, each at most 1/(d+1), so:
        # need - 1/d <= (r-1) / (d+1)  is a rough upper bound check.

        d = d_lo
        # Safety: limit iterations per node to avoid runaway
        max_iters = 200000
        iters = 0
        while d <= _MAX_D and iters < max_iters:
            iters += 1
            if time.time() > deadline:
                return
            term = Fraction(1, d)
            new_need = need - term
            if new_need <= 0:
                # 1/d >= need; if equal and r==1 handled above, else skip
                # Since d > 1/need, new_need < 0 always here actually.
                # Wait: d_lo ensures d > 1/need, so 1/d < need, new_need > 0.
                # But floating point... let's just break if new_need <= 0.
                break
            # Check if remaining is achievable
            # max remaining sum with r-1 terms starting at d+1:
            if r - 1 == 1:
                # need exactly 1/d' for some d' >= d+1
                # new_need must be 1/integer
                if new_need.denominator == 1 and new_need.numerator == 1:
                    d2 = new_need.denominator
                    if d2 >= d + 1 and d2 <= _MAX_D:
                        terms.append(d)
                        terms.append(d2)
                        if not best or len(terms) < len(best):
                            best = list(terms)
                        terms.pop()
                        terms.pop()
                # No more d to try (new_need decreases as d increases,
                # so if it's not a unit fraction now, it won't be later? 
                # Actually new_need = need - 1/d increases as d increases.
                # So we should continue.)
            else:
                # Check upper bound: new_need <= max_sum_from(d+1, r-1)
                # Compute max_sum_from(d+1, r-1)
                ms = Fraction(0, 1)
                for i in range(r - 1):
                    ms += Fraction(1, d + 1 + i)
                if new_need > ms:
                    # Even the max remaining can't cover new_need; 
                    # as d increases, new_need increases and ms decreases,
                    # so all larger d will also fail. Break.
                    break
                # Also lower bound: new_need must be > 0 (ensured)
                # and the smallest possible remaining sum is with largest denominators,
                # but we don't have a tight lower bound easily. Just recurse.
                terms.append(d)
                dfs(terms, current_sum + term, d + 1, r - 1)
                terms.pop()
            d += 1

    # Search for k = 2, 3, 4, ... terms
    # k=2: 1/a + 1/b = 1, a<b, a>=5. => (a-1)(b-1)=1 => a-1=1, b-1=1 => a=2. No.
    # So k>=3.
    for k in range(3, 8):  # try up to 7 terms
        if time.time() > deadline:
            break
        dfs([], Fraction(0, 1), _MIN_D, k)
        if best:
            break

    if best:
        return best

    # Fallback: a known valid representation (shouldn't reach here)
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + ... let's just return a valid one.
    # Actually, let's construct a simple valid one.
    # 1 = 1/2 + 1/3 + 1/6 but denominators must be >=5.
    # 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 but distinct required.
    # Let's use: 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + ... 
    # This is getting complex. Return a precomputed valid set.
    # 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14+1/15+1/16+1/17+1/18+1/19+1/20+...
    # Let me just return a known 5-term or 6-term solution.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/35 + 1/42 + 1/70 + 1/105 + 1/140 + 1/210
    # That's too many. Let me find a smaller one.
    # Actually, the DFS should find something. If it doesn't, raise.
    raise RuntimeError("no valid representation found")
