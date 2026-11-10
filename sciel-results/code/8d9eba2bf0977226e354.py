"""Rewritten program: maximize cardinality of a dissociated reciprocal subset of [1,22]."""
import time
import itertools

# EVOLVE-BLOCK-START
def _is_dissociated(nums):
    """Exact check: no non-empty signed subset of 1/nums sums to 0.
    Equivalent to: after scaling by LCM, no non-empty signed integer combination == 0."""
    n = len(nums)
    if n == 0:
        return True
    # Work in integers scaled by lcm to avoid float error.
    from math import gcd
    L = 1
    for x in nums:
        L = L // gcd(L, x) * x
    vals = [L // x for x in nums]  # positive integers
    # Meet in the middle over sign choices.
    half = n // 2
    left = vals[:half]
    right = vals[half:]
    # All signed sums of left (excluding the all-zero assignment handled later).
    def signed_sums(arr):
        s = {0}
        for v in arr:
            s = {t + v for t in s} | {t - v for t in s}
        return s
    Ls = signed_sums(left)
    Rs = signed_sums(right)
    # A collision exists iff there is a (not-all-zero) assignment with total 0.
    # total = l + r == 0  =>  l == -r.
    # The trivial all-zero assignment gives l=0,r=0 which is fine.
    # A real collision: some l in Ls, r = -l in Rs, and the combined assignment not all zero.
    # If l==0 and r==0 it could be all-zero OR a non-trivial zero on each side.
    # We must detect any non-trivial zero.
    # Build sets of achievable sums; a non-trivial zero exists iff:
    #   (0 in Ls via non-trivial) OR (0 in Rs via non-trivial) OR (l != 0 with -l in Rs)
    # Determine whether 0 is achievable non-trivially on each side.
    def has_nontrivial_zero(arr):
        if not arr:
            return False
        # 0 non-trivially achievable iff some non-empty signed subset sums to 0
        # which for positive ints means two disjoint subsets equal -> check via same method but
        # simpler: 0 in signed_sums(arr) is always true (all zero); need non-all-zero.
        # Count distinct: if any non-empty signed combo == 0.
        # Use DP set but track whether all-zero is the only way.
        # Easier: a non-empty signed subset sums to 0 iff there exist non-empty disjoint A,B
        # with sum(A)==sum(B). Check via subset sums.
        from math import gcd as _g
        L2 = 1
        for x in arr:
            L2 = L2 // _g(L2, x) * x
        vv = [L2 // x for x in arr]
        m = len(vv)
        # subset sums of vv
        ss = {0}
        for v in vv:
            ss = ss | {t + v for t in ss}
        # We need two disjoint non-empty subsets with equal sum.
        # Equivalent: there exist A,B disjoint, nonempty, sumA==sumB.
        # Check: for each target s, count of ways? Simpler: 0 non-trivially in signed sums
        # iff there is a non-zero sign vector in kernel. Detect by: does signed_sums contain 0
        # from a non-all-zero vector? We can detect by checking if any v can be formed as
        # signed sum of others... Instead just test: build set of (sum, used_mask) is heavy.
        # Practical: a non-trivial zero in signed sums of positive ints exists iff
        # some element equals a signed sum of the rest, i.e. v == sum of signed others.
        for i in range(m):
            rest = vv[:i] + vv[i+1:]
            if not rest:
                continue
            if vv[i] in signed_sums(rest):
                return True
        return False
    # Simpler robust approach: detect any non-trivial zero directly via the combined check below.
    # We'll just scan: for each l in Ls, if -l in Rs and (l != 0 or r has nontrivial) ... 
    # To be safe and simple: enumerate all sign vectors is 2^n; for n up to ~12 that's 4096, fine.
    # But we used MITM; do a direct check for small n and MITM existence for the l/r pairing.
    # Direct full check is simplest and exact:
    #   enumerate all 2^n sign vectors, sum, if any non-zero vector gives 0 -> collision.
    # For n <= 22, 2^22 = 4M which is borderline; use MITM for the pairing test instead.
    # Pairing test: collision iff exists (l in Ls, r in Rs) with l + r == 0 and not (l==0 and r==0 from all-zero).
    # The all-zero vector yields l=0,r=0. Any other vector giving l+r=0 is a collision.
    # So: if there is a pair (l,r) with l+r==0 and (l,r) != (0,0)  -> collision.
    # But (0,0) might also be produced by non-trivial vectors. We need: does a non-all-zero
    # full assignment give 0? That equals: (exists l in Ls, r=-l in Rs) AND not (the only such
    # is the all-zero). Detect non-trivial: 
    #   - if there exists l != 0 with -l in Rs -> collision.
    #   - else only l=0 pairs with r=0; collision iff 0 is achievable non-trivially on left OR right.
    if any(l != 0 and (-l) in Rs for l in Ls):
        return False
    # else need non-trivial zero on a side
    if has_nontrivial_zero(left) or has_nontrivial_zero(right):
        return False
    return True


def _is_dissociated_fast(nums):
    """Fast exact check using MITM; returns True if dissociated."""
    n = len(nums)
    if n == 0:
        return True
    from math import gcd
    L = 1
    for x in nums:
        L = L // gcd(L, x) * x
    vals = [L // x for x in nums]
    half = n // 2
    left, right = vals[:half], vals[half:]

    def signed_set(arr):
        s = {0}
        for v in arr:
            s = {t + v for t in s} | {t - v for t in s}
        return s

    Ls = signed_set(left)
    Rs = signed_set(right)
    # non-trivial cross collision
    for l in Ls:
        if l != 0 and (-l) in Rs:
            return False
    # non-trivial zero within a side
    def side_nontriv_zero(arr):
        if not arr:
            return False
        m = len(arr)
        for i in range(m):
            rest = arr[:i] + arr[i+1:]
            if not rest:
                continue
            if arr[i] in signed_set(rest):
                return True
        return False
    if side_nontriv_zero(left) or side_nontriv_zero(right):
        return False
    return True


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s

    # Candidate universe: prioritize larger denominators (smaller reciprocals) which are
    # "safer" against collisions. Search over subsets of [1,22].
    universe = list(range(22, 0, -1))  # 22..1

    best = [1]  # seed fallback (nonempty)

    # Greedy build in decreasing order: add element if it keeps dissociation.
    chosen = []
    for x in universe:
        cand = chosen + [x]
        if _is_dissociated_fast(cand):
            chosen = cand
        if time.time() > deadline:
            break
    if len(chosen) > len(best):
        best = chosen

    # Local improvement: try swapping a chosen element for an unchosen one to grow size.
    # First, try to add any remaining element that fits (in case order mattered).
    remaining = [x for x in universe if x not in chosen]
    for x in remaining:
        if len(chosen) + 1 <= len(best):
            continue
        cand = chosen + [x]
        if _is_dissociated_fast(cand):
            chosen = cand
        if time.time() > deadline:
            break
    if len(chosen) > len(best):
        best = chosen

    # Swap search: replace one chosen with one unchosen to enable further growth.
    chosen_set = set(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in list(chosen):
            for b in [x for x in universe if x not in chosen_set]:
                cand = [c for c in chosen if c != a] + [b]
                if _is_dissociated_fast(cand):
                    # keep if it allows eventual larger set; accept if same size (enables growth)
                    chosen = cand
                    chosen_set = set(chosen)
                    improved = True
                    break
            if improved:
                break
        # after swaps, try to add more
        for x in [x for x in universe if x not in chosen_set]:
            cand = chosen + [x]
            if _is_dissociated_fast(cand):
                chosen = cand
                chosen_set = set(chosen)
                improved = True
            if time.time() > deadline:
                break
    if len(chosen) > len(best):
        best = chosen

    return best
# EVOLVE-BLOCK-END
