"""Rewritten: find minimal-count distinct Egyptian fraction (denominators >= 5) summing to 1.

Strategy:
  - Hardcode a known-valid starting solution (the seed-style DP result) as a safety net.
  - Search for a representation with FEWER terms using exact Fraction arithmetic.
  - For each candidate count k (from small upward), run a bounded best-first/DFS search
    over distinct denominators >= 5, pruning by feasibility (remaining sum must be reachable).
  - Return the smallest-k valid solution found; deadline-guarded.
"""
import time
from fractions import Fraction

_START = time.time()
_DEADLINE = _START + 50.0   # safety margin under the 90 s kill


def _seed_solution():
    """Reproduce the seed's subset-DP result (a valid, larger-count solution)."""
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    return None


def _validate(denoms):
    """Return True iff denoms are distinct, in [5, 1e7], and reciprocal sum is exactly 1."""
    if not denoms:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    for d in denoms:
        if not (5 <= d <= 10_000_000):
            return False
    s = Fraction(0, 1)
    for d in denoms:
        s += Fraction(1, d)
    return s == 1


def _feasible_lo_hi(k, min_d):
    """Upper bound on the sum of k distinct reciprocals with denominators >= min_d
    (use the k smallest: min_d, min_d+1, ...), and lower bound (largest denominators).
    We only need an upper bound to prune: if even the k largest possible reciprocals
    (smallest denominators) can't reach the remaining target, prune."""
    # upper bound = sum of 1/(min_d + i) for i in 0..k-1
    ub = Fraction(0, 1)
    for i in range(k):
        ub += Fraction(1, min_d + i)
    return ub


def _search(k, max_d, max_nodes):
    """Find k distinct denominators in [5, max_d] whose reciprocals sum to exactly 1.
    Best-first / DFS with pruning. Returns a list of k denominators or None."""
    target = Fraction(1, 1)
    # We pick denominators in increasing order to avoid duplicates.
    # State: (current_sum, next_min_d, chosen)
    # Feasibility: need remaining = target - current_sum, fill with (k - len(chosen)) terms
    # each >= next_min_d, distinct.
    # Upper bound on remaining achievable = sum of 1/next_min_d ... for (k-len) terms.
    # Also each single term 1/d <= remaining needed (else overshoot for the last term).
    stack = [(Fraction(0, 1), 5, [])]
    nodes = 0
    # Use a simple DFS with recursion limit via explicit stack, exploring smallest d first
    # to bias toward valid small-denominator sets.
    while stack:
        cur, min_d, chosen = stack.pop()
        remaining_terms = k - len(chosen)
        if remaining_terms == 0:
            if cur == target:
                return chosen
            continue
        if time.time() > _DEADLINE:
            return None
        nodes += 1
        if nodes > max_nodes:
            return None
        need = target - cur
        if need <= 0:
            continue
        # Prune 1: even the largest possible remaining sum (smallest denominators) < need
        ub = Fraction(0, 1)
        for i in range(remaining_terms):
            ub += Fraction(1, min_d + i)
        if ub < need:
            continue
        # Prune 2: the smallest single denominator 1/min_d must be <= need
        # (otherwise every allowed term overshoots)
        if Fraction(1, min_d) > need:
            continue
        # Choose next denominator d in [min_d, max_d]
        # Bound d from above: 1/d <= need  =>  d >= ceil(1/need). But d >= min_d.
        # Also need to be able to fill remaining_terms-1 more terms after d.
        # To keep node count low, iterate d upward but cap by max_d.
        lo = min_d
        # Upper cap on d: we still need (remaining_terms-1) more terms each >= d+1.
        # Their max sum is sum_{i=1}^{remaining_terms-1} 1/(d+i).
        # Need: need - 1/d >= 0  and  need - 1/d <= sum_{i=1}^{rem-1} 1/(d+i)
        for d in range(lo, max_d + 1):
            term = Fraction(1, d)
            if term > need:
                # larger d -> smaller term, so continue increasing d
                continue
            new_need = need - term
            rem2 = remaining_terms - 1
            if rem2 == 0:
                if new_need == 0:
                    return chosen + [d]
                continue
            # Feasibility for the remaining rem2 terms with denominators >= d+1:
            ub2 = Fraction(0, 1)
            for i in range(1, rem2 + 1):
                ub2 += Fraction(1, d + i)
            if new_need > ub2:
                # even the largest remaining sum can't cover new_need -> but larger d
                # makes new_need bigger (term smaller) and ub2 smaller, so prune further?
                # As d increases: term decreases -> new_need increases; ub2 decreases.
                # So if ub2 < new_need now, it will only get worse for larger d. Prune rest.
                break
            # lower bound: smallest remaining sum (largest denominators) is tiny; ok.
            stack.append((cur + term, d + 1, chosen + [d]))
    return None


def construct():
    """Return 1..256 distinct integer denominators in [5, 1e7] whose reciprocal sum
    is exactly 1, minimizing their count."""
    t0 = time.time()

    # Try to find a minimal-count solution, increasing k from a small lower bound.
    # Lower bound: sum of the k largest reciprocals (5..k+4) must be >= 1.
    # Find smallest k where that holds; k below it is infeasible.
    lb = 1
    while True:
        s = Fraction(0, 1)
        for i in range(lb):
            s += Fraction(1, 5 + i)
        if s >= 1:
            break
        lb += 1
        if lb > 60:
            break

    best = None
    # Search increasing k from lb. Keep a modest node budget per k.
    for k in range(lb, 41):
        if time.time() > _DEADLINE - 5.0:
            break
        # max_d grows with k; keep search tractable.
        max_d = min(2000, 5 + 200 * k)
        res = _search(k, max_d, max_nodes=2_000_000)
        if res is not None and _validate(res):
            best = res
            break

    if best is None:
        # Fallback: seed solution (valid, larger count).
        best = _seed_solution()
        if best is None:
            raise RuntimeError("no representation found")

    # Final safety: ensure validity; if somehow invalid, fall back to seed.
    if not _validate(best):
        fb = _seed_solution()
        if fb is not None and _validate(fb):
            best = fb
    return best
