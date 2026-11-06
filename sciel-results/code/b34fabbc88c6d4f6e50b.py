"""Divisibility-antichain representations by smooth numbers."""
import time
from itertools import combinations

_DEADLINE = 85.0  # safety margin before 90s wall-clock kill

def _smooth_numbers_up_to(n):
    """Return sorted list of all 2^a*3^b*5^c <= n (positive)."""
    out = set()
    p2 = 1
    while p2 <= n:
        p23 = p2
        while p23 <= n:
            p235 = p23
            while p235 <= n:
                out.add(p235)
                p235 *= 5
            p23 *= 3
        p2 *= 2
    return sorted(out)

def _is_antichain(terms):
    """No term divides another (terms are distinct positives)."""
    for i in range(len(terms)):
        a = terms[i]
        for j in range(i + 1, len(terms)):
            b = terms[j]
            if a < b and b % a == 0:
                return False
            if b < a and a % b == 0:
                return False
    return True

def _find_minimal(target, smooth, max_terms=30, deadline=None):
    """Find a representation of target as sum of k distinct smooth terms,
    k as small as possible, forming a divisibility antichain.
    Returns list of terms or None."""
    cands = [x for x in smooth if x <= target]
    cands_set = set(cands)
    for k in range(1, max_terms + 1):
        if deadline is not None and time.time() > deadline:
            return None
        # Quick feasibility: sum of k smallest >= target? Not needed; just search.
        for terms in combinations(cands, k):
            if sum(terms) != target:
                continue
            if _is_antichain(terms):
                return list(terms)
    return None

def _find_any(target, smooth, max_terms=30, deadline=None):
    """Fallback: any valid antichain representation (not necessarily minimal)."""
    cands = [x for x in smooth if x <= target]
    for k in range(1, max_terms + 1):
        if deadline is not None and time.time() > deadline:
            return None
        for terms in combinations(cands, k):
            if sum(terms) == target and _is_antichain(terms):
                return list(terms)
    return None

def construct():
    start = time.time()
    deadline = start + (_DEADLINE - (time.time() - start))
    smooth = _smooth_numbers_up_to(128)
    answers = [None] * 128
    # Try each target, minimal first
    for n in range(1, 129):
        if time.time() > deadline:
            break
        ans = _find_minimal(n, smooth, 30, deadline)
        if ans is None:
            ans = _find_any(n, smooth, 30, deadline)
        answers[n - 1] = ans
    # If deadline hit, ensure we have something for all (use any valid)
    for n in range(1, 129):
        if answers[n - 1] is None:
            answers[n - 1] = _find_any(n, smooth, 30)
    return answers
