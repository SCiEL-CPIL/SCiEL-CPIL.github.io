"""Divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

_DEADLINE = 90.0  # evaluator kills after 90s; leave margin
_START = time.time()


def _time_left():
    return _DEADLINE - 6.0 - (time.time() - _START)  # 6s safety margin


def _build_smooths(limit):
    """All 2^a*3^b*5^c <= limit, sorted."""
    out = []
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                out.append(c)
                c *= 5
            b *= 3
        a *= 2
    out = sorted(set(out))
    return out


def _precompute(limit=200):
    smooths = _build_smooths(limit)
    # Precompute divisibility relation between smooth numbers
    # For each smooth number, which other smooth numbers divide it
    # We'll use a set for O(1) lookup
    smooth_set = set(smooths)
    return smooths, smooth_set


def _is_antichain(terms):
    """Check no term divides another."""
    n = len(terms)
    for i in range(n):
        ai = terms[i]
        for j in range(i + 1, n):
            bj = terms[j]
            if bj % ai == 0:
                return False
            if ai % bj == 0:
                return False
    return True


def _find_1term(n, smooths, smooth_set):
    """Check if n itself is smooth."""
    if n in smooth_set:
        return [n]
    return None


def _find_2terms(n, smooths, smooth_set):
    """Find two distinct smooth numbers summing to n, neither divides the other."""
    # For each smooth a < n, check if n-a is smooth and antichain
    for a in smooths:
        if a >= n:
            break
        b = n - a
        if b <= 0:
            break
        if b in smooth_set:
            if b % a != 0 and a % b != 0:
                return [a, b]
    return None


def _find_3terms(n, smooths, smooth_set):
    """Find three distinct smooth numbers summing to n, forming antichain."""
    # Use two-sum approach: for each pair (a,b) with a+b < n, check if n-a-b is smooth
    # To be efficient, iterate a over smooths, then b over smooths > a
    n_smooths = [x for x in smooths if x < n]
    for i, a in enumerate(n_smooths):
        for j in range(i + 1, len(n_smooths)):
            b = n_smooths[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= 0:
                break
            if c in smooth_set:
                # Check distinctness
                if a == b or b == c or a == c:
                    continue
                # Check antichain
                if _is_antichain([a, b, c]):
                    return [a, b, c]
    return None


def _find_4terms(n, smooths, smooth_set):
    """Find four distinct smooth numbers summing to n, forming antichain."""
    # Use three-sum approach with pruning
    n_smooths = [x for x in smooths if x < n]
    # For efficiency, iterate first two, then use two-sum for remaining
    for i, a in enumerate(n_smooths):
        if a * 4 > n:  # minimum sum of 4 distinct >= a is a + (a+1) + ... but we need smooth
            # Actually just check if remaining 3 can sum to n-a
            if n - a < 3:
                break
        for j in range(i + 1, len(n_smooths)):
            b = n_smooths[j]
            rem = n - a - b
            if rem < 3:
                break
            # Find two smooths c, d with c+d=rem, c>d>b (or just c,d distinct from a,b)
            for k in range(j + 1, len(n_smooths)):
                c = n_smooths[k]
                if c >= rem:
                    break
                d = rem - c
                if d <= 0:
                    break
                if d in smooth_set:
                    # Check distinctness
                    if a == b or b == c or c == d or a == c or a == d or b == d:
                        continue
                    # Check antichain
                    if _is_antichain([a, b, c, d]):
                        return [a, b, c, d]
    return None


def _find_5terms(n, smooths, smooth_set):
    """Find five distinct smooth numbers summing to n, forming antichain."""
    n_smooths = [x for x in smooths if x < n]
    # Iterative deepening with pruning
    for i, a in enumerate(n_smooths):
        if n - a < 4:  # need at least 4 more terms each >= 1
            break
        for j in range(i + 1, len(n_smooths)):
            b = n_smooths[j]
            if n - a - b < 3:
                break
            for k in range(j + 1, len(n_smooths)):
                c = n_smooths[k]
                if n - a - b - c < 2:
                    break
                for l in range(k + 1, len(n_smooths)):
                    d = n_smooths[l]
                    e = n - a - b - c - d
                    if e <= 0:
                        break
                    if e in smooth_set:
                        if a == b or b == c or c == d or d == e or a == c or a == d or a == e or b == d or b == e or c == e:
                            continue
                        if _is_antichain([a, b, c, d, e]):
                            return [a, b, c, d, e]
    return None


def _find_k_terms(n, k, smooths, smooth_set):
    """General k-term search using recursion with pruning."""
    if k == 1:
        return _find_1term(n, smooths, smooth_set)
    if k == 2:
        return _find_2terms(n, smooths, smooth_set)
    if k == 3:
        return _find_3terms(n, smooths, smooth_set)
    if k == 4:
        return _find_4terms(n, smooths, smooth_set)
    if k == 5:
        return _find_5terms(n, smooths, smooth_set)
    
    # General recursive search for k > 5
    n_smooths = [x for x in smooths if x < n]
    
    def backtrack(start, remaining, terms):
        if len(terms) == k - 1:
            # Last term must be smooth
            last = remaining
            if last in smooth_set:
                # Check distinct
                if last in terms:
                    return None
                # Check antichain
                if _is_antichain(terms + [last]):
                    return terms + [last]
            return None
        if len(terms) == k:
            return None
        
        # Pruning: remaining sum must be >= sum of smallest (k - len(terms)) unused smooths
        # Simple bound: each remaining term >= 1, so remaining >= (k - len(terms))
        if remaining < (k - len(terms)):
            return None
        
        for idx in range(start, len(n_smooths)):
            a = n_smooths[idx]
            if a >= remaining:
                break
            new_terms = terms + [a]
            # Quick antichain check for new term
            valid = True
            for t in terms:
                if a % t == 0 or t % a == 0:
                    valid = False
                    break
            if not valid:
                continue
            result = backtrack(idx + 1, remaining - a, new_terms)
            if result is not None:
                return result
        return None
    
    return backtrack(0, n, [])


def construct():
    global _START
    _START = time.time()
    
    smooths, smooth_set = _precompute(200)
    
    answers = [None] * 128
    
    for n in range(1, 129):
        if _time_left() < 0:
            # Emergency: fill remaining with valid (possibly non-minimal) solutions
            # Use a fallback that always works
            for m in range(n, 129):
                if answers[m - 1] is None:
                    answers[m - 1] = _emergency_solution(m, smooths, smooth_set)
            break
        
        # Try minimal k first
        found = None
        for k in range(1, 6):
            if _time_left() < 1.0:
                break
            found = _find_k_terms(n, k, smooths, smooth_set)
            if found is not None:
                break
        
        if found is None:
            # Try larger k if needed
            for k in range(6, 11):
                if _time_left() < 0.5:
                    break
                found = _find_k_terms(n, k, smooths, smooth_set)
                if found is not None:
                    break
        
        if found is None:
            # Emergency fallback
            found = _emergency_solution(n, smooths, smooth_set)
        
        answers[n - 1] = found
    
    return answers


def _emergency_solution(n, smooths, smooth_set):
    """Fallback: greedily build a valid antichain summing to n."""
    # Start with all 1s... no, must be distinct.
    # Use greedy: pick largest smooth that doesn't cause issues
    # Simple approach: use smooth numbers and check
    terms = []
    remaining = n
    # Try to use large smooths first for antichain property
    for a in reversed(smooths):
        if a >= remaining:
            continue
        # Check antichain with existing
        valid = True
        for t in terms:
            if a % t == 0 or t % a == 0:
                valid = False
                break
        if valid:
            terms.append(a)
            remaining -= a
            if remaining == 0:
                return terms
    # If greedy didn't work, try brute force with small k
    for k in range(1, 31):
        result = _find_k_terms(n, k, smooths, smooth_set)
        if result is not None:
            return result
    # Last resort: return [n] if n is smooth, else something
    if n in smooth_set:
        return [n]
    # This shouldn't happen for n <= 128
    return [1]
# EVOLVE-BLOCK-END
