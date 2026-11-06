"""Optimized 3-AP-free subset construction for [1,256] maximizing reciprocal sum."""
import time
from fractions import Fraction

def _is_3ap_free(S):
    """Check if set S has no 3-term AP."""
    Sset = set(S)
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            a = S[i]
            b = S[j]
            c = 2*b - a
            if c in Sset and c > b:
                return False
    return True

def _reciprocal_sum(S):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total

def _valid_addition(S_set, x):
    """Check if adding x to S_set creates a 3-AP."""
    for a in S_set:
        if a < x:
            c = 2*x - a
            if c in S_set and c > x:
                return False
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S_set and a < b < x:
                    return False
        elif a > x:
            s = x + a
            if s % 2 == 0:
                b = s // 2
                if b in S_set and x < b < a:
                    return False
            c = 2*a - x
            if c in S_set and c > a:
                return False
    return True

def construct():
    deadline = time.time() + 85
    
    # Phase 1: Greedy construction
    selected = []
    S_set = set()
    for x in range(1, 257):
        if _valid_addition(S_set, x):
            selected.append(x)
            S_set.add(x)
    
    best_set = set(selected)
    best_score = _reciprocal_sum(selected)
    
    # Phase 2: Iterated local search with swaps
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for r in sorted(best_set, reverse=True):
            if time.time() >= deadline - 3:
                break
            for a in range(1, 257):
                if a in best_set:
                    continue
                new_set = best_set - {r} | {a}
                if _is_3ap_free(new_set):
                    new_score = _reciprocal_sum(new_set)
                    if new_score > best_score:
                        best_score = new_score
                        best_set = new_set
                        improved = True
                        break
    
    # Phase 3: Try adding elements
    if time.time() < deadline - 2:
        for a in range(1, 257):
            if a in best_set:
                continue
            if _valid_addition(best_set, a):
                best_set.add(a)
                best_score = _reciprocal_sum(best_set)
    
    # Phase 4: More aggressive swaps
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for r in sorted(best_set, reverse=True):
                if time.time() >= deadline - 2:
                    break
                for a in range(1, 257):
                    if a in best_set:
                        continue
                    new_set = best_set - {r} | {a}
                    if _is_3ap_free(new_set):
                        new_score = _reciprocal_sum(new_set)
                        if new_score > best_score:
                            best_score = new_score
                            best_set = new_set
                            improved = True
                            break
    
    return sorted(best_set)
