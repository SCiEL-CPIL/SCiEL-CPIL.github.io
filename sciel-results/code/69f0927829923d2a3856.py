"""
Combined approach for the reciprocal-subsum-free set task.

Structural idea (from Inspiration 2):
  - Use incremental validity checks when adding a single element to a known-valid set.
  - This is much faster than re-checking the whole set from scratch.
  - Use local search (remove-1-add-1, remove-1-add-2) to escape local optima.

Refinement machinery (from Current Program):
  - Backtracking search with strong pruning based on remaining candidates.
  - Meet-in-the-middle for subset-sum checks.
  - Tight deadline management.

The key insight is that the constraint is hereditary (subsets of valid sets are valid),
so we can build up sets incrementally and use local search to improve.
"""
import time
from fractions import Fraction

_START = None
DEADLINE = 85.0  # Safety margin under 90s kill


def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _subset_sum_possible(vals, target):
    """True iff some subset of `vals` (Fractions) sums exactly to `target`.
    Meet-in-the-middle, O(2^(n/2))."""
    n = len(vals)
    if n == 0:
        return target == Fraction(0)
    if n <= 1:
        return any(v == target for v in vals)
    
    # Quick upper bound check: if sum of all vals < target, impossible
    total = sum(vals)
    if total < target:
        return False
    
    mid = n // 2
    left, right = vals[:mid], vals[mid:]
    
    # Generate all subset sums for left
    L = {Fraction(0)}
    for v in left:
        L = L | {x + v for x in L}
    
    # Generate all subset sums for right
    R = {Fraction(0)}
    for v in right:
        R = R | {x + v for x in R}
    
    # Check if any combination matches target
    if len(L) <= len(R):
        for l in L:
            if (target - l) in R:
                return True
    else:
        for r in R:
            if (target - r) in L:
                return True
    return False


def _check_validity_incremental(current_set, new_elem):
    """Check if adding new_elem to current_set preserves validity.
    Only need to check:
    1. 1/new_elem is not a subsum of reciprocals of current_set
    2. For each a in current_set, 1/a is not a subsum of reciprocals of 
       (current_set - {a}) + {new_elem}
    """
    new_recip = Fraction(1, new_elem)
    current_recips = [Fraction(1, v) for v in current_set]
    
    # Check 1: new element's reciprocal vs. current set
    if _subset_sum_possible(current_recips, new_recip):
        return False
    
    # Check 2: each existing element's reciprocal vs. (others + new)
    for a in current_set:
        a_recip = Fraction(1, a)
        others = [Fraction(1, v) for v in current_set if v != a]
        others.append(new_recip)
        if _subset_sum_possible(others, a_recip):
            return False
    
    return True


def _is_valid(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    if not S:
        return False
    S_list = list(S)
    recips = {v: Fraction(1, v) for v in S_list}
    for a in S_list:
        others = [recips[v] for v in S_list if v != a]
        if _subset_sum_possible(others, recips[a]):
            return False
    return True


def _greedy_add(best, all_elems):
    """Try to add elements greedily to best."""
    improved = True
    while improved and not _deadline_hit():
        improved = False
        best_set = set(best)
        for v in all_elems:
            if _deadline_hit():
                break
            if v in best_set:
                continue
            if _check_validity_incremental(best, v):
                best = best + [v]
                improved = True
                break
    return best


def _local_search(best):
    """Local search: try remove-1-add-1 and remove-1-add-2 swaps."""
    all_elems = set(range(1, 25))
    improved = True
    while improved and not _deadline_hit():
        improved = False
        best_set = set(best)
        
        # Try remove-1-add-1
        for i in range(len(best)):
            if _deadline_hit():
                break
            for j in range(1, 25):
                if _deadline_hit():
                    break
                if j in best_set:
                    continue
                candidate = [x for x in best if x != best[i]] + [j]
                if _is_valid(candidate) and len(candidate) > len(best):
                    best = candidate
                    best_set = set(best)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # Try remove-1-add-2 (net gain of 1)
        for i in range(len(best)):
            if _deadline_hit():
                break
            for j in range(1, 25):
                if _deadline_hit():
                    break
                if j in best_set:
                    continue
                for k in range(j + 1, 25):
                    if _deadline_hit():
                        break
                    if k in best_set:
                        continue
                    candidate = [x for x in best if x != best[i]] + [j, k]
                    if _is_valid(candidate) and len(candidate) > len(best):
                        best = candidate
                        best_set = set(best)
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best


def _backtrack_search(candidates, initial_set, best_size):
    """Backtracking search to find larger valid sets."""
    best_set = initial_set[:]
    
    def backtrack(idx, S):
        nonlocal best_set, best_size
        
        if _deadline_hit():
            return
            
        # Pruning: even if we take all remaining, can we beat best?
        if len(S) + (len(candidates) - idx) <= best_size:
            return
            
        if idx == len(candidates):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
            
        v = candidates[idx]
        
        # Option 1: Include candidates[idx]
        if _check_validity_incremental(S, v):
            S.append(v)
            backtrack(idx + 1, S)
            S.pop()
            
        # Option 2: Exclude candidates[idx]
        backtrack(idx + 1, S)
        
    backtrack(0, initial_set[:])
    return best_set


def construct():
    global _START
    _START = time.time()
    
    # Start with a good initial solution: top half
    best = list(range(13, 25))
    
    # Greedy add to expand
    all_elems = list(range(1, 25))
    best = _greedy_add(best, all_elems)
    
    # Local search to improve
    best = _local_search(best)
    
    # Backtracking search for further improvement
    # Try to add more elements via backtracking
    current_size = len(best)
    if not _deadline_hit() and current_size < 24:
        # Build a candidate list of elements not in best
        candidates = [v for v in range(1, 25) if v not in best]
        if candidates:
            new_best = _backtrack_search(candidates, best, current_size)
            if len(new_best) > len(best):
                best = new_best
                best = _local_search(best)
    
    # Safety: ensure nonempty
    if not best:
        best = [24]
        
    return best
