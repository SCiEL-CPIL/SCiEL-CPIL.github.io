"""Optimized program for sum-product extremal sets using geometric progressions."""
import time
from math import log, exp, gcd
from itertools import combinations
import random

# EVOLVE-BLOCK-START
def _compute_objective(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    # Sum set: all a_i + a_j for i <= j
    sum_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sum_set.add(ai + A[j])
    # Product set: all a_i * a_j for i <= j
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            prod_set.add(ai * A[j])
    return max(len(sum_set), len(prod_set))

def _geometric_progression(a, r, n=20, max_val=10000):
    """Generate geometric progression a, a*r, a*r^2, ..., a*r^(n-1)."""
    A = []
    val = a
    for _ in range(n):
        if val > max_val:
            return None
        A.append(val)
        val *= r
    return A

def _is_valid_geometric(a, r, n=20, max_val=10000):
    """Check if geometric progression is valid (distinct, positive, <= max_val)."""
    A = _geometric_progression(a, r, n, max_val)
    if A is None:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    return True

def _evaluate_geometric(a, r, n=20):
    """Evaluate objective for a geometric progression."""
    A = _geometric_progression(a, r, n)
    if A is None:
        return float('inf')
    if len(set(A)) != n:
        return float('inf')
    return _compute_objective(A)

def _search_geometric_progressions(time_budget):
    """Search over geometric progressions to minimize objective."""
    start_time = time.time()
    best_obj = float('inf')
    best_A = list(range(1, 21))
    
    # For geometric progression with ratio r, |A*A| = n(n+1)/2 = 210 (minimum possible)
    # since all products a*r^i * a*r^j = a^2 * r^(i+j) are distinct for i+j distinct
    # So we need to minimize |A+A|
    
    # Try integer ratios r >= 2
    # For a given r, find smallest a such that a*r^19 <= 10000
    # a <= 10000 / r^19
    
    for r in range(2, 100):  # r up to 99
        if time.time() - start_time > time_budget - 5:
            break
        
        # Maximum a for this r
        max_a = 10000 // (r ** 19)
        if max_a < 1:
            continue
        
        # Try various values of a
        # For small r, we can try more values
        if r <= 3:
            a_values = range(1, max_a + 1)
        elif r <= 5:
            a_values = range(1, max_a + 1)
        else:
            # For larger r, max_a is small, try all
            a_values = range(1, max_a + 1)
        
        for a in a_values:
            if time.time() - start_time > time_budget - 5:
                break
            A = _geometric_progression(a, r, 20)
            if A is None or len(set(A)) != 20:
                continue
            obj = _compute_objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = A
    
    return best_A, best_obj

def _local_search(A, time_budget):
    """Perform local search to improve the solution."""
    start_time = time.time()
    current_obj = _compute_objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    n = len(A)
    max_val = 10000
    
    # Try single-element perturbations
    improved = True
    while improved and (time.time() - start_time < time_budget - 5):
        improved = False
        for i in range(n):
            if time.time() - start_time > time_budget - 5:
                break
            # Try replacing A[i] with nearby values
            old_val = A[i]
            for delta in range(-50, 51):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > max_val:
                    continue
                if new_val in A:
                    continue
                
                new_A = list(A)
                new_A[i] = new_val
                new_A = sorted(new_A)
                
                obj = _compute_objective(new_A)
                if obj < best_obj:
                    best_obj = obj
                    best_A = new_A
                    A = new_A
                    improved = True
                    break
    
    return best_A, best_obj

def construct():
    """Construct 20 distinct positive integers <= 10000 minimizing max(|A+A|, |A*A|)."""
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # Start with geometric progression search
    best_A, best_obj = _search_geometric_progressions(30)
    
    # Try to improve with local search if time permits
    if time.time() < deadline - 10:
        best_A, best_obj = _local_search(best_A, deadline - start_time)
    
    # Verify validity
    if len(best_A) != 20 or len(set(best_A)) != 20 or any(x < 1 or x > 10000 for x in best_A):
        # Fallback to simple range
        best_A = list(range(1, 21))
    
    return best_A
# EVOLVE-BLOCK-END
