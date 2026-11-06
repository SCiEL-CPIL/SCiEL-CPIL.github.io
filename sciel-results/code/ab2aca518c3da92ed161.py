"""
Integer sum-product extremal sets: minimize max(|A+A|, |A*A|) for 20 distinct positive ints <= 10000.
"""
import time
import random
import math
from itertools import combinations

# Deadline guard
_DEADLINE = time.time() + 85.0  # 85s safety margin (evaluator kills at 90s)

def _compute_sizes(A):
    """Compute |A+A| and |A*A| for a sorted list of distinct positive integers."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return len(sums), len(prods)

def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    s, p = _compute_sizes(A)
    return max(s, p)

def _gen_ap_seed():
    """Generate a random arithmetic progression seed of 20 distinct positive ints <= 10000."""
    # Pick a random start and step such that start + 19*step <= 10000
    max_step = 10000 // 19
    step = random.randint(1, max_step)
    start = random.randint(1, 10000 - 19 * step)
    A = [start + i * step for i in range(20)]
    return sorted(A)

def _gen_geometric_seed():
    """Generate a set with multiplicative structure (geometric-like)."""
    # Use powers of a small base, then perturb
    base = random.choice([2, 3, 4, 5])
    A = []
    for i in range(20):
        val = base ** i
        if val > 10000:
            # Cap and perturb
            val = 10000 - random.randint(0, 500)
        A.append(val)
    # Make distinct and sort
    A = sorted(set(A))
    while len(A) < 20:
        A.append(random.randint(1, 10000))
    A = sorted(set(A))[:20]
    return A

def _gen_mixed_seed():
    """Generate a mixed seed: some AP structure, some multiplicative."""
    A = list(range(1, 21))  # base
    # Perturb some elements
    for _ in range(random.randint(3, 8)):
        idx = random.randint(0, 19)
        A[idx] = random.randint(1, 10000)
    A = sorted(set(A))
    while len(A) < 20:
        A.append(random.randint(1, 10000))
    A = sorted(set(A))[:20]
    return A

def _local_search(A, max_iters=5000):
    """
    Local search: try replacing one element with another to reduce objective.
    Use a smooth surrogate (sum of squares of |A+A| and |A*A|) for tie-breaking.
    """
    A = list(A)
    A.sort()
    best_obj = _objective(A)
    best_A = list(A)
    
    # Precompute for speed: we'll recompute objective each time (20 elements -> 210 pairs, fast)
    no_improve = 0
    max_no_improve = 200
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Try random single-element replacement
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Try several candidate replacements
        found_improvement = False
        for _ in range(30):
            if time.time() > _DEADLINE:
                break
            new_val = random.randint(1, 10000)
            if new_val == old_val:
                continue
            # Check distinctness
            if new_val in A:
                continue
            
            A[idx] = new_val
            A.sort()
            obj = _objective(A)
            
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
                found_improvement = True
                no_improve = 0
                break
            else:
                A[idx] = old_val
                A.sort()
        
        if not found_improvement:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        
        A = list(best_A)
    
    return best_A, best_obj

def _simulated_annealing(A, max_iters=10000):
    """
    Simulated annealing to escape local minima.
    Uses the max(|A+A|, |A*A|) as objective, with occasional acceptance of worse moves.
    """
    A = list(A)
    A.sort()
    best_A = list(A)
    best_obj = _objective(A)
    current_obj = best_obj
    
    T = 10.0
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Propose move: replace one element
        idx = random.randint(0, 19)
        old_val = A[idx]
        new_val = random.randint(1, 10000)
        if new_val == old_val or new_val in A:
            continue
        
        A[idx] = new_val
        A.sort()
        new_obj = _objective(A)
        delta = new_obj - current_obj
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = list(A)
        else:
            A[idx] = old_val
            A.sort()
        
        T *= cooling
    
    return best_A, best_obj

def construct():
    """
    Construct 20 distinct positive integers <= 10000 minimizing max(|A+A|, |A*A|).
    Uses local search from AP seeds with simulated annealing for escape.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 85.0  # Reset deadline
    
    best_A = None
    best_score = float('inf')
    
    # Strategy 1: Local search from AP seeds
    for restart in range(30):
        if time.time() > _DEADLINE - 5:
            break
        
        seed = _gen_ap_seed()
        A, obj = _local_search(seed, max_iters=3000)
        if obj < best_score:
            best_score = obj
            best_A = list(A)
    
    # Strategy 2: Simulated annealing from best found
    if best_A is not None and time.time() < _DEADLINE - 10:
        A, obj = _simulated_annealing(best_A, max_iters=8000)
        if obj < best_score:
            best_score = obj
            best_A = list(A)
    
    # Strategy 3: More local search from mixed seeds
    for restart in range(20):
        if time.time() > _DEADLINE - 5:
            break
        
        seed = _gen_mixed_seed()
        A, obj = _local_search(seed, max_iters=2000)
        if obj < best_score:
            best_score = obj
            best_A = list(A)
    
    # Final polish: aggressive local search from best
    if best_A is not None and time.time() < _DEADLINE - 3:
        A, obj = _local_search(best_A, max_iters=5000)
        if obj < best_score:
            best_score = obj
            best_A = list(A)
    
    if best_A is None:
        # Fallback: simple AP
        best_A = list(range(1, 21))
    
    # Ensure validity: 20 distinct positive integers <= 10000
    best_A = sorted(set(best_A))
    while len(best_A) < 20:
        best_A.append(random.randint(1, 10000))
    best_A = sorted(set(best_A))[:20]
    
    return best_A
