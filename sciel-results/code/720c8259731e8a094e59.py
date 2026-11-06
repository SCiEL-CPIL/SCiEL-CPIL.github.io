"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations, product

# EVOLVE-BLOCK-START
def _compute_objective(A):
    """Compute max(|A+A|, |A*A|) for sorted list A of distinct positive integers."""
    A = sorted(A)
    # A+A: all sums a_i + a_j for i <= j
    sums = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
    # A*A: all products a_i * a_j for i <= j
    prods = set()
    for i in range(n):
        for j in range(i, n):
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _is_valid(A):
    """Check if A is a valid set of 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _geometric_start(ratio, start_val=1):
    """Generate a geometric progression-based start."""
    A = []
    val = start_val
    for _ in range(20):
        A.append(int(val))
        val *= ratio
    # Ensure distinct and within bounds
    A = sorted(set(int(x) for x in A if 1 <= int(x) <= 10000))
    # If we don't have 20, fill in with random values
    while len(A) < 20:
        candidate = random.randint(1, 10000)
        if candidate not in A:
            A.append(candidate)
    return sorted(A[:20])

def _random_start():
    """Generate a random valid set."""
    A = random.sample(range(1, 10001), 20)
    return sorted(A)

def _neighborhood_move(A, move_type='replace'):
    """Generate a neighbor of A."""
    A = list(A)
    if move_type == 'replace':
        idx = random.randint(0, 19)
        A[idx] = random.randint(1, 10000)
    elif move_type == 'swap':
        i, j = random.sample(range(20), 2)
        A[i], A[j] = A[j], A[i]
    elif move_type == 'shift':
        idx = random.randint(0, 19)
        delta = random.randint(-10, 10)
        A[idx] = max(1, min(10000, A[idx] + delta))
    elif move_type == 'scale':
        factor = random.uniform(0.8, 1.2)
        for i in range(20):
            A[i] = max(1, min(10000, int(A[i] * factor)))
        # Ensure distinct
        A = list(set(A))
        while len(A) < 20:
            candidate = random.randint(1, 10000)
            if candidate not in A:
                A.append(candidate)
        A = sorted(A[:20])
    return sorted(A)

def _local_search(A, max_iters=500, time_limit=5.0):
    """Perform local search with simulated annealing."""
    A = list(A)
    current_obj = _compute_objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    T = 10.0
    cooling = 0.99
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Generate neighbor
        move_type = random.choice(['replace', 'swap', 'shift', 'scale'])
        new_A = _neighborhood_move(A, move_type)
        
        if not _is_valid(new_A):
            continue
        
        new_obj = _compute_objective(new_A)
        delta = new_obj - current_obj
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            A = new_A
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = list(A)
        
        T *= cooling
    
    return best_A, best_obj

def _refine(A, max_rounds=50, time_limit=5.0):
    """Refine A by trying to reduce the objective further."""
    A = list(A)
    current_obj = _compute_objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    for _ in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Try replacing A[i] with nearby values
            original = A[i]
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = original + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A and new_val != original:
                    continue
                
                A[i] = new_val
                if not _is_valid(A):
                    A[i] = original
                    continue
                
                new_obj = _compute_objective(A)
                if new_obj < current_obj:
                    current_obj = new_obj
                    best_obj = new_obj
                    best_A = list(A)
                    improved = True
                else:
                    A[i] = original
            
            if not improved and random.random() < 0.5:
                break
        
        if not improved:
            break
    
    return best_A, best_obj

def construct():
    """Main construct function with time budget guard."""
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit
    
    random.seed(42)
    np.random.seed(42)
    
    best_A = None
    best_obj = float('inf')
    
    # Try multiple starting points
    starts = []
    
    # Geometric progressions with various ratios
    for ratio in [1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        starts.append(_geometric_start(ratio))
    
    # Random starts
    for _ in range(5):
        starts.append(_random_start())
    
    # Try each start with local search
    for i, start in enumerate(starts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        if not _is_valid(start):
            continue
        
        # Local search
        local_A, local_obj = _local_search(start, max_iters=300, time_limit=5.0)
        
        if local_obj < best_obj:
            best_obj = local_obj
            best_A = list(local_A)
        
        # Refine
        refined_A, refined_obj = _refine(local_A, max_rounds=20, time_limit=5.0)
        
        if refined_obj < best_obj:
            best_obj = refined_obj
            best_A = list(refined_A)
    
    # Final polish on the best solution
    if best_A is not None:
        final_A, final_obj = _refine(best_A, max_rounds=30, time_limit=10.0)
        if final_obj < best_obj:
            best_obj = final_obj
            best_A = list(final_A)
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
# EVOLVE-BLOCK-END
