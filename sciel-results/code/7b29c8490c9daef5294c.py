import time
import random
import math
from itertools import combinations

def _compute_objective(A):
    """Compute max(|A+A|, |A*A|) for a set A."""
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        for j in range(i, n):
            sumset.add(A[i] + A[j])
            prodset.add(A[i] * A[j])
    return max(len(sumset), len(prodset))

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if x < 1 or x > 10000:
            return False
    return True

def _geometric_init(ratio, start):
    """Generate a geometric-like sequence."""
    A = []
    x = start
    for i in range(20):
        A.append(int(round(x)))
        x *= ratio
    # Ensure distinctness and bounds
    A = sorted(set(A))
    if len(A) < 20:
        # Fill in gaps
        for v in range(1, 10001):
            if v not in A and len(A) < 20:
                A.append(v)
            if len(A) >= 20:
                break
    A = sorted(A[:20])
    return A

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    random.seed(42)
    
    best_A = list(range(1, 21))
    best_obj = _compute_objective(best_A)
    
    # Try several geometric initializations
    for ratio in [1.05, 1.08, 1.1, 1.12, 1.15, 1.2, 1.25, 1.3]:
        for start in [2, 3, 4, 5, 6, 7, 8, 10]:
            A = _geometric_init(ratio, start)
            if _is_valid(A):
                obj = _compute_objective(A)
                if obj < best_obj:
                    best_obj = obj
                    best_A = A[:]
    
    # Local search / simulated annealing
    T = 50.0
    cooling = 0.999
    
    current_A = best_A[:]
    current_obj = best_obj
    
    while time.time() < deadline:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Swap two elements
            i, j = random.sample(range(20), 2)
            new_A = current_A[:]
            new_A[i], new_A[j] = new_A[j], new_A[i]
        elif move_type < 0.6:
            # Perturb one element
            i = random.randint(0, 19)
            delta = random.randint(-50, 50)
            new_val = current_A[i] + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_A:
                continue
            new_A = current_A[:]
            new_A[i] = new_val
        elif move_type < 0.8:
            # Replace one element with a random value
            i = random.randint(0, 19)
            new_val = random.randint(1, 10000)
            if new_val in current_A:
                continue
            new_A = current_A[:]
            new_A[i] = new_val
        else:
            # Geometric re-centering: shift the whole set
            shift = random.randint(-20, 20)
            new_A = [x + shift for x in current_A]
            if any(x < 1 or x > 10000 for x in new_A):
                continue
            if len(set(new_A)) != 20:
                continue
        
        if not _is_valid(new_A):
            continue
        
        new_obj = _compute_objective(new_A)
        
        # Acceptance criterion
        if new_obj < current_obj:
            current_A = new_A
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = current_A[:]
                # Reduce temperature on improvement
                T = max(T * 0.95, 1.0)
        else:
            # Accept with probability based on temperature
            delta = new_obj - current_obj
            if delta > 0:
                accept_prob = math.exp(-delta / max(T, 0.1))
                if random.random() < accept_prob:
                    current_A = new_A
                    current_obj = new_obj
        
        T *= cooling
        T = max(T, 0.5)
    
    return sorted(best_A)
