import random
import time
import sys

def _compute_objective(A):
    """Compute max(|A+A|, |A*A|) for a set A."""
    A = sorted(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _compute_set_sizes(A):
    """Return (|A+A|, |A*A|) for a set A."""
    A = sorted(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def _is_valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x < 1 or x > maximum for x in A):
        return False
    return True

def _local_search(A, max_iterations=50000, timeout=55.0):
    """Local search with swaps and nudges."""
    A = sorted(A)
    n = len(A)
    current_obj = _compute_objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    rng = random.Random(42)
    
    it = 0
    while it < max_iterations:
        if time.time() - start_time > timeout:
            break
        it += 1
        
        # Try a random move
        move_type = rng.random()
        if move_type < 0.5:
            # Swap two elements
            i, j = rng.sample(range(n), 2)
            new_A = list(A)
            new_A[i], new_A[j] = new_A[j], new_A[i]
            if not _is_valid(new_A):
                continue
        elif move_type < 0.8:
            # Nudge one element
            i = rng.randint(0, n-1)
            delta = rng.randint(-3, 3)
            if delta == 0:
                continue
            new_A = list(A)
            new_A[i] += delta
            if not _is_valid(new_A):
                continue
        else:
            # Replace one element with a random valid one
            i = rng.randint(0, n-1)
            new_val = rng.randint(1, 10000)
            new_A = list(A)
            new_A[i] = new_val
            if not _is_valid(new_A):
                continue
        
        new_obj = _compute_objective(new_A)
        
        if new_obj < current_obj:
            A = new_A
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = list(A)
    
    return best_A, best_obj

def _structured_init():
    """Generate a structured initial candidate."""
    # Try powers of 2 up to 10000, then fill with small numbers
    powers = [2**i for i in range(14)]  # 1 to 8192
    # Need 20 total, have 14, add 6 more small distinct numbers not in powers
    small = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    existing = set(powers)
    extra = []
    for s in small:
        if s not in existing:
            extra.append(s)
        if len(extra) == 6:
            break
    return sorted(powers + extra)

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s
    
    # Start with structured initialization
    A = _structured_init()
    best_A = list(A)
    best_obj = _compute_objective(A)
    
    # Local search from structured init
    if time.time() - start_time < deadline:
        A2, obj2 = _local_search(A, max_iterations=30000, timeout=deadline - (time.time() - start_time) - 2)
        if obj2 < best_obj:
            best_A = A2
            best_obj = obj2
    
    # Try a few more random restarts if time permits
    rng = random.Random(123)
    for restart in range(5):
        if time.time() - start_time > deadline:
            break
        
        # Random initial set biased toward small numbers
        A_init = set()
        while len(A_init) < 20:
            # Bias toward small numbers
            if rng.random() < 0.7:
                val = rng.randint(1, 50)
            else:
                val = rng.randint(1, 10000)
            A_init.add(val)
        A_init = sorted(A_init)
        
        A3, obj3 = _local_search(A_init, max_iterations=20000, timeout=deadline - (time.time() - start_time) - 2)
        if obj3 < best_obj:
            best_A = A3
            best_obj = obj3
    
    # Final polish: try to improve best_A with more iterations
    if time.time() - start_time < deadline - 2:
        remaining = deadline - (time.time() - start_time) - 1
        A4, obj4 = _local_search(best_A, max_iterations=50000, timeout=remaining)
        if obj4 < best_obj:
            best_A = A4
            best_obj = obj4
    
    return best_A
