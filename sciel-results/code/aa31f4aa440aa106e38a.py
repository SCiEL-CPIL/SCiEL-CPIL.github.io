# EVOLVE-BLOCK-START
import time
import random
import math

def _set_sizes(A):
    """
    Compute |A+A| and |A*A| for a list A of distinct integers.
    Returns (sum_count, prod_count).
    """
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def _objective(A):
    s, p = _set_sizes(A)
    return max(s, p)

def _generate_smooth_numbers(max_val):
    """Generate all 7-smooth numbers (2^a * 3^b * 5^c) up to max_val."""
    smooths = set()
    a = 0
    while 2**a <= max_val:
        b = 0
        while 2**a * 3**b <= max_val:
            c = 0
            while 2**a * 3**b * 5**c <= max_val:
                smooths.add(2**a * 3**b * 5**c)
                c += 1
            b += 1
        a += 1
    return sorted(smooths)

def _simulated_annealing(A_init, n, max_val, deadline, pool=None):
    """
    Simulated annealing on the set A, using a pool of candidate values
    (e.g., 7-smooth numbers) to propose moves.
    """
    if pool is None:
        pool = list(range(1, max_val + 1))
    
    current_A = sorted(A_init[:])
    current_cost = _objective(current_A)
    best_A = current_A[:]
    best_cost = current_cost
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        if T < T_min:
            break
        
        # Pick a random index to replace
        idx = random.randint(0, n - 1)
        old_val = current_A[idx]
        
        # Propose a new value from the pool (avoiding current members)
        # Try a few times to get a valid candidate
        new_val = None
        for _ in range(10):
            candidate = random.choice(pool)
            if candidate != old_val and candidate not in current_A:
                new_val = candidate
                break
        
        if new_val is None:
            continue
        
        # Apply move
        current_A[idx] = new_val
        new_cost = _objective(current_A)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_A = current_A[:]
        else:
            # Reject move, revert
            current_A[idx] = old_val
        
        T *= cooling
    
    return best_A, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    n = 20
    max_val = 10000
    
    best_A = None
    best_cost = float('inf')
    
    # Generate pool of 7-smooth numbers
    smooth_pool = _generate_smooth_numbers(max_val)
    
    # Initial candidate: first 20 7-smooth numbers
    if len(smooth_pool) >= n:
        A_init = smooth_pool[:n]
        cost = _objective(A_init)
        if cost < best_cost:
            best_cost = cost
            best_A = A_init[:]
    
    # Run simulated annealing from this initial set
    if best_A and time.time() < deadline - 5:
        improved_A, improved_cost = _simulated_annealing(
            best_A, n, max_val, deadline, pool=smooth_pool
        )
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_A = improved_A[:]
    
    # If we still have time, try another SA run from a different initial set
    if time.time() < deadline - 10:
        # Try a mixed initial set: some smooth, some random
        A_init2 = smooth_pool[:n//2] + random.sample(smooth_pool[n//2:], n//2)
        A_init2 = sorted(A_init2)
        if len(set(A_init2)) == n:
            cost2 = _objective(A_init2)
            if cost2 < best_cost:
                best_cost = cost2
                best_A = A_init2[:]
            
            if time.time() < deadline - 5:
                improved_A2, improved_cost2 = _simulated_annealing(
                    A_init2, n, max_val, deadline, pool=smooth_pool
                )
                if improved_cost2 < best_cost:
                    best_cost = improved_cost2
                    best_A = improved_A2[:]
    
    return best_A

# EVOLVE-BLOCK-END
