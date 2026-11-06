import time
import random

MAX_VAL = 10000
N = 12

def _precompute():
    limit = 2 * MAX_VAL
    factors = [set() for _ in range(limit + 1)]
    for i in range(2, limit + 1):
        if not factors[i]:
            for j in range(i, limit + 1, i):
                factors[j].add(i)
    return factors

FACTORS = _precompute()

def _get_cost(arr):
    u = set()
    n = len(arr)
    # Local variable lookup is faster
    f = FACTORS
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            u.update(f[ai + arr[j]])
    return len(u)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_arr = None
    best_cost = float('inf')
    
    # 1. Structured Initializations
    # A: All even
    arr_a = list(range(2, 2*N + 1, 2))
    c_a = _get_cost(arr_a)
    if c_a < best_cost:
        best_cost = c_a
        best_arr = arr_a[:]
        
    # B: All multiples of 3
    arr_b = list(range(3, 3*N + 1, 3))
    c_b = _get_cost(arr_b)
    if c_b < best_cost:
        best_cost = c_b
        best_arr = arr_b[:]
        
    # C: Random
    for _ in range(20):
        arr_c = random.sample(range(1, MAX_VAL + 1), N)
        c_c = _get_cost(arr_c)
        if c_c < best_cost:
            best_cost = c_c
            best_arr = arr_c[:]
            
    # 2. Simulated Annealing
    current_arr = best_arr[:]
    current_cost = best_cost
    T = 5.0
    cooling = 0.99995
    
    while time.time() < deadline:
        i = random.randint(0, N - 1)
        old_val = current_arr[i]
        
        # Generate neighbor
        if random.random() < 0.7:
            # Local move
            step = random.randint(-50, 50)
            new_val = old_val + step
            if new_val < 1: new_val = 1
            if new_val > MAX_VAL: new_val = MAX_VAL
        else:
            # Global jump
            new_val = random.randint(1, MAX_VAL)
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        # Since N is small, simple check
        distinct = True
        for k in range(N):
            if k != i and current_arr[k] == new_val:
                distinct = False
                break
        if not distinct:
            continue
            
        current_arr[i] = new_val
        new_cost = _get_cost(current_arr)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = current_arr[:]
        else:
            delta = new_cost - current_cost
            if delta > 0:
                # exp(-delta/T)
                # delta is integer, T is float
                if T > 1e-6:
                    p = 2.718281828 ** (-delta / T)
                    if random.random() < p:
                        current_cost = new_cost
                    else:
                        current_arr[i] = old_val
                else:
                    current_arr[i] = old_val
                    
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_arr
