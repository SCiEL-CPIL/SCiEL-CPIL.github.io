"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _eval_set(A):
    """Compute max(|A+A|, |A*A|) for a list of 20 distinct positive ints."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _generate_pool():
    """Generate a diverse pool of candidate numbers."""
    pool = set()
    # Smooth numbers (2,3,5,7-smooth up to 10000)
    def gen_smooth(current, primes, idx, max_val=10000):
        if current > 1:
            pool.add(current)
        for i in range(idx, len(primes)):
            p = primes[i]
            nxt = current * p
            if nxt > max_val:
                break
            gen_smooth(nxt, primes, i, max_val)
    gen_smooth(1, [2, 3, 5, 7], 0)
    
    # Add arithmetic progression numbers
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for k in range(1, 1000):
            v = base * k
            if v <= 10000:
                pool.add(v)
    
    # Add some random numbers for diversity
    for _ in range(200):
        pool.add(random.randint(1, 10000))
    
    return sorted(pool)

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    pool = _generate_pool()
    
    # Smart initialization: mix of small smooth, medium, and large numbers
    small = [x for x in pool if x <= 100]
    medium = [x for x in pool if 100 < x <= 1000]
    large = [x for x in pool if x > 1000]
    
    random.shuffle(small)
    random.shuffle(medium)
    random.shuffle(large)
    
    current = sorted(small[:8] + medium[:6] + large[:6])
    while len(current) < 20:
        for x in pool:
            if x not in set(current):
                current.append(x)
                break
        current = sorted(current)
    current = current[:20]
    
    current_score = _eval_set(current)
    best = list(current)
    best_score = current_score
    
    # Simulated annealing with swap moves
    T = 100.0
    T_min = 0.5
    cooling = 0.9997
    
    while time.time() < deadline and T > T_min:
        if time.time() >= deadline - 3:
            break
        
        idx = random.randint(0, 19)
        old_val = current[idx]
        
        for attempt in range(20):
            new_val = random.choice(pool)
            if new_val != old_val and new_val not in current:
                break
        else:
            new_val = random.randint(1, 10000)
            if new_val in current or new_val == old_val:
                continue
        
        neighbor = list(current)
        neighbor[idx] = new_val
        neighbor.sort()
        
        new_score = _eval_set(neighbor)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < np.exp(-delta / T):
            current = neighbor
            current_score = new_score
            if current_score < best_score:
                best = list(current)
                best_score = current_score
        
        T *= cooling
    
    return best
# EVOLVE-BLOCK-END
