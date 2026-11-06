import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) efficiently."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            s = ai + aj
            p = ai * aj
            sum_set.add(s)
            if i != j:
                sum_set.add(aj + ai)  # same value, but keep logic clear
                prod_set.add(aj * ai)
            prod_set.add(p)
    # Actually, since addition and multiplication are commutative,
    # we can just do all pairs including i==j
    sum_set = set()
    prod_set = set()
    for i in range(n):
        for j in range(n):
            sum_set.add(A[i] + A[j])
            prod_set.add(A[i] * A[j])
    return max(len(sum_set), len(prod_set)), len(sum_set), len(prod_set)

def compute_metrics_fast(A):
    """Faster version: only compute max, not individual sizes."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set))

def generate_smooth_pool():
    """Generate a pool of smooth numbers (2,3,5,7-smooth) up to 10000."""
    pool = set()
    # 2-smooth
    v = 1
    while v <= 10000:
        pool.add(v)
        v *= 2
    # 2,3-smooth
    for a in range(0, 15):
        v2 = 2**a
        if v2 > 10000:
            break
        for b in range(0, 15):
            v = v2 * (3**b)
            if v > 10000:
                break
            pool.add(v)
    # 2,3,5-smooth
    for a in range(0, 15):
        v2 = 2**a
        if v2 > 10000:
            break
        for b in range(0, 15):
            v23 = v2 * (3**b)
            if v23 > 10000:
                break
            for c in range(0, 15):
                v = v23 * (5**c)
                if v > 10000:
                    break
                pool.add(v)
    # 2,3,5,7-smooth
    for a in range(0, 15):
        v2 = 2**a
        if v2 > 10000:
            break
        for b in range(0, 15):
            v23 = v2 * (3**b)
            if v23 > 10000:
                break
            for c in range(0, 15):
                v235 = v23 * (5**c)
                if v235 > 10000:
                    break
                for d in range(0, 15):
                    v = v235 * (7**d)
                    if v > 10000:
                        break
                    pool.add(v)
    # Add small integers
    for i in range(1, 101):
        pool.add(i)
    # Add some random numbers for diversity
    for _ in range(200):
        pool.add(random.randint(1, 10000))
    return list(pool)

def local_search_polish(A, pool, deadline, max_iters=10000):
    """Systematic neighborhood exploration: try all single swaps."""
    current_A = A[:]
    current_score = compute_metrics_fast(current_A)
    best_A = current_A[:]
    best_score = current_score
    pool_set = set(pool)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(20):
            if time.time() > deadline:
                break
            for c in pool:
                if c in current_A:
                    continue
                new_A = current_A[:]
                new_A[i] = c
                new_score = compute_metrics_fast(new_A)
                if new_score < current_score:
                    current_A = new_A
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_A = current_A[:]
                    break  # restart inner loop with improved set
            if improved:
                break
        if not improved:
            break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Generate candidate pool
    pool = generate_smooth_pool()
    
    # Initial set: start with small integers
    A = list(range(1, 21))
    best_score = compute_metrics_fast(A)
    best_A = A[:]
    
    # Simulated annealing phase
    current_A = A[:]
    current_score = best_score
    
    T = 20.0
    cooling = 0.998
    
    iterations = 0
    while time.time() < deadline - 10.0:  # Reserve 10s for polish
        i = random.randint(0, 19)
        c = random.choice(pool)
        
        if c in current_A:
            continue
            
        new_A = current_A[:]
        new_A[i] = c
        new_score = compute_metrics_fast(new_A)
        
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            delta = new_score - current_score
            if delta <= 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)
            
            if accept:
                current_A = new_A
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A[:]
        
        T *= cooling
        if T < 0.5:
            T = 0.5
        
        # Occasionally restart from best to escape local minima
        if random.random() < 0.02:
            current_A = best_A[:]
            current_score = best_score
        
        iterations += 1
    
    # Final polish phase: systematic local search
    polish_deadline = deadline - 2.0  # 2s safety margin
    best_A, best_score = local_search_polish(best_A, pool, polish_deadline, max_iters=5000)
    
    return best_A
