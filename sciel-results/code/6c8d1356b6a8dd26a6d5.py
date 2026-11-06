import time
import random

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A. Uses symmetry to halve work."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set)), len(sum_set), len(prod_set)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Build candidate pool: small numbers, powers of primes, smooth numbers
    candidate_pool = set()
    for n in range(1, 201):
        candidate_pool.add(n)
    for p in [2, 3, 5, 7, 11, 13]:
        power = 1
        while power <= 10000:
            candidate_pool.add(power)
            power *= p
    for a in range(0, 14):
        v2 = 2**a
        if v2 > 10000: break
        for b in range(0, 10):
            v = v2 * (3**b)
            if v > 10000: break
            for c in range(0, 8):
                v2_ = v * (5**c)
                if v2_ > 10000: break
                for d in range(0, 5):
                    v3 = v2_ * (7**d)
                    if v3 > 10000: break
                    candidate_pool.add(v3)
    
    candidate_pool = sorted([x for x in candidate_pool if 1 <= x <= 10000])
    pool_size = len(candidate_pool)
    
    # Initial sets
    initial_sets = [
        list(range(1, 21)),
        [2**i for i in range(20)],
        sorted(set([2**i for i in range(10)] + [3**i for i in range(1, 11)])),
        list(candidate_pool[:20]),
        [1 + 2*i for i in range(20)],
        [i*i for i in range(1, 21)],
        sorted(set(list(range(1, 11)) + [2**i for i in range(3, 13)])),
        sorted(set([2**i for i in range(5)] + [3**i for i in range(5)] + [5**i for i in range(5)] + [7**i for i in range(5)])),
        sorted(set(list(range(1, 6)) + [2**i for i in range(2, 14)])),
    ]
    
    valid_initials = []
    for s in initial_sets:
        s = sorted(set(s))
        s = [x for x in s if 1 <= x <= 10000]
        if len(s) >= 20:
            valid_initials.append(s[:20])
    
    if not valid_initials:
        valid_initials = [list(range(1, 21))]
    
    best_score = float('inf')
    best_A = valid_initials[0][:]
    for s in valid_initials:
        score, _, _ = compute_metrics(s)
        if score < best_score:
            best_score = score
            best_A = s[:]
    
    # Simulated annealing
    current_A = best_A[:]
    current_score = best_score
    
    random.seed(42)
    T = 10.0
    T_min = 0.5
    cooling = 0.9995
    
    while time.time() < deadline:
        if time.time() >= deadline:
            break
        
        idx = random.randint(0, 19)
        new_val = candidate_pool[random.randint(0, pool_size - 1)]
        
        if new_val in current_A:
            T = max(T * cooling, T_min)
            continue
        
        trial_A = current_A[:]
        trial_A[idx] = new_val
        trial_A = sorted(set(trial_A))
        
        if len(trial_A) != 20:
            T = max(T * cooling, T_min)
            continue
        
        trial_score, _, _ = compute_metrics(trial_A)
        delta = trial_score - current_score
        
        if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 0.01)):
            current_A = trial_A[:]
            current_score = trial_score
        
        T = max(T * cooling, T_min)
        
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
    
    return best_A
