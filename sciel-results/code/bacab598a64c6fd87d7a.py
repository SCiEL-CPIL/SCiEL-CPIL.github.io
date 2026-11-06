import time
import random
from itertools import combinations

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Candidate pool: small numbers, multiples, powers, etc.
    # Focus on numbers that create structure
    base_pool = []
    # Small numbers
    for i in range(1, 101):
        base_pool.append(i)
    # Multiples of small bases
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for k in range(1, 21):
            v = base * k
            if v <= 10000 and v not in base_pool:
                base_pool.append(v)
    # Powers of 2
    for p in range(1, 14):
        v = 2 ** p
        if v <= 10000 and v not in base_pool:
            base_pool.append(v)
    # Powers of 3
    for p in range(1, 9):
        v = 3 ** p
        if v <= 10000 and v not in base_pool:
            base_pool.append(v)
    # Some structured sets
    for base in [12, 18, 24, 30, 36, 48, 60]:
        for k in range(1, 16):
            v = base * k
            if v <= 10000 and v not in base_pool:
                base_pool.append(v)
    
    base_pool = list(set(base_pool))
    base_pool.sort()
    
    # Initial solution: try a few structured starting points
    initial_candidates = [
        list(range(1, 21)),
        [2*i for i in range(1, 21)],
        [3*i for i in range(1, 21)],
        [4*i for i in range(1, 21)],
        [5*i for i in range(1, 21)],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 2, 4, 8, 16, 32, 64][:20],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 3, 6, 12, 24, 48, 96],
    ]
    
    # Ensure all are valid
    valid_initials = []
    for cand in initial_candidates:
        if len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            valid_initials.append(sorted(cand))
    
    if not valid_initials:
        valid_initials = [sorted(base_pool[:20])]
    
    # Evaluate initial candidates
    best_set = None
    best_score = float('inf')
    for cand in valid_initials:
        s = compute_metrics(cand)
        if s < best_score:
            best_score = s
            best_set = cand
    
    # Simulated annealing
    current_set = best_set[:]
    current_score = best_score
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.01
    num_steps = 50000
    
    step = 0
    while step < num_steps:
        if time.time() > deadline - 5.0:
            break
        
        step += 1
        # Linear cooling
        T = T_start * (T_end / T_start) ** (step / num_steps)
        
        # Random move: replace one element with another from pool
        idx = random.randint(0, 19)
        old_val = current_set[idx]
        
        # Choose a new value from the pool, avoiding duplicates
        new_val = None
        for _ in range(20):
            candidate = random.choice(base_pool)
            if candidate not in current_set:
                new_val = candidate
                break
        if new_val is None:
            continue
        
        # Compute new score efficiently
        current_set[idx] = new_val
        new_score = compute_metrics(current_set)
        
        # Accept or reject
        delta = new_score - current_score
        if delta <= 0 or random.random() < min(1.0, (-delta) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val
        
        # Occasionally do a 2-swap
        if step % 5 == 0:
            idx1 = random.randint(0, 19)
            idx2 = random.randint(0, 19)
            if idx1 != idx2:
                v1 = current_set[idx1]
                v2 = current_set[idx2]
                new_v1 = random.choice(base_pool)
                new_v2 = random.choice(base_pool)
                if new_v1 not in current_set and new_v2 not in current_set and new_v1 != new_v2:
                    current_set[idx1] = new_v1
                    current_set[idx2] = new_v2
                    new_score = compute_metrics(current_set)
                    delta = new_score - current_score
                    if delta <= 0 or random.random() < min(1.0, (-delta) / T):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_set = current_set[:]
                    else:
                        current_set[idx1] = v1
                        current_set[idx2] = v2
    
    # Final polish: try local improvements
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                for new_val in base_pool:
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_set[i] = old_val
                best_set[i] = old_val
    
    return sorted(best_set)
