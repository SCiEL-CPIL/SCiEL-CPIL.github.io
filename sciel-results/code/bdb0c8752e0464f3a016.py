import time
import random
import itertools
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for the 90s limit
    
    rng = random.Random(42)
    
    def calc_score_fast(A):
        A = sorted(A)
        sumset = set()
        prodset = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
        
    # Generate candidate pool
    # Focus on smooth numbers to keep product sets small
    pool = set()
    # Add small integers
    for i in range(1, 51):
        pool.add(i)
    # Add 2,3-smooth numbers
    for a in range(14):
        for b in range(10):
            val = (2**a) * (3**b)
            if val <= 10000:
                pool.add(val)
    # Add 2,3,5-smooth numbers
    for a in range(14):
        for b in range(8):
            for c in range(6):
                val = (2**a) * (3**b) * (5**c)
                if val <= 10000:
                    pool.add(val)
    # Add some random numbers to allow exploration
    for _ in range(500):
        pool.add(rng.randint(1, 10000))
    
    pool = list(pool)
    rng.shuffle(pool)
    
    candidates = []
    
    # Candidate 1: AP 1..20
    candidates.append(list(range(1, 21)))
    
    # Candidate 2: Powers of 2 + 3*Powers of 2
    A2 = []
    for k in range(10):
        A2.append(2**k)
    for k in range(10):
        A2.append(3 * 2**k)
    if len(set(A2)) == 20:
        candidates.append(A2)
        
    # Candidate 3: Small smooth numbers
    smooth = [v for v in pool if v <= 100]
    smooth.sort()
    if len(smooth) >= 20:
        candidates.append(smooth[:20])
        
    # Evaluate initial candidates
    best_A = None
    best_score = float('inf')
    for cand in candidates:
        if len(set(cand)) != 20:
            continue
        s = calc_score_fast(cand)
        if s < best_score:
            best_score = s
            best_A = list(cand)
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = calc_score_fast(best_A)
        
    # Simulated Annealing
    current_A = list(best_A)
    current_score = best_score
    
    T = 10.0 # Initial temperature
    cooling = 0.995
    
    # We want to find a set with score < best_score
    # The score is integer. 
    # We accept worse moves with probability exp(-delta / T)
    
    while time.time() < deadline - 2.0:
        # Pick a random index to replace
        idx = rng.randint(0, 19)
        old_val = current_A[idx]
        
        # Pick a new value from pool, ensuring it's not in current_A
        new_val = None
        for _ in range(50): # Try a few times to find a valid replacement
            candidate_val = rng.choice(pool)
            if candidate_val != old_val and candidate_val not in current_A:
                new_val = candidate_val
                break
        
        if new_val is None:
            continue
            
        new_A = current_A.copy()
        new_A[idx] = new_val
        new_score = calc_score_fast(new_A)
        
        delta = new_score - current_score
        
        if delta <= 0:
            # Improvement or equal
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = list(current_A)
        else:
            # Worse move, accept with probability exp(-delta / T)
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_A = new_A
                    current_score = new_score
                    
        T *= cooling
        
    # Final polish: Greedy local search to ensure we are at a local minimum
    improved = True
    while improved and time.time() < deadline - 0.5:
        improved = False
        for i in range(len(current_A)):
            if time.time() > deadline - 0.5:
                break
            old_val = current_A[i]
            # Try all candidates in pool (or a subset if pool is huge, but pool is ~1000)
            # To speed up, we can skip if we've tried many recently, but let's just iterate
            for new_val in pool:
                if new_val == old_val or new_val in current_A:
                    continue
                new_A = current_A.copy()
                new_A[i] = new_val
                new_score = calc_score_fast(new_A)
                if new_score < current_score:
                    current_score = new_score
                    current_A = new_A
                    improved = True
                    break
            if time.time() > deadline - 0.5:
                break
        if current_score < best_score:
            best_score = current_score
            best_A = list(current_A)

    return best_A
