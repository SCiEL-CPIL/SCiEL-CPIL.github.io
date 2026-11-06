import time
import random
import math

def compute_metrics(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    A is assumed to be a list of distinct positive integers.
    """
    n = len(A)
    # Using set comprehensions for speed
    sum_set = {A[i] + A[j] for i in range(n) for j in range(i, n)}
    prod_set = {A[i] * A[j] for i in range(n) for j in range(i, n)}
    return max(len(sum_set), len(prod_set))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Candidate pool: Small numbers are preferred for collisions
    # We go up to 2000 to allow some flexibility
    max_pool_val = 2000
    base_pool = list(range(1, max_pool_val + 1))
    
    # Generate some structured initial candidates
    initial_candidates = []
    
    # 1. Arithmetic Progressions
    initial_candidates.append(list(range(1, 21)))
    initial_candidates.append(list(range(2, 22)))
    initial_candidates.append([2*i for i in range(1, 21)])
    initial_candidates.append([3*i for i in range(1, 21)])
    
    # 2. Geometric Progressions
    initial_candidates.append([2**i for i in range(20)])
    gp3 = [3**i for i in range(20)]
    if all(x <= 10000 for x in gp3):
        initial_candidates.append(gp3)
        
    # 3. Highly Composite Numbers (approx)
    # Numbers with many divisors
    hcn = [1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040, 7560]
    if len(hcn) >= 20 and all(x <= 10000 for x in hcn):
        initial_candidates.append(hcn[:20])
        
    # 4. Random sets from small range
    for _ in range(20):
        cand = random.sample(range(1, 101), 20)
        initial_candidates.append(cand)
        
    # 5. Random sets from medium range
    for _ in range(10):
        cand = random.sample(range(1, 501), 20)
        initial_candidates.append(cand)
        
    # 6. Biased Random Sets (favouring small numbers)
    for _ in range(10):
        cand = []
        while len(cand) < 20:
            val = int(random.random()**2 * 200) + 1
            if val not in cand:
                cand.append(val)
        initial_candidates.append(cand)

    # Validate and sort initial candidates
    valid_initials = []
    for cand in initial_candidates:
        if len(cand) == 20 and len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            valid_initials.append(sorted(cand))
            
    if not valid_initials:
        valid_initials = [sorted(random.sample(base_pool, 20))]
        
    # Evaluate initial candidates to find the best starting point
    best_set = None
    best_score = float('inf')
    current_set = None
    current_score = float('inf')
    
    for cand in valid_initials:
        s = compute_metrics(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]
            current_set = cand[:]
            current_score = s
            
    # Simulated Annealing
    T_start = 10.0
    T_end = 0.01
    # Estimate steps: ~5000 steps/sec? Let's aim for 50000 steps max
    # But bounded by time
    num_steps = 0
    max_steps = 1000000 # High cap, time is real limit
    
    # Pre-compute pool for faster random choice?
    # random.choice is fast enough.
    
    while time.time() < deadline - 2.0:
        step = num_steps
        num_steps += 1
        
        # Cooling schedule
        progress = min(1.0, num_steps / 50000.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Random Move: Replace one element
        idx = random.randint(0, 19)
        old_val = current_set[idx]
        
        # Choose new value
        # Bias towards smaller numbers using square
        new_val = int(random.random()**2 * max_pool_val) + 1
        
        # Check duplicates
        if new_val in current_set:
            # Try a few more times
            found = False
            for _ in range(5):
                new_val = int(random.random()**2 * max_pool_val) + 1
                if new_val not in current_set:
                    found = True
                    break
            if not found:
                continue
            
        current_set[idx] = new_val
        new_score = compute_metrics(current_set)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta <= 0 or (T > 0 and random.random() < min(1.0, -delta / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val
            
        # Occasionally do a swap
        if num_steps % 10 == 0 and time.time() < deadline - 2.0:
             idx1 = random.randint(0, 19)
             idx2 = random.randint(0, 19)
             if idx1 != idx2:
                 v1 = current_set[idx1]
                 v2 = current_set[idx2]
                 new_v1 = int(random.random()**2 * max_pool_val) + 1
                 new_v2 = int(random.random()**2 * max_pool_val) + 1
                 
                 # Ensure distinctness
                 if new_v1 == new_v2:
                     continue
                 if new_v1 in current_set or new_v2 in current_set:
                     continue
                     
                 current_set[idx1] = new_v1
                 current_set[idx2] = new_v2
                 new_score = compute_metrics(current_set)
                 
                 delta = new_score - current_score
                 if delta <= 0 or (T > 0 and random.random() < min(1.0, -delta / T)):
                     current_score = new_score
                     if new_score < best_score:
                         best_score = new_score
                         best_set = current_set[:]
                 else:
                     current_set[idx1] = v1
                     current_set[idx2] = v2

    # Final Polish: Hill Climbing on best_set
    if time.time() < deadline - 1.0:
        improved = True
        # Limit polish time
        polish_deadline = min(deadline - 0.5, time.time() + 10.0)
        # Use a slightly larger pool for polish: 1..1000
        polish_pool = list(range(1, 1001))
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                for new_val in polish_pool:
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
