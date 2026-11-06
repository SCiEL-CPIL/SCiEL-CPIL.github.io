import time
import random
import math

def compute_metrics(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    A is assumed to be a list of distinct positive integers.
    """
    n = len(A)
    sum_set = set()
    prod_set = set()
    
    # Using index to ensure i <= j for pairs
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
            
    return max(len(sum_set), len(prod_set))

def compute_metrics_fast(A):
    """
    Optimized version using local variables and direct set operations.
    """
    n = len(A)
    sum_set = set()
    prod_set = set()
    add_sum = sum_set.add
    add_prod = prod_set.add
    
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            add_sum(a + b)
            add_prod(a * b)
            
    return max(len(sum_set), len(prod_set))

def get_smooth_numbers(limit, max_val=10000):
    """
    Generate 2,3-smooth numbers (Hamming numbers) up to max_val.
    These have extreme product-set compression.
    """
    smooth = set()
    i = 1
    while i <= max_val:
        j = i
        while j <= max_val:
            smooth.add(j)
            j *= 3
            if j > max_val:
                break
        i *= 2
        if i > max_val:
            break
    return sorted(smooth)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Candidate pool: Small numbers are preferred for collisions
    # We go up to 2000 to allow some flexibility
    max_pool_val = 2000
    base_pool = list(range(1, max_pool_val + 1))
    
    # Generate smooth numbers for better initial candidates
    smooth_nums = get_smooth_numbers(20, 10000)
    
    # Generate some structured initial candidates
    initial_candidates = []
    
    # 1. Arithmetic Progressions
    initial_candidates.append(list(range(1, 21)))
    initial_candidates.append(list(range(2, 22)))
    initial_candidates.append([2*i for i in range(1, 21)])
    initial_candidates.append([3*i for i in range(1, 21)])
    
    # 2. Geometric Progressions
    initial_candidates.append([2**i for i in range(20)])
    initial_candidates.append([3**i for i in range(20) if 3**i <= 10000][:20]) # This might be <20, filter later
    
    # 3. Highly Composite Numbers (approx)
    # Numbers with many divisors
    hcn = [1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040, 7560]
    if len(hcn) >= 20 and all(x <= 10000 for x in hcn):
        initial_candidates.append(hcn[:20])
        
    # 4. Smooth number subsets - KEY IMPROVEMENT
    if len(smooth_nums) >= 20:
        # Take first 20 smooth numbers
        initial_candidates.append(smooth_nums[:20])
        # Take a mix
        for offset in range(0, min(5, len(smooth_nums) - 19)):
            initial_candidates.append(smooth_nums[offset:offset+20])
        # Random subsets of smooth numbers
        for _ in range(10):
            if len(smooth_nums) >= 20:
                cand = random.sample(smooth_nums, 20)
                initial_candidates.append(cand)
        
    # 5. Random sets from small range
    for _ in range(20):
        cand = random.sample(range(1, 101), 20)
        initial_candidates.append(cand)
        
    # 6. Random sets from medium range
    for _ in range(10):
        cand = random.sample(range(1, 501), 20)
        initial_candidates.append(cand)
        
    # 7. Mixed smooth + small random
    if len(smooth_nums) >= 15:
        for _ in range(5):
            cand = random.sample(smooth_nums[:50], 15)
            remaining = random.sample(range(1, 1000), 5)
            # Ensure no duplicates
            cand_set = set(cand)
            for r in remaining:
                if r not in cand_set:
                    cand.append(r)
                    cand_set.add(r)
            if len(cand) == 20:
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
        s = compute_metrics_fast(cand)
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
        # Bias towards smaller numbers and smooth numbers
        if random.random() < 0.4 and len(smooth_nums) > 0:
            new_val = random.choice(smooth_nums[:100])
        else:
            # Bias towards smaller numbers
            r = random.random()
            if r < 0.3:
                new_val = random.randint(1, 100)
            elif r < 0.6:
                new_val = random.randint(1, 500)
            else:
                new_val = random.choice(base_pool)
        
        # Check duplicates
        if new_val in current_set:
            # Try a few more times
            found = False
            for _ in range(5):
                if random.random() < 0.4 and len(smooth_nums) > 0:
                    new_val = random.choice(smooth_nums[:100])
                else:
                    new_val = random.choice(base_pool)
                if new_val not in current_set:
                    found = True
                    break
            if not found:
                continue
            
        current_set[idx] = new_val
        new_score = compute_metrics_fast(current_set)
        
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
                 
                 # Bias towards smooth numbers
                 if random.random() < 0.4 and len(smooth_nums) > 0:
                     new_v1 = random.choice(smooth_nums[:100])
                     new_v2 = random.choice(smooth_nums[:100])
                 else:
                     new_v1 = random.choice(base_pool)
                     new_v2 = random.choice(base_pool)
                 
                 # Ensure distinctness
                 if new_v1 == new_v2:
                     continue
                 if new_v1 in current_set or new_v2 in current_set:
                     continue
                     
                 current_set[idx1] = new_v1
                 current_set[idx2] = new_v2
                 new_score = compute_metrics_fast(current_set)
                 
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
        polish_deadline = min(deadline - 0.5, time.time() + 15.0)
        
        # Create a focused polish pool: smooth numbers + small integers
        polish_pool = set()
        for s in smooth_nums[:100]:
            if s <= 10000:
                polish_pool.add(s)
        for v in range(1, 300):
            polish_pool.add(v)
        polish_pool = sorted(polish_pool)
        
        best_set_set = set(best_set)
        
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                best_set_set.discard(old_val)
                
                for new_val in polish_pool:
                    if new_val in best_set_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics_fast(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                        best_set_set.add(new_val)
                    else:
                        best_set[i] = old_val
                        best_set_set.add(old_val)
                best_set[i] = old_val
                best_set_set.add(old_val)
                
        # One more pass with even smaller pool
        if time.time() < deadline - 0.5:
            small_pool = list(range(1, 100))
            for s in smooth_nums[:50]:
                if s <= 100 and s not in small_pool:
                    small_pool.append(s)
            small_pool = sorted(set(small_pool))
            
            best_set_set = set(best_set)
            for i in range(20):
                old_val = best_set[i]
                best_set_set.discard(old_val)
                for new_val in small_pool:
                    if new_val in best_set_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics_fast(best_set)
                    if s < best_score:
                        best_score = s
                        best_set_set.add(new_val)
                    else:
                        best_set[i] = old_val
                        best_set_set.add(old_val)
                best_set[i] = old_val
                best_set_set.add(old_val)
                
    return sorted(best_set)
