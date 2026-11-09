import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_set = None
    best_score = float('inf')
    
    # Helper to compute score
    def get_score(A):
        sums = set()
        prods = set()
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
        
    # Helper to check validity
    def is_valid(A):
        if len(A) != 20: return False
        if len(set(A)) != 20: return False
        if any(x < 1 or x > 10000 for x in A): return False
        return True

    # Initial candidates
    candidates = []
    
    # 1. Arithmetic progression
    candidates.append(list(range(1, 21)))
    
    # 2. Geometric progression (scaled to fit)
    # 1, 2, 4, ... 2^19 is too big.
    # Let's use ratio 1.1? No, integers.
    # Use 5-smooth numbers
    smooth = []
    a, b, c = 0, 0, 0
    while len(smooth) < 20:
        val = (2**a) * (3**b) * (5**c)
        if val <= 10000:
            if val not in smooth:
                smooth.append(val)
        # Advance to next 5-smooth number (simple generation)
        # This simple loop is inefficient, let's just hardcode or generate properly
        # Better: generate all 5-smooth <= 10000 and take first 20
        pass
    
    # Proper 5-smooth generation
    smooth_nums = []
    i2, i3, i5 = 0, 0, 0
    next2, next3, next5 = 1, 3, 5
    smooth_nums = [1]
    while len(smooth_nums) < 20:
        m = min(next2, next3, next5)
        if m == next2:
            i2 += 1
            next2 = smooth_nums[i2-1] * 2
        if m == next3:
            i3 += 1
            next3 = smooth_nums[i3-1] * 3
        if m == next5:
            i5 += 1
            next5 = smooth_nums[i5-1] * 5
        smooth_nums.append(m)
        if m > 10000: break
    if len(smooth_nums) >= 20:
        candidates.append(smooth_nums[:20])

    # 3. Random
    for _ in range(10):
        candidates.append(random.sample(range(1, 10001), 20))
        
    # Evaluate initial candidates
    for A in candidates:
        if is_valid(A):
            score = get_score(A)
            if score < best_score:
                best_score = score
                best_set = A[:]
                
    # Simulated Annealing
    # We will run multiple restarts
    num_restarts = 20
    steps_per_restart = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start from best found so far or a random perturbation of best
        if r == 0:
            A = best_set[:]
        else:
            # Perturb best set
            A = best_set[:]
            for _ in range(5):
                idx = random.randint(0, 19)
                A[idx] = random.randint(1, 10000)
            if not is_valid(A):
                continue
                
        score = get_score(A)
        T = 10.0
        T_min = 0.01
        decay = 0.99
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Pick a random index to change
            idx = random.randint(0, 19)
            old_val = A[idx]
            
            # Generate new value
            # Mix of local and global moves
            if random.random() < 0.5:
                # Local move: small change
                delta = random.randint(-50, 50)
                new_val = old_val + delta
            else:
                # Global move
                new_val = random.randint(1, 10000)
                
            # Clamp
            if new_val < 1: new_val = 1
            if new_val > 10000: new_val = 10000
            
            if new_val == old_val:
                continue
                
            # Check distinctness
            if new_val in A:
                # If new_val is already in A (and not at idx), it's invalid
                # We need to check if it's at idx? No, we are replacing A[idx].
                # So if new_val is in A and A[A.index(new_val)] != A[idx] (which is old_val), it's a collision.
                # Simpler: create a set of A without old_val, check if new_val is in it.
                # But A is a list.
                # Let's just check:
                if new_val in A:
                    # Check if it's the same element?
                    # If new_val == old_val, skipped above.
                    # So if new_val in A, it must be a different element.
                    continue
                    
            A[idx] = new_val
            new_score = get_score(A)
            
            delta_score = new_score - score
            
            if delta_score < 0 or random.random() < pow(2.718, -delta_score / T):
                score = new_score
                # Accept
                if score < best_score:
                    best_score = score
                    best_set = A[:]
            else:
                # Reject
                A[idx] = old_val
                
            T *= decay
            if T < T_min:
                T = T_min
                
    return best_set
