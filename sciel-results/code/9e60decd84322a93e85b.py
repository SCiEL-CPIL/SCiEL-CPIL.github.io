import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max possible d: 1 + 2*255 = 511 <= 512
    
    def time_ok():
        return time.time() < deadline

    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    # Fast delta: when replacing old_val with new_val
    def delta_replace(mask, old_val, new_val):
        """Return (delta, new_mask) when replacing old_val with new_val."""
        new_mask = mask & ~(1 << (old_val - 1)) | (1 << (new_val - 1))
        
        # Find d's that could be affected
        affected_ds = set()
        # d's involving old_val
        for d in range(1, MAX_D + 1):
            if old_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if old_val - d >= 1 and old_val + d <= MAX_VAL:
                affected_ds.add(d)
            if old_val - 2 * d >= 1:
                affected_ds.add(d)
        
        # d's involving new_val
        for d in range(1, MAX_D + 1):
            if new_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - d >= 1 and new_val + d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - 2 * d >= 1:
                affected_ds.add(d)
        
        delta = 0
        for d in affected_ds:
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        
        return delta, new_mask

    # Greedy construction for initial state
    def greedy_construct():
        mask = 0
        s_set = set()
        
        for _ in range(N):
            best_new = None
            best_s = -1
            # Sample candidates to speed up
            candidates = random.sample(range(1, MAX_VAL + 1), 100)
            for v in candidates:
                if v in s_set:
                    continue
                new_mask = mask | (1 << (v - 1))
                s = score_mask(new_mask)
                if s > best_s:
                    best_s = s
                    best_new = v
            
            if best_new is None:
                best_new = random.choice([v for v in range(1, MAX_VAL + 1) if v not in s_set])
            
            mask |= (1 << (best_new - 1))
            s_set.add(best_new)
        
        return mask, score_mask(mask), mask_to_list(mask)

    # Simulated Annealing
    def simulated_annealing(init_mask, init_score, init_list):
        current_mask = init_mask
        current_score = init_score
        current_list = init_list[:]
        current_set = set(current_list)
        
        best_mask = current_mask
        best_score = current_score
        best_list = current_list[:]
        
        # SA Parameters
        T = 10.0  # Initial temperature
        cooling_rate = 0.9995
        min_T = 0.01
        
        while time_ok() and T > min_T:
            # Pick a random element to replace
            idx = random.randint(0, N - 1)
            old_val = current_list[idx]
            
            # Pick a new value
            new_val = random.randint(1, MAX_VAL)
            if new_val == old_val or new_val in current_set:
                continue
            
            delta, new_mask = delta_replace(current_mask, old_val, new_val)
            new_score = current_score + delta
            
            # Acceptance criterion
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept move
                current_mask = new_mask
                current_score = new_score
                current_set.discard(old_val)
                current_set.add(new_val)
                current_list[idx] = new_val
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_list = current_list[:]
            
            T *= cooling_rate
            
            # Check time more frequently
            if not time_ok():
                break
        
        return best_mask, best_score, best_list

    random.seed(42)
    
    best_mask = 0
    best_score = 0
    best_list = []
    
    # Run multiple restarts of SA to explore different basins
    num_restarts = 10
    for i in range(num_restarts):
        if not time_ok():
            break
            
        # Use different initializations
        if i == 0:
            # Greedy start
            mask, score, lst = greedy_construct()
        elif i == 1:
            # Interval start
            lst = list(range(1, 33))
            mask = list_to_mask(lst)
            score = score_mask(mask)
        else:
            # Random start
            lst = random.sample(range(1, MAX_VAL + 1), N)
            mask = list_to_mask(lst)
            score = score_mask(mask)
        
        # Run SA
        m, s, l = simulated_annealing(mask, score, lst)
        
        if s > best_score:
            best_score = s
            best_mask = m
            best_list = l[:]
            
        print(f"SA Restart {i}: {s}", file=sys.stderr)
        
        # If we have time, we can do more restarts or longer SA
        # The loop above handles time via time_ok inside SA
    
    print(f"Final Score: {best_score}", file=sys.stderr)
    return best_list
