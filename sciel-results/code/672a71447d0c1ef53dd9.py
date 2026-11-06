import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 512
    K = 32
    
    def score_bitmask(mask):
        count = 0
        # d can be at most (N-1)//2 = 255
        # If max element is 511, min is 0, max d is 255
        for d in range(1, 256):
            # Check if there exists a such that a, a+d, a+2d are in mask
            # mask & (mask >> d) & (mask >> 2*d)
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def random_set():
        s = set(random.sample(range(N), K))
        mask = 0
        for x in s:
            mask |= (1 << x)
        return s, mask, score_bitmask(mask)

    # Multi-start or single start with long SA?
    # Given 110s, a single long SA with good cooling might be enough, 
    # but multi-start with shorter runs might find better basins.
    # Let's do a few restarts if time permits.
    
    best_score = -1
    best_set = None
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        current_set, current_mask, current_score = random_set()
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        # Estimate steps per restart
        # Each step involves set update, mask update, score calculation.
        # Score calculation is ~256 bitwise ops. Very fast.
        # Let's run until time limit for this restart.
        
        restart_deadline = start_time + time_per_restart * (r + 1)
        
        while time.time() < restart_deadline:
            # Generate move
            # Pick x in current_set, y in [0, N) \ current_set
            x = random.choice(list(current_set))
            # Pick y not in current_set
            # Efficient way: pick random y, if in set, reject or resample.
            # With K=32, N=512, prob of collision is 32/512 = 1/16. Resampling is fine.
            while True:
                y = random.randint(0, N - 1)
                if y not in current_set:
                    break
            
            # Apply move
            new_set = current_set.copy()
            new_set.remove(x)
            new_set.add(y)
            
            # Update mask
            new_mask = current_mask
            new_mask &= ~(1 << x)
            new_mask |= (1 << y)
            
            new_score = score_bitmask(new_mask)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_set = new_set
                current_mask = new_mask
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            
            T *= cooling
            if T < T_min:
                T = T_min

    # Convert best_set (0-based) to 1-based list
    return sorted(list(best_set) + [1] * 0) # Wait, just add 1 to each
    
    # Correction:
    res = sorted([x + 1 for x in best_set])
    return res
