import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def score_set(U):
        if not U or 0 not in U:
            return 0.0
        U_list = list(U)
        max_u = max(U_list)
        if max_u == 0:
            return 0.0
        
        # Calculate sumset and difference set
        # Using sets for uniqueness
        sumset = set()
        diffset = set()
        
        # Optimization: If U is large, this might be slow.
        # N ~ 100-500. N^2 ~ 10k - 250k.
        # 250k iterations in Python is ~0.01-0.05s.
        # We can do many iterations in 140s.
        
        for a in U_list:
            for b in U_list:
                sumset.add(a + b)
                diffset.add(a - b)
        
        size_sum = len(sumset)
        size_diff = len(diffset)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1:
             # Score will be <= 1. 
             # The task asks to maximize. Seed is 1.079.
             # We only care about improvements.
             pass
             
        log_ratio = math.log(ratio)
        log_diameter = math.log(2 * max_u + 1)
        
        if log_diameter == 0:
            return 0.0
            
        return 1.0 + log_ratio / log_diameter

    def local_search(U, max_iters=10000):
        U = set(U)
        current_score = score_set(U)
        best_U = U.copy()
        best_score = current_score
        
        # Heuristic for new elements to try
        # Range up to some bound
        max_val = max(U) if U else 10
        search_range = max_val * 3 + 100
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Random move
            move_type = random.random()
            new_U = U.copy()
            
            if move_type < 0.3:
                # Add element
                if len(new_U) < 500: # Cap size to keep N^2 manageable
                    new_val = random.randint(0, search_range)
                    new_U.add(new_val)
            elif move_type < 0.6:
                # Remove element (keep 0)
                if len(new_U) > 1:
                    candidates = [x for x in new_U if x != 0]
                    if candidates:
                        new_U.remove(random.choice(candidates))
            else:
                # Replace element
                if len(new_U) > 1:
                    candidates = [x for x in new_U if x != 0]
                    if candidates:
                        old_val = random.choice(candidates)
                        new_val = random.randint(0, search_range)
                        new_U.remove(old_val)
                        new_U.add(new_val)
            
            new_score = score_set(new_U)
            
            if new_score > best_score:
                best_score = new_score
                best_U = new_U.copy()
                U = new_U
                # Expand search range if we found a better solution with larger max
                if max(U) > max_val:
                    max_val = max(U)
                    search_range = max_val * 3 + 100
                    
            # Occasionally restart or shuffle if stuck? 
            # Simple hill climbing might get stuck.
            # Add some randomization to accept worse moves?
            # Given the discrete nature, simple hill climbing with random restarts is good.
            
        return best_U, best_score

    # Generate diverse initial seeds
    seeds = []
    
    # 1. Small random sets
    for _ in range(10):
        size = random.randint(3, 20)
        s = set([0])
        while len(s) < size:
            s.add(random.randint(1, 100))
        seeds.append(s)
        
    # 2. Geometric progressions
    for base in range(2, 5):
        s = set([0])
        val = 1
        for i in range(10):
            s.add(val)
            val *= base
        seeds.append(s)
        
    # 3. Arithmetic progressions with gaps
    for gap in range(2, 5):
        s = set([0])
        for i in range(1, 20):
            s.add(i * gap)
        seeds.append(s)
        
    # 4. The known good seed
    seeds.append({0, 1, 3})
    
    best_overall_U = {0, 1, 3}
    best_overall_score = score_set(best_overall_U)
    
    # Run local search on each seed
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Limit time per seed
        seed_deadline = min(deadline, time.time() + 20.0)
        # Adjust max_iters based on remaining time? 
        # Just use a fixed reasonable number, the loop checks time.
        U, score = local_search(seed, max_iters=20000)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_U = U
    
    # Final check and return
    # Ensure 0 is in the set
    best_overall_U.add(0)
    return list(best_overall_U)

# EVOLVE-BLOCK-END
