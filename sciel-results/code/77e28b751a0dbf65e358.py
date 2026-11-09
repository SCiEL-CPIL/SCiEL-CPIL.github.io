import time
import random
import sys
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    # Bitmask representation: bit i is set if i is in the set
    # We use 1..512, so we need bits 1..512.
    
    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        """
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        """Convert bitmask to list of integers."""
        res = []
        for i in range(1, 513):
            if (mask >> i) & 1:
                res.append(i)
        return res

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Greedy construction to get a good initial solution
    def greedy_construct():
        # Start with an empty set and greedily add elements that maximize new progressions
        current_set = set()
        current_mask = 0
        
        for _ in range(N):
            best_elem = None
            best_score = -1
            # Try all possible elements not in current set
            # To speed up, we can sample a subset or use heuristics
            # For now, let's try all 512 elements (512 * 256 * 32 ops per addition is too slow)
            # Better: sample random candidates and pick the best among them
            candidates = random.sample(range(1, 513), 50)
            candidates = [c for c in candidates if c not in current_set]
            
            for c in candidates:
                new_mask = current_mask | (1 << c)
                s = score_mask(new_mask)
                if s > best_score:
                    best_score = s
                    best_elem = c
            
            if best_elem is not None:
                current_set.add(best_elem)
                current_mask |= (1 << best_elem)
        
        return list(current_set)

    # Local search with bitmask
    def local_search(initial_list, max_time=5.0):
        current_list = initial_list[:]
        current_mask = list_to_mask(current_list)
        current_score = score_mask(current_mask)
        
        best_list = current_list[:]
        best_mask = current_mask
        best_score = current_score
        
        T = 10.0
        T_min = 0.5
        cooling = 0.999
        start_ls = time.time()
        
        while time.time() - start_ls < max_time:
            # Pick element to remove
            val_out = current_list[random.randint(0, N - 1)]
            
            # Pick element to add
            val_in = random.randint(1, 512)
            # Check if val_in is in current_set (using mask for speed)
            while (current_mask >> val_in) & 1:
                val_in = random.randint(1, 512)
            
            # Apply move
            new_mask = (current_mask & ~(1 << val_out)) | (1 << val_in)
            new_score = score_mask(new_mask)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept
                current_mask = new_mask
                current_score = new_score
                # Update list
                for i in range(N):
                    if current_list[i] == val_out:
                        current_list[i] = val_in
                        break
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_list = current_list[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_list, best_score

    # Multi-start local search
    best_overall_list = None
    best_overall_score = -1
    
    # First, try a few greedy constructions
    for i in range(5):
        if time.time() > deadline - 30:
            break
        greedy_list = greedy_construct()
        ls_list, ls_score = local_search(greedy_list, max_time=20.0)
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_list = ls_list
    
    # Then, do random restarts with local search
    while time.time() < deadline - 5:
        random_list = random.sample(range(1, 513), N)
        ls_list, ls_score = local_search(random_list, max_time=10.0)
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_list = ls_list
    
    # Final polish: one more local search on the best found
    if best_overall_list is not None:
        final_list, final_score = local_search(best_overall_list, max_time=10.0)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_list = final_list
    
    return best_overall_list
# EVOLVE-BLOCK-END
