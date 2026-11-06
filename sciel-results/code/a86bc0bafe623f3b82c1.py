import time
import random

def construct():
    """
    Constructs a set of 32 distinct integers in 1..512 maximizing the number
    of distinct differences d > 0 such that a, a+d, a+2d are in the set.
    Uses Greedy Local Search with Random Restarts and Bitmask-based scoring.
    """
    
    def score_mask(mask):
        """
        mask: integer representing the set. Bit i is 1 if (i+1) is in the set.
        Returns the number of distinct differences d > 0 forming a 3-AP.
        """
        count = 0
        # d can range from 1 to 255 (since 1 + 2*255 = 511 <= 512)
        # If max element is 512, 1 + 2*d <= 512 => 2*d <= 511 => d <= 255.5
        for d in range(1, 256):
            # Check if there exists a such that a, a+d, a+2d are in set
            # mask has bit i set if i+1 is in set.
            # We want to check if (mask & (mask >> d) & (mask >> 2*d)) != 0
            # Note: shifting right by d aligns bit i with bit i-d.
            # If mask has bit i, mask >> d has bit i-d.
            # Intersection at bit j means:
            # mask has j, mask has j+d, mask has j+2d.
            # So j, j+d, j+2d are in the set.
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_set(mask):
        return [i + 1 for i in range(512) if (mask >> i) & 1]

    def set_to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m

    random.seed(42)
    start_time = time.time()
    time_limit = 110.0
    
    best_score = 0
    best_mask = 0
    
    # Number of restarts/perturbations
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize or Perturb
        if restart == 0:
            # Initial random set
            current_set = set(random.sample(range(1, 513), 32))
        else:
            # Perturb the best found so far
            current_set = set(mask_to_set(best_mask))
            # Randomly swap 2 to 5 elements
            num_swaps = random.randint(2, 5)
            for _ in range(num_swaps):
                if len(current_set) < 2:
                    break
                to_remove = random.choice(list(current_set))
                to_add = random.randint(1, 512)
                while to_add in current_set:
                    to_add = random.randint(1, 512)
                current_set.remove(to_remove)
                current_set.add(to_add)
        
        current_mask = set_to_mask(current_set)
        current_score = score_mask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        # Local Search
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                break
                
            # Try all single swaps
            # To speed up, we can iterate over elements in current_set
            # and random candidates from the complement
            # But to ensure we find the best local move, we might need to check more.
            # Let's try a heuristic: check all removals and a subset of additions?
            # Or just all swaps. 32 * 480 = 15360 swaps.
            # Each swap: 1 score calculation (~255 bitwise ops).
            # 15360 * 255 = 3.9M ops. ~0.2-0.4s per pass.
            
            best_delta = 0
            best_swap = None
            
            # Optimization: Precompute list of elements in set and not in set
            in_set = list(current_set)
            not_in_set = [i for i in range(1, 513) if i not in current_set]
            
            for x in in_set:
                # We can break early if we find a large improvement? 
                # Greedy usually takes the first improvement or best improvement.
                # Taking best improvement is better for convergence.
                for y in not_in_set:
                    # Calculate new mask
                    # Remove x: clear bit x-1
                    # Add y: set bit y-1
                    new_mask = current_mask & ~(1 << (x - 1)) | (1 << (y - 1))
                    new_score = score_mask(new_mask)
                    delta = new_score - current_score
                    if delta > best_delta:
                        best_delta = delta
                        best_swap = (x, y)
                        
            if best_swap:
                x, y = best_swap
                current_set.remove(x)
                current_set.add(y)
                current_mask = set_to_mask(current_set)
                current_score += best_delta
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
    
    return sorted(mask_to_set(best_mask))
