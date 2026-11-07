import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n_universe = 6
    m = 40
    total_masks = 1 << n_universe
    
    def get_score(family):
        score = 0
        for i in range(len(family)):
            a = family[i]
            for j in range(i + 1, len(family)):
                b = family[j]
                if (a & b) == a or (b & a) == a:
                    score += 1
        return score

    def get_contrib(mask, family, exclude_mask=None):
        """
        Calculate the number of comparable pairs involving 'mask' with members of 'family'.
        If exclude_mask is provided, it is ignored in the summation.
        """
        count = 0
        for y in family:
            if y == exclude_mask:
                continue
            if (mask & y) == mask or (y & mask) == mask:
                count += 1
        return count

    def local_search_best_ascent(initial_family, time_limit):
        family = list(initial_family)
        score = get_score(family)
        
        # Precompute list of all masks not in family for faster access?
        # We can just generate on the fly or maintain a set.
        in_family = set(family)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_a = None
            best_b = None
            
            candidates_b = [m for m in range(total_masks) if m not in in_family]
            
            for a in family:
                # Early exit if time is critical
                if time.time() > time_limit:
                    break
                
                # Calculate contribution of A
                contrib_a = get_contrib(a, family, a)
                
                for b in candidates_b:
                    # Calculate contribution of B (excluding A, since A is being removed)
                    contrib_b = get_contrib(b, family, a)
                    
                    delta = contrib_b - contrib_a
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_a is not None and best_delta > 0:
                # Apply swap
                family.remove(best_a)
                family.append(best_b)
                in_family.remove(best_a)
                in_family.add(best_b)
                score += best_delta
                improved = True
        
        return family, score

    best_family = []
    best_score = -1
    
    # Strategy 1: Analytical Start (Down-set modification)
    # Universe 0..5
    # All sets size <= 3 (42 sets)
    base_family = [mask for mask in range(total_masks) if bin(mask).count('1') <= 3]
    # Remove 1 size 2 and 1 size 3 (comparable) to get 40
    # Size 2: {0,1} -> mask 0b000011 = 3
    # Size 3: {0,1,2} -> mask 0b000111 = 7
    # 3 is subset of 7.
    init1 = [m for m in base_family if m != 3 and m != 7]
    
    # Strategy 2: Random Starts
    # Generate a few random valid families
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts == 1:
            current_family = init1
        else:
            # Random restart: pick 40 random masks
            current_family = random.sample(range(total_masks), m)
            
        # Run local search
        # Allocate time based on remaining time
        # Simple: run until no improvement or 2s per attempt?
        # Better: run until no improvement, but check global deadline.
        # Since local search converges fast for N=40, we can just let it run.
        
        # To prevent one search from taking too long, we can limit iterations or time per search.
        # But Best-Ascent on 40 items is fast.
        
        family, score = local_search_best_ascent(current_family, deadline)
        
        if score > best_score:
            best_score = score
            best_family = family
            
        # If we found a very high score, we might stop early?
        # Max possible is 780.
        # If best_score > 200, it's likely optimal or near-optimal.
        # For now, keep searching until deadline.

    # Convert masks to lists of ints
    result = []
    for mask in best_family:
        subset = []
        for i in range(n_universe):
            if (mask >> i) & 1:
                subset.append(i)
        result.append(subset)
        
    # Ensure exactly 40 distinct lists
    # The local search ensures distinct masks, so distinct lists.
    
    return result
