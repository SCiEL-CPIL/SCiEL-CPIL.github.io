"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
# EVOLVE-BLOCK-START

def _count_3aps_bitmask(mask):
    """Count distinct 3-term APs in the set represented by bitmask."""
    count = 0
    # Iterate over all pairs (a, b) with a < b
    # For each pair, check if c = 2b - a is in the set and c > b
    # Using bit operations is tricky for arbitrary c, so we iterate elements.
    # To speed up, we can iterate d (difference) and check if a, a+d, a+2d are in set.
    # But simple iteration over elements is O(N^2) which is ~65k ops for N=256, very fast.
    elements = [i+1 for i in range(256) if (mask >> i) & 1]
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i+1, n):
            b = elements[j]
            c = 2*b - a
            if c > 256:
                break
            if (mask >> (c-1)) & 1:
                count += 1
    return count

def _has_4ap_fast(mask):
    """Check if mask contains any 4-term AP. Returns True if it does."""
    # A 4-AP is a, a+d, a+2d, a+3d.
    # We can check for each d, if there exists a such that all 4 bits are set.
    # This is equivalent to checking if (mask & (mask >> d) & (mask >> 2d) & (mask >> 3d)) != 0
    # But we must ensure a, a+d, a+2d, a+3d are within [1, 256].
    # The bit shift handles the "within bounds" implicitly if we mask out higher bits, 
    # but we need to be careful with the start 'a'.
    # If bit k is set in the intersection, it means a=k+1 is the start of a 4-AP with diff d.
    # We need a+3d <= 256. So k+1+3d <= 256 => k <= 255-3d.
    # So we only care about bits 0 to 255-3d.
    
    for d in range(1, 86): # max d such that 1+3d <= 256 => 3d <= 255 => d <= 85
        # Shifts: d, 2d, 3d
        # Intersection of mask and shifted masks
        # We want to know if there is a bit set in mask AND mask>>d AND mask>>2d AND mask>>3d
        # Let M = mask
        # We check if (M & (M >> d) & (M >> 2*d) & (M >> 3*d)) has any bits set in the valid range.
        # Valid range for start 'a' (0-indexed bit pos k): k+3d < 256 => k < 256-3d
        # So we mask the result with (1 << (256 - 3*d)) - 1
        
        inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
        valid_mask = (1 << (256 - 3*d)) - 1
        if inter & valid_mask:
            return True
    return False

def _construct_greedy():
    """Greedy construction of a set with no 4-APs."""
    mask = 0
    # We iterate through numbers 1 to 256.
    # To maximize 3-APs, we might want a denser set.
    # Standard greedy: add x if it doesn't create a 4-AP.
    # To bias towards more 3-APs, we can try adding x if it creates many 3-APs, 
    # but greedy addition order matters. 
    # Let's try a few different orders or just standard increasing order first.
    
    # A known good strategy for no 4-AP is to take numbers with only 0s and 1s in base 3? No, that's for no 3-AP.
    # For no 4-AP, we can take numbers that don't have certain patterns.
    # Let's stick to greedy: add x if safe.
    
    # To get a better starting point, we can run multiple greedy passes with different random perturbations or orders.
    # But let's start with a deterministic greedy that prioritizes adding elements that form 3-APs with existing ones.
    
    # Actually, simple increasing greedy often yields ~100-120 elements.
    # Let's try to be smarter: 
    # We want to maximize 3-APs. 
    # Let's just do a standard greedy first, then local search.
    
    for x in range(1, 257):
        # Check if adding x creates a 4-AP
        new_mask = mask | (1 << (x-1))
        if not _has_4ap_fast(new_mask):
            mask = new_mask
    return mask

def _local_search(mask, time_limit):
    """Improve the set by swapping elements or adding/removing."""
    start_time = time.time()
    best_mask = mask
    best_score = _count_3aps_bitmask(mask)
    
    # Precompute list of elements in set
    def get_elements(m):
        return [i+1 for i in range(256) if (m >> i) & 1]
    
    current_mask = mask
    current_score = best_score
    elements = get_elements(current_mask)
    
    # We will try random swaps: remove one element, add another.
    # Or just try adding a random element not in set (if valid).
    # Or removing an element that is in few 3-APs.
    
    # Strategy: 
    # 1. Try to add a random element not in the set. If valid, accept if it increases score.
    # 2. Try to swap a random element in the set with a random element not in the set. If valid and score increases, accept.
    
    # To speed up, we can cache the score change.
    # Delta score for adding x:
    # New 3-APs formed: (a, b, x) where x=2b-a, or (a, x, b) where x=(a+b)/2, or (x, a, b) where a=x+d, b=x+2d? No, x is part of AP.
    # If we add x:
    # - x as end: a, b, x => 2b-x=a. a in set, b in set.
    # - x as mid: a, x, b => x=(a+b)/2. a in set, b in set.
    # - x as start: x, b, c => 2b-x=c. b in set, c in set.
    
    # This is complex to calculate delta quickly without iterating.
    # Given N=256, full recount is O(N^2) ~ 65k ops. 
    # 60 seconds / 65k ops = ~9000 full recounts per second.
    # We can do ~500k-1M local search steps.
    
    while time.time() < start_time + time_limit:
        # Random move
        if random.random() < 0.5:
            # Try to add
            candidates = [x for x in range(1, 257) if not (current_mask >> (x-1)) & 1]
            if not candidates: continue
            x = random.choice(candidates)
            new_mask = current_mask | (1 << (x-1))
        else:
            # Try to swap
            if not elements: continue
            x_remove = random.choice(elements)
            candidates = [x for x in range(1, 257) if x != x_remove and not (current_mask >> (x-1)) & 1]
            if not candidates: continue
            x_add = random.choice(candidates)
            new_mask = (current_mask & ~(1 << (x_remove-1))) | (1 << (x_add-1))
        
        # Check validity
        if _has_4ap_fast(new_mask):
            continue
            
        # Check score
        new_score = _count_3aps_bitmask(new_mask)
        if new_score > current_score:
            current_mask = new_mask
            current_score = new_score
            elements = get_elements(current_mask)
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
            # Optional: if we find a big improvement, we could restart or just continue
        # We don't accept worse moves (greedy hill climbing) to keep it fast and simple.
        # If we want to escape local optima, we could allow small worsening, but let's stick to greedy for speed.
        
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 5 second safety margin
    
    # Initial construction
    # Try a few different greedy strategies to get a good start
    best_initial_mask = 0
    best_initial_score = 0
    
    # Strategy 1: Standard increasing greedy
    mask1 = _construct_greedy()
    score1 = _count_3aps_bitmask(mask1)
    
    # Strategy 2: Random order greedy (multiple times)
    for _ in range(5):
        if time.time() > deadline - 10: break
        random_order = list(range(1, 257))
        random.shuffle(random_order)
        mask = 0
        for x in random_order:
            new_mask = mask | (1 << (x-1))
            if not _has_4ap_fast(new_mask):
                mask = new_mask
        score = _count_3aps_bitmask(mask)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_mask = mask
    
    if score1 > best_initial_score:
        best_initial_score = score1
        best_initial_mask = mask1
        
    # Local search
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        best_mask = _local_search(best_initial_mask, remaining_time - 1)
    else:
        best_mask = best_initial_mask
        
    # Final check and convert to list
    if _has_4ap_fast(best_mask):
        # Fallback: repair by removing elements involved in 4-APs
        # This should rarely happen if local search is correct
        mask = best_mask
        while _has_4ap_fast(mask):
             # Find a 4-AP and remove one element
             # Simple approach: iterate d, find a, remove a
             found = False
             for d in range(1, 86):
                 inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
                 valid_mask = (1 << (256 - 3*d)) - 1
                 inter &= valid_mask
                 if inter:
                     # Find the lowest set bit
                     k = (inter & -inter).bit_length() - 1
                     a = k + 1
                     mask &= ~(1 << (a-1))
                     found = True
                     break
             if not found: break
        best_mask = mask

    result = [i+1 for i in range(256) if (best_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
