from fractions import Fraction
import random

def _check_valid_and_get_collision(S):
    """Check if all reciprocal subset sums are distinct. Returns collision if found."""
    sums = {Fraction(0, 1): None} # Map sum to subset (list of indices)
    # Actually, storing subsets is expensive. 
    # We just need to know if a collision exists.
    # For repair, we need to know WHICH elements caused it.
    # We can store the subset as a bitmask or tuple.
    
    # For efficiency, we'll just return True/False for the main check,
    # but for repair, we need the collision.
    # Let's use a set of sums. If collision, we don't know which elements.
    # So we need to track subsets.
    
    # Given the size (<= 22), bitmask is fine.
    n = len(S)
    sums = {Fraction(0, 1): 0}
    
    for i, x in enumerate(S):
        bit = 1 << i
        new_sums = {}
        for s, mask in sums.items():
            new_s = s + Fraction(1, x)
            if new_s in sums:
                # Collision between new_s (using i) and existing s' in sums
                # Existing subset is sums[new_s]
                # New subset is mask | bit
                return (mask | bit, sums[new_s])
            new_sums[new_s] = mask | bit
        sums.update(new_sums)
        
    return None

def construct():
    # Seed is size 9. We know 16 is possible.
    # Try to find 17, 18, etc. using Local Search.
    
    all_elements = list(range(1, 23))
    
    # Try sizes from 22 down to 16
    for target_size in range(22, 15, -1):
        # Try multiple random starts
        for attempt in range(50):
            S = random.sample(all_elements, target_size)
            # Local search
            for step in range(1000):
                collision = _check_valid_and_get_collision(S)
                if collision is None:
                    return S
                
                mask1, mask2 = collision
                # S is a list. mask1, mask2 are indices in S.
                # We need to remove one element from S that is in mask1 or mask2.
                # Let's pick a random index from mask1.
                indices_in_mask1 = [i for i in range(len(S)) if (mask1 >> i) & 1]
                if not indices_in_mask1:
                    break
                remove_idx = random.choice(indices_in_mask1)
                removed_elem = S[remove_idx]
                
                # Remove it
                S.pop(remove_idx)
                
                # Add a random new element not in S
                available = [x for x in all_elements if x not in S]
                if not available:
                    break
                new_elem = random.choice(available)
                S.append(new_elem)
                
            # If loop finishes without finding valid, try next random start
            
    # Fallback to known good set (Size 16 is likely found by greedy, but we don't have it hardcoded)
    # Let's do a quick greedy to find a good fallback
    current_set = []
    for x in range(22, 0, -1):
        new_set = current_set + [x]
        if _check_valid_and_get_collision(new_set) is None:
            current_set = new_set
            
    return current_set
