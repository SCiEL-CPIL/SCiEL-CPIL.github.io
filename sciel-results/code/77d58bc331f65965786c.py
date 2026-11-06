"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0
    best_set = []
    
    # Strategy: Use a geometric skeleton (powers of 4) to guarantee unique triple sums,
    # then attempt to insert smaller integers into the gaps.
    
    # Try multiple random seeds for the skeleton base
    for seed in range(100):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Start with powers of 4 (or a variant) as the skeleton
        # 4^0=1, 4^1=4, 4^2=16, 4^3=64, 4^4=256, 4^5=1024, 4^6=4096
        # But we need distinct positive integers <= 4096
        # Let's use a slightly different approach: start with a known-valid sparse set
        
        # Actually, let me think about this differently.
        # The key insight is that if we have elements a1 < a2 < ... < ak,
        # and the gaps are large enough, triple sums will be unique.
        
        # A simple valid set: powers of 3 give unique base-3 representations for sums of 3 elements.
        # But 3^8 = 6561 > 4096, so we can use at most 3^0 to 3^7 = 1,3,9,27,81,243,729,2187 (8 elements)
        
        # Let's try: start with a geometric progression and then greedily add more elements.
        
        selected = []
        existing_sums = set()
        
        # Option 1: Start with powers of 3
        # Option 2: Start with a random sparse set
        # Let's try both approaches
        
        if seed % 2 == 0:
            # Powers of 3
            skeleton = [1, 3, 9, 27, 81, 243, 729, 2187]
        else:
            # Random sparse skeleton
            skeleton = []
            current = 1
            while current <= 4096 and len(skeleton) < 12:
                skeleton.append(current)
                # Next element should be at least 3*current to ensure separation
                # Actually, for B_3[1], we need more care
                # Let's use a factor of 4
                current = current * 4 + random.randint(1, 2)
            # Ensure all <= 4096
            skeleton = [x for x in skeleton if x <= 4096]
        
        # Add skeleton elements in order
        for x in sorted(skeleton):
            if x > 4096:
                continue
            if x in selected:
                continue
            if _check_and_add(selected, existing_sums, x):
                pass  # Already added
        
        # Now try to fill in gaps with smaller numbers
        # Try all numbers from 1 to 4096 in random order
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in selected:
                continue
            if _check_and_add(selected, existing_sums, x):
                pass
        
        # Update best if better
        if len(selected) > len(best_set):
            best_set = list(selected)
        
        # Early stop if we've reached a good size
        if len(best_set) >= 20:
            break
    
    return best_set

def _check_and_add(selected, existing_sums, x):
    """Check if x can be added to maintain B_3[1] property. If yes, add it and return True."""
    # Triples involving x:
    # 1. (x, x, x) -> 3x
    # 2. (y, x, x) for y in selected -> y + 2x
    # 3. (y, y, x) for y in selected -> 2y + x
    # 4. (y, z, x) for y < z in selected -> y + z + x
    
    new_sums = set()
    valid = True
    
    # 1. (x, x, x)
    s = 3 * x
    if s in existing_sums:
        return False
    new_sums.add(s)
    
    # 2 and 3: Iterate through selected
    for y in selected:
        # (y, x, x) -> y + 2x
        s1 = y + 2 * x
        if s1 in existing_sums or s1 in new_sums:
            return False
        new_sums.add(s1)
        
        # (y, y, x) -> 2y + x
        s2 = 2 * y + x
        if s2 in existing_sums or s2 in new_sums:
            return False
        new_sums.add(s2)
    
    # 4. (y, z, x) with y < z in selected
    n = len(selected)
    for i in range(n):
        y = selected[i]
        for j in range(i + 1, n):
            z = selected[j]
            s = y + z + x
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)
    
    # All checks passed, add x
    selected.append(x)
    existing_sums.update(new_sums)
    return True
# EVOLVE-BLOCK-END
