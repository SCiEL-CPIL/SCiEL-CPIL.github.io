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
    for seed in range(200):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        selected = []
        existing_sums = set()
        pair_sums = set()  # Precompute pair sums for faster conflict checking
        
        # Choose skeleton strategy based on seed
        strategy = seed % 4
        
        if strategy == 0:
            # Powers of 3
            skeleton = [1, 3, 9, 27, 81, 243, 729, 2187]
        elif strategy == 1:
            # Powers of 4 with small perturbations
            skeleton = []
            current = 1
            while current <= 4096 and len(skeleton) < 10:
                skeleton.append(current)
                current = current * 4 + random.randint(1, 3)
            skeleton = [x for x in skeleton if x <= 4096]
        elif strategy == 2:
            # Start with 1, then geometric with factor ~3.5
            skeleton = [1]
            current = 1
            while current <= 4096 and len(skeleton) < 12:
                current = int(current * 3.5) + random.randint(1, 5)
                if current <= 4096:
                    skeleton.append(current)
        else:
            # Random sparse with minimum gap
            skeleton = [1]
            current = 1
            while current <= 4096 and len(skeleton) < 12:
                current = current + random.randint(15, 50)
                if current <= 4096:
                    skeleton.append(current)
        
        # Add skeleton elements in order, maintaining pair_sums
        for x in sorted(skeleton):
            if x > 4096 or x in selected:
                continue
            # Check conflicts
            valid = True
            # (x,x,x)
            if 3*x in existing_sums:
                valid = False
            else:
                for y in selected:
                    # (y,x,x) = y+2x
                    if y + 2*x in existing_sums:
                        valid = False
                        break
                    # (y,y,x) = 2y+x
                    if 2*y + x in existing_sums:
                        valid = False
                        break
                    # (y,z,x) for z in selected, z > y: check y+z+x
                if valid:
                    for i in range(len(selected)):
                        for j in range(i+1, len(selected)):
                            if selected[i] + selected[j] + x in existing_sums:
                                valid = False
                                break
                        if not valid:
                            break
            if valid:
                # Add x and update structures
                selected.append(x)
                existing_sums.add(3*x)
                for y in selected[:-1]:
                    existing_sums.add(y + 2*x)
                    existing_sums.add(2*y + x)
                    pair_sums.add(y + x)
                for i in range(len(selected)-1):
                    for j in range(i+1, len(selected)-1):
                        existing_sums.add(selected[i] + selected[j] + x)
        
        # Now try to fill in gaps
        # Use a smarter candidate ordering: try numbers that are far from existing ones first
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        # Sort by "distance" to existing set - prefer numbers that create larger gaps
        # Actually, just try in random order but with time check
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if x in selected:
                continue
            
            # Fast pre-check: is 3x already in existing_sums?
            if 3*x in existing_sums:
                continue
            
            # Check all conflicts
            valid = True
            for y in selected:
                if y + 2*x in existing_sums:
                    valid = False
                    break
                if 2*y + x in existing_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if selected[i] + selected[j] + x in existing_sums:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Add x
                selected.append(x)
                existing_sums.add(3*x)
                for y in selected[:-1]:
                    existing_sums.add(y + 2*x)
                    existing_sums.add(2*y + x)
                for i in range(len(selected)-1):
                    for j in range(i+1, len(selected)-1):
                        existing_sums.add(selected[i] + selected[j] + x)
        
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
    # Optimize: for each pair (y,z), check y+z+x
    # We can precompute all pair sums to make this faster
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
