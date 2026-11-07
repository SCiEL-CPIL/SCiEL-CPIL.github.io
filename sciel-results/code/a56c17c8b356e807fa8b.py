import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 85  # 5s safety margin
    
    # Precompute: for each mask, which other masks are it a subset of (and vice versa)
    # But that's too much memory. Instead, use fast bit operations.
    
    def is_comparable(a, b):
        """Check if a and b are comparable (one is subset of other, strict)."""
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0
    
    def count_comparables(family_masks):
        """Count comparable pairs in a list of masks."""
        cnt = 0
        L = len(family_masks)
        for i in range(L):
            ai = family_masks[i]
            for j in range(i + 1, L):
                bj = family_masks[j]
                if (ai & ~bj) == 0 or (bj & ~ai) == 0:
                    cnt += 1
        return cnt
    
    def initial_family():
        """Create a good initial family using chains."""
        # Take a long chain: empty, {0}, {0,1}, {0,1,2}, ... 
        # Plus some additional sets to fill up to 40
        masks = []
        # Chain 1: add elements one by one
        chain1 = 0
        for i in range(n):
            chain1 |= (1 << i)
            masks.append(chain1)
        # That's 12 sets (including empty at start? No, we start from 0 and add)
        # Actually chain1 starts at 0, then 1, 3, 7, ... so 13 sets (0 through full)
        # Wait, we append after OR, so we get: 1, 3, 7, 15, ... (12 sets, missing empty)
        # Let me redo: include empty set
        masks = []
        cur = 0
        masks.append(0)  # empty set
        for i in range(n):
            cur |= (1 << i)
            masks.append(cur)
        # Now we have 13 sets: empty, {0}, {0,1}, ..., {0..11}
        # Need 40 - 13 = 27 more
        # Add sets that are comparable with existing ones
        # Try adding singletons, pairs, etc.
        used = set(masks)
        # Add all singletons
        for i in range(n):
            s = 1 << i
            if s not in used:
                masks.append(s)
                used.add(s)
        # Add all pairs
        for i in range(n):
            for j in range(i + 1, n):
                s = (1 << i) | (1 << j)
                if s not in used:
                    masks.append(s)
                    used.add(s)
                if len(masks) >= m:
                    break
            if len(masks) >= m:
                break
        # Trim to m if needed
        return masks[:m]
    
    # Generate all possible masks
    all_masks = list(range(total_masks))
    
    # Start with initial family
    best_family = initial_family()
    best_score = count_comparables(best_family)
    
    # Local search: hill climbing with random restarts
    rng = random.Random(42)
    
    def try_improve(family, score):
        """Try to improve family by swapping one element."""
        L = len(family)
        fam_set = set(family)
        
        # Try each position
        for pos in range(L):
            if time.time() > deadline - 1:
                break
            old_mask = family[pos]
            
            # Try replacing with a random other mask
            # But we should be smart: try masks that are comparable with many existing
            # For speed, just try random replacements
            for _ in range(20):  # try 20 random replacements per position
                if time.time() > deadline - 1:
                    break
                new_mask = rng.choice(all_masks)
                if new_mask in fam_set:
                    continue
                
                # Compute new score efficiently
                # Remove old_mask, add new_mask
                # Score change = (comparables of new_mask with others) - (comparables of old_mask with others)
                old_comp = 0
                new_comp = 0
                for i in range(L):
                    if i == pos:
                        continue
                    other = family[i]
                    if (old_mask & ~other) == 0 or (other & ~old_mask) == 0:
                        old_comp += 1
                    if (new_mask & ~other) == 0 or (other & ~new_mask) == 0:
                        new_comp += 1
                
                delta = new_comp - old_comp
                if delta > 0:
                    # Accept
                    family[pos] = new_mask
                    fam_set.remove(old_mask)
                    fam_set.add(new_mask)
                    score += delta
                    break  # restart from beginning since family changed
        
        return family, score
    
    # Hill climbing with multiple restarts
    for restart in range(50):
        if time.time() > deadline - 2:
            break
        
        if restart == 0:
            family = best_family[:]
            score = best_score
        else:
            # Perturb best family
            family = best_family[:]
            # Randomly swap 5 elements
            for _ in range(5):
                pos = rng.randrange(len(family))
                new_mask = rng.choice(all_masks)
                if new_mask not in set(family):
                    family[pos] = new_mask
            score = count_comparables(family)
        
        # Hill climb
        family, score = try_improve(family, score)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
