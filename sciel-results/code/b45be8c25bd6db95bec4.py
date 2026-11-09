import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    max_mask = 1 << n
    
    # 1. Initialization
    # Build 3 chains of 13 and 1 extra
    family = []
    used = set()
    
    # Helper to build a chain
    def build_chain(start_mask, length, used):
        chain = []
        current = start_mask
        # Greedily add bits to grow the chain
        # We want to reach size 12 if possible, or just grow as much as possible
        # Actually, just pick a path in the hypercube
        # Simple path: add bits in a fixed order, skipping used
        # Better: add bit that keeps option open?
        # For now, just add random bits
        bits = list(range(n))
        random.shuffle(bits)
        
        # Ensure start is valid
        if current in used:
             # find a new start
             while current in used:
                 current = random.randint(0, max_mask-1)
        
        chain.append(current)
        used.add(current)
        
        for _ in range(length - 1):
            # Try to add a bit
            moved = False
            # Shuffle bits each step to allow different chains
            random.shuffle(bits)
            for b in bits:
                if not (current & (1 << b)):
                    new_mask = current | (1 << b)
                    if new_mask not in used:
                        current = new_mask
                        chain.append(current)
                        used.add(current)
                        moved = True
                        break
            if not moved:
                break
                
        return chain

    # Create 3 chains
    starts = [0, 1, 2] # 0, {0}, {1}
    for s in starts:
        c = build_chain(s, 13, used)
        family.extend(c)
        
    # Fill remaining with random
    while len(family) < m:
        s = random.randint(0, max_mask-1)
        if s not in used:
            family.append(s)
            used.add(s)
            
    family = np.array(family, dtype=np.int32)
    
    # 2. Local Search
    # Precompute all masks
    all_masks = np.arange(max_mask, dtype=np.int32)
    
    current_score = _calc_score(family)
    best_family = family.copy()
    best_score = current_score
    
    in_set = np.zeros(max_mask, dtype=bool)
    for s in family:
        in_set[s] = True
        
    deadline = time.time() + 55.0
    
    no_improve = 0
    
    while time.time() < deadline:
        best_delta = 0
        best_idx = -1
        best_new = -1
        
        # Try each index
        for i in range(m):
            if time.time() > deadline: break
            old_s = family[i]
            
            # Compute old comps
            # We can maintain a score array? No, just recalculate for the 39 others
            # Actually, we can compute the gain for ALL candidates at once for this i
            
            others = np.delete(family, i)
            
            # Vectorized gain calculation
            # cands: (4096, 1)
            # oth: (1, 39)
            c = all_masks[:, np.newaxis]
            o = others[np.newaxis, :]
            
            # Subset checks
            # (c & o) == c  => c subset o
            # (o & c) == o  => o subset c
            c_sub_o = (c & o) == c
            o_sub_c = (o & c) == o
            
            # Exclude equality
            eq = c == o
            strict_c_sub_o = c_sub_o & ~eq
            strict_o_sub_c = o_sub_c & ~eq
            
            new_comps = strict_c_sub_o.sum(axis=1) + strict_o_sub_c.sum(axis=1)
            
            # Old comps for old_s
            # old_s is at index i in family
            # We need count of j != i such that family[j] comparable with old_s
            # This is just the sum of comparabilities of old_s with others
            # We can compute this quickly
            old_comps = 0
            for j in range(m):
                if j == i: continue
                sj = family[j]
                if (old_s & sj) == old_s and old_s != sj:
                    old_comps += 1
                elif (sj & old_s) == sj and sj != old_s:
                    old_comps += 1
            
            deltas = new_comps - old_comps
            
            # Mask out used sets
            deltas[in_set] = -999999
            
            max_idx = np.argmax(deltas)
            max_delta = deltas[max_idx]
            
            if max_delta > best_delta:
                best_delta = max_delta
                best_idx = i
                best_new = all_masks[max_idx]
                
        if best_delta > 0:
            # Apply
            old_s = family[best_idx]
            in_set[old_s] = False
            family[best_idx] = best_new
            in_set[best_new] = True
            current_score += best_delta
            no_improve = 0
            
            if current_score > best_score:
                best_score = current_score
                best_family = family.copy()
        else:
            no_improve += 1
            if no_improve > 10:
                # Restart or Perturb
                # Simple perturb: swap 2 random
                i1, i2 = random.sample(range(m), 2)
                s1, s2 = family[i1], family[i2]
                # Find replacements
                # Just random for now
                new1 = random.randint(0, max_mask-1)
                new2 = random.randint(0, max_mask-1)
                if new1 != new2 and new1 not in in_set and new2 not in in_set:
                     # Check if they are distinct from each other and not in set
                     # This is a weak perturb
                     pass
                no_improve = 0
                 
    # Convert to lists
    result = []
    for s in best_family:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    return result

def _calc_score(family):
    m = len(family)
    count = 0
    for i in range(m):
        si = family[i]
        for j in range(i+1, m):
            sj = family[j]
            if (si & sj) == si and si != sj:
                count += 1
            elif (si & sj) == sj and si != sj:
                count += 1
    return count
