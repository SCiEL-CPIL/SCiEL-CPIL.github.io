import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit
    
    N = 51975
    
    # 1. Generate divisors
    divisors = []
    for d in range(3, N + 1, 2):
        if N % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # 2. Precompute masks for all (m, r)
    # masks[m_index][r] = integer mask
    # We map modulus to an index for faster access
    moduli = divisors
    M = len(moduli)
    
    # Precompute masks
    # To save memory and time, we only compute masks for moduli we might use.
    # Since we can use all 47, we compute for all.
    # Total masks ~ 12000. Each is an int of ~6500 bits. This is fine.
    
    masks = {} # (m, r) -> mask
    # Actually, storing in a list of lists is faster.
    # mask_list[m_idx][r]
    mask_list = []
    for m in moduli:
        m_masks = [0] * m
        # For each residue r, the mask has bits at r, r+m, r+2m...
        # We can construct this efficiently.
        # Alternatively, just loop x from 0 to N-1? No, that's O(N*M).
        # Better: for r in range(m):
        #   mask = 0
        #   for x in range(r, N, m):
        #       mask |= (1 << x)
        # This is O(N) per modulus, total O(N*M) ~ 51975 * 47 ~ 2.4M operations. Very fast.
        
        # Even faster: 
        # base_mask for r=0 is sum(1 << (k*m) for k in range(N//m))
        # mask_r = base_mask << r ? No, bits are at r, r+m...
        # Let's just use the loop, it's simple and fast enough in Python if optimized.
        # Actually, creating big integers by shifting is fast.
        
        # Optimization: 
        # We can generate the mask for r=0, then shift?
        # No, because the bits wrap around? No, modulo N is not involved in the mask, 
        # the mask represents the set {x in [0, N) | x % m == r}.
        # So bits are exactly at indices r, r+m, r+2m...
        
        # Let's precompute the base pattern for each m.
        # pattern_m has 1s at positions 0, m, 2m...
        # Then mask_r is pattern_m shifted by r? 
        # If pattern_m has bit i set if i % m == 0.
        # Then (pattern_m << r) has bit i+r set if (i+r) % m == 0 => i % m == -r.
        # This doesn't directly give us residue r.
        
        # Let's stick to the simple generation for safety, it's only 2.4M bit sets.
        # But 2.4M Python loops might take ~1s. Acceptable.
        
        for r in range(m):
            mask = 0
            # Unroll or use sum? 
            # mask = sum(1 << x for x in range(r, N, m))
            # This creates many big ints. 
            # Better to build incrementally.
            # Or use a trick:
            # The mask is periodic with period m? No.
            
            # Let's try a different way to build masks.
            # We can use numpy to create the array of indices, then convert to int?
            # No, standard library only.
            
            # Let's just loop.
            for x in range(r, N, m):
                mask |= (1 << x)
            m_masks[r] = mask
        mask_list.append(m_masks)
        
    # 3. Greedy Initialization
    current_classes = [] # List of [m_idx, r]
    current_union = 0
    used_moduli = set()
    
    # Greedy: repeatedly pick the class that adds the most new bits
    # We need to evaluate gain for all unused moduli and all their residues.
    # This is expensive if done naively (47 moduli * avg 250 residues * bit_count).
    # 47 * 250 * ~100ns = ~1ms per iteration. 47 iterations = 50ms. Very fast.
    
    available_moduli = list(range(M))
    
    while available_moduli:
        best_gain = 0
        best_m_idx = None
        best_r = None
        
        for m_idx in available_moduli:
            m = moduli[m_idx]
            m_masks = mask_list[m_idx]
            # Find r that maximizes popcount(current_union | m_masks[r])
            # Which is equivalent to maximizing popcount((~current_union) & m_masks[r])
            # Since popcount(A|B) = popcount(A) + popcount(B & ~A)
            # So we want to maximize popcount(B & ~current_union)
            not_current = (~current_union) & ((1 << N) - 1)
            
            # Iterate residues
            max_gain = 0
            max_r = 0
            for r in range(m):
                gain = (m_masks[r] & not_current).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    max_r = r
            
            if max_gain > best_gain:
                best_gain = max_gain
                best_m_idx = m_idx
                best_r = max_r
                
        if best_gain == 0:
            break
            
        # Add the best class
        current_classes.append([best_m_idx, best_r])
        current_union |= mask_list[best_m_idx][best_r]
        used_moduli.add(best_m_idx)
        available_moduli.remove(best_m_idx)
        
        if time.time() > deadline - 10:
            break
            
    best_score = N - current_union.bit_count()
    best_classes = [c[:] for c in current_classes]
    
    # 4. Local Search / Iterated Local Search
    # We will refine the solution.
    
    def get_union(classes):
        u = 0
        for m_idx, r in classes:
            u |= mask_list[m_idx][r]
        return u
    
    def local_search(classes, union):
        # Coordinate descent on residues
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(classes)):
                m_idx, r = classes[i]
                m = moduli[m_idx]
                # Calculate union of all other classes
                others_union = union
                # Remove current class contribution? 
                # We can't just XOR because of overlaps.
                # We must recompute union of others.
                # This is O(K) where K is number of classes.
                # Since K <= 47, this is fast.
                
                others = 0
                for j in range(len(classes)):
                    if j != i:
                        mj, rj = classes[j]
                        others |= mask_list[mj][rj]
                        
                # Now find best r for m_idx given 'others'
                not_others = (~others) & ((1 << N) - 1)
                m_masks = mask_list[m_idx]
                
                best_r = r
                best_gain = (m_masks[r] & not_others).bit_count()
                
                for new_r in range(m):
                    if new_r == r: continue
                    gain = (m_masks[new_r] & not_others).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_r = new_r
                        
                if best_r != r:
                    # Update class
                    # Remove old mask from union? 
                    # We will recompute union at the end of the loop or update incrementally?
                    # Incremental is tricky with overlaps. 
                    # Let's just update the class and recompute union for next iteration?
                    # No, we can update union:
                    # old_union = union
                    # new_union = others | m_masks[best_r]
                    # This is correct because 'others' is the union of all except i.
                    classes[i][1] = best_r
                    union = others | m_masks[best_r]
                    improved = True
                    # Check if score improved
                    # Score is N - union.bit_count()
                    # We don't need to track score explicitly here, just the union.
                    
        # Try adding unused moduli
        used = set(c[0] for c in classes)
        for m_idx in range(M):
            if m_idx in used: continue
            if time.time() > deadline - 5: break
            m = moduli[m_idx]
            m_masks = mask_list[m_idx]
            not_current = (~union) & ((1 << N) - 1)
            best_gain = 0
            best_r = 0
            for r in range(m):
                gain = (m_masks[r] & not_current).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                classes.append([m_idx, best_r])
                union |= m_masks[best_r]
                used.add(m_idx)
                
        # Try removing redundant moduli
        # Check if removing a class decreases coverage
        # We iterate from largest index to smallest to handle removals safely
        for i in range(len(classes) - 1, -1, -1):
            if time.time() > deadline - 5: break
            m_idx, r = classes[i]
            # Compute union without class i
            others = 0
            for j in range(len(classes)):
                if j != i:
                    mj, rj = classes[j]
                    others |= mask_list[mj][rj]
            
            if others.bit_count() == union.bit_count():
                # No loss in coverage
                classes.pop(i)
                union = others
                
        return classes, union

    # Run local search on the initial solution
    current_classes, current_union = local_search(current_classes, current_union)
    current_score = N - current_union.bit_count()
    if current_score < best_score:
        best_score = current_score
        best_classes = [c[:] for c in current_classes]
        
    # Iterated Local Search: Perturb and search again
    # We will try several random restarts/perturbations if time allows.
    
    while time.time() < deadline - 10:
        # Perturb best solution
        # Pick a random class and change its residue
        if not best_classes:
            break
            
        perturbed_classes = [c[:] for c in best_classes]
        
        # Strategy 1: Change one residue
        if perturbed_classes:
            idx = random.randint(0, len(perturbed_classes) - 1)
            m_idx = perturbed_classes[idx][0]
            m = moduli[m_idx]
            new_r = random.randint(0, m - 1)
            perturbed_classes[idx][1] = new_r
            
        # Strategy 2: Swap a class with a random unused one (if any)
        # For simplicity, let's just do residue changes and maybe random class addition/removal
        # Let's add a random unused modulus
        used = set(c[0] for c in perturbed_classes)
        unused = [i for i in range(M) if i not in used]
        if unused and random.random() < 0.5:
            new_m_idx = random.choice(unused)
            m = moduli[new_m_idx]
            new_r = random.randint(0, m - 1)
            perturbed_classes.append([new_m_idx, new_r])
            
        # Run local search on perturbed
        p_union = 0
        for c in perturbed_classes:
            p_union |= mask_list[c[0]][c[1]]
            
        p_classes, p_union = local_search(perturbed_classes, p_union)
        p_score = N - p_union.bit_count()
        
        if p_score < best_score:
            best_score = p_score
            best_classes = [c[:] for c in p_classes]
            
    # Final return
    # Convert indices back to moduli
    result = []
    for m_idx, r in best_classes:
        result.append([moduli[m_idx], r])
        
    return result

