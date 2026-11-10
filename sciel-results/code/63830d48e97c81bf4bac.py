import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 51975
    
    # Generate divisors
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter odd divisors > 1
    divisors = [d for d in divisors if d > 1 and d % 2 != 0]
    divisors.sort()
    
    # Precompute bitmasks for each (m, r)
    # mask[m][r] will be an integer with bits set for residues r, r+m, r+2m...
    masks = {}
    for m in divisors:
        masks[m] = []
        for r in range(m):
            mask = 0
            # Set bits for r, r+m, r+2m, ...
            for x in range(r, N, m):
                mask |= (1 << x)
            masks[m].append(mask)
            
    # Greedy Construction
    covered = 0
    selected_classes = [] # List of [m, r]
    used_moduli = set()
    
    # We prioritize smaller moduli as they cover more residues
    for m in divisors:
        if time.time() > deadline:
            break
            
        best_r = 0
        best_gain = -1
        m_masks = masks[m]
        
        for r in range(m):
            # Count bits in (m_masks[r] & ~covered)
            new_bits = m_masks[r] & ~covered
            # Python int.bit_count() is fast in 3.10+
            gain = new_bits.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_r = r
                if gain == N // m: # Maximum possible gain for this m
                    break
                    
        if best_gain > 0:
            selected_classes.append([m, best_r])
            used_moduli.add(m)
            covered |= masks[m][best_r]
        else:
            # If no gain, we might still want to include it if it doesn't hurt?
            # No, we want to maximize coverage. If gain is 0, it adds nothing.
            # But the problem says "distinct moduli". It doesn't say we MUST use all.
            # "With every divisor ... available once".
            # Score is uncovered residues.
            # Adding a class with 0 gain doesn't change score, but uses up a modulus.
            # It's better to skip it to save time, or include it if it might help later?
            # Actually, if gain is 0, it covers nothing new. It's redundant.
            # We should skip it to keep the list small and potentially allow better swaps later?
            # Or just skip it.
            pass

    # Local Search
    # We will iterate until no improvement is found or time runs out.
    # Steps:
    # 1. Optimize residues of current set.
    # 2. Try swapping any selected modulus with any unselected modulus.
    # 3. Try adding any unselected modulus.
    # 4. Try removing any selected modulus if it's redundant.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Optimize residues
        for i, (m, r) in enumerate(selected_classes):
            if time.time() > deadline:
                return selected_classes
                
            # Calculate coverage without this class
            C_other = 0
            for j, (m_j, r_j) in enumerate(selected_classes):
                if j != i:
                    C_other |= masks[m_j][r_j]
            
            best_r = r
            best_gain = -1
            m_masks = masks[m]
            
            for r_new in range(m):
                new_bits = m_masks[r_new] & ~C_other
                gain = new_bits.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r_new
                    if gain == N // m:
                        break
                        
            if best_gain > 0 and best_r != r:
                old_total = covered.bit_count()
                new_total = (C_other | m_masks[best_r]).bit_count()
                if new_total > old_total:
                    selected_classes[i] = [m, best_r]
                    covered = C_other | m_masks[best_r]
                    improved = True
        
        # 2. Try Swaps (Systematic)
        if not improved and time.time() < deadline:
            unselected_moduli = [m for m in divisors if m not in used_moduli]
            
            for i, (m_old, r_old) in enumerate(selected_classes):
                if time.time() > deadline:
                    break
                    
                # Coverage without the old class
                C_without_old = 0
                for j, (m_j, r_j) in enumerate(selected_classes):
                    if j != i:
                        C_without_old |= masks[m_j][r_j]
                
                old_total = covered.bit_count()
                
                for m_new in unselected_moduli:
                    if time.time() > deadline:
                        break
                        
                    # Find best residue for m_new given C_without_old
                    best_r_new = 0
                    best_gain_new = -1
                    m_new_masks = masks[m_new]
                    
                    for r_new in range(m_new):
                        new_bits = m_new_masks[r_new] & ~C_without_old
                        gain = new_bits.bit_count()
                        if gain > best_gain_new:
                            best_gain_new = gain
                            best_r_new = r_new
                            if gain == N // m_new:
                                break
                                
                    new_total = (C_without_old | m_new_masks[best_r_new]).bit_count()
                    
                    if new_total > old_total:
                        # Perform swap
                        selected_classes[i] = [m_new, best_r_new]
                        used_moduli.remove(m_old)
                        used_moduli.add(m_new)
                        covered = C_without_old | m_new_masks[best_r_new]
                        improved = True
                        break # Break inner loop for m_new
                
                if improved:
                    break # Break outer loop for i, restart while loop
        
        # 3. Try Additions
        if not improved and time.time() < deadline:
            for m_new in divisors:
                if m_new in used_moduli:
                    continue
                if time.time() > deadline:
                    break
                    
                best_r_new = 0
                best_gain_new = -1
                m_new_masks = masks[m_new]
                
                for r_new in range(m_new):
                    new_bits = m_new_masks[r_new] & ~covered
                    gain = new_bits.bit_count()
                    if gain > best_gain_new:
                        best_gain_new = gain
                        best_r_new = r_new
                        if gain == N // m_new:
                            break
                            
                if best_gain_new > 0:
                    new_total = (covered | m_new_masks[best_r_new]).bit_count()
                    old_total = covered.bit_count()
                    if new_total > old_total:
                        selected_classes.append([m_new, best_r_new])
                        used_moduli.add(m_new)
                        covered = covered | m_new_masks[best_r_new]
                        improved = True
                        break # Restart while loop
        
        # 4. Try Deletions (Redundancy Removal)
        if not improved and time.time() < deadline:
            for i in range(len(selected_classes)):
                if time.time() > deadline:
                    break
                    
                m_old, r_old = selected_classes[i]
                # Check if removing this class reduces coverage
                C_without = 0
                for j, (m_j, r_j) in enumerate(selected_classes):
                    if j != i:
                        C_without |= masks[m_j][r_j]
                
                old_total = covered.bit_count()
                new_total = C_without.bit_count()
                
                if new_total == old_total:
                    # Redundant, remove it
                    selected_classes.pop(i)
                    used_moduli.remove(m_old)
                    covered = C_without
                    improved = True
                    break # Restart while loop

    return selected_classes
