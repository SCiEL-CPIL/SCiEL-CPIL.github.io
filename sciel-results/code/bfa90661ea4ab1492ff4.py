import time
import math
import numpy as np

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s buffer for safety

    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    
    divisors.sort()
    num_divisors = len(divisors)
    
    # 2. Precompute masks for each (m, r) using numpy
    # masks[m_idx][r] is a boolean array of length 'base'
    # We store these in a dictionary for easy access by modulus value
    masks = {}
    for m in divisors:
        masks[m] = np.zeros((m, base), dtype=bool)
        for r in range(m):
            # Set True at indices r, r+m, ...
            masks[m][r, r::m] = True

    # 3. Greedy Set Cover Initialization
    # uncovered is a boolean array where True means "not covered"
    uncovered = np.ones(base, dtype=bool)
    remaining_moduli = set(divisors)
    classes = []
    
    # Precompute gains for initial greedy selection
    # gain[m][r] = number of True in (uncovered & masks[m][r])
    # We will recompute this as uncovered changes
    
    while remaining_moduli:
        if time.time() > deadline - 20.0:
            break
            
        best_gain = 0
        best_m = None
        best_r = None
        
        # Find the modulus and residue that maximizes coverage of uncovered
        for m in remaining_moduli:
            # Vectorized calculation of gains for all r for this m
            # masks[m] is shape (m, base)
            # We want count of (uncovered & masks[m][r]) for each r
            # This is sum along axis 1 of (uncovered[None, :] & masks[m])
            gains = np.count_nonzero(uncovered[None, :] & masks[m], axis=1)
            max_gain = np.max(gains)
            
            if max_gain > best_gain:
                best_gain = max_gain
                best_m = m
                best_r = np.argmax(gains)
                
        if best_gain == 0:
            break
            
        # Select the best class
        classes.append([best_m, best_r])
        
        # Update uncovered
        uncovered &= ~masks[best_m][best_r]
        
        # Remove the modulus from consideration
        remaining_moduli.remove(best_m)
        
    # 4. Local Search Refinement
    # We maintain a list of selected classes.
    # We will try to improve by:
    # a) Optimizing residues for current moduli
    # b) Swapping moduli
    # c) Adding/removing moduli (though distinct moduli constraint makes adding/removing tricky if we want to keep 'distinct')
    # Actually, the constraint is "distinct moduli". It does not say "maximal set".
    # However, usually more classes = better coverage.
    # Let's stick to optimizing residues and swapping moduli.
    
    selected_moduli = {c[0] for c in classes}
    
    # Helper to calculate current score
    def get_score():
        return np.count_nonzero(uncovered)
    
    current_score = get_score()
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # --- Step 4a: Optimize Residues ---
        # For each selected modulus, find the best residue r that minimizes uncovered count
        # given the other selected moduli.
        
        # Calculate coverage of other classes
        # other_covered = OR of masks for all other selected classes
        # If we change class i to (m, r), new_uncovered = ~( (other_covered | mask[m][r]) )
        # We want to minimize count of new_uncovered, which is equivalent to maximizing count of (other_covered | mask[m][r])
        # count(other_covered | mask) = count(other_covered) + count(mask & ~other_covered)
        # Since count(other_covered) is constant for a fixed set of other classes,
        # we just need to maximize count(mask & ~other_covered).
        
        # Precompute ~other_covered for each selected class?
        # That's expensive. Let's do it one by one.
        
        for i, (m, r) in enumerate(classes):
            if time.time() > deadline - 5.0:
                break
            
            # Calculate coverage of all OTHER classes
            # Start with all False
            other_covered = np.zeros(base, dtype=bool)
            for j, (m_j, r_j) in enumerate(classes):
                if j != i:
                    other_covered |= masks[m_j][r_j]
            
            # We want to find r' in [0, m) that maximizes count(masks[m][r'] & ~other_covered)
            # Let potential = ~other_covered
            # gains = count(masks[m] & potential)
            potential = ~other_covered
            gains = np.count_nonzero(masks[m] & potential[None, :], axis=1)
            best_r_new = np.argmax(gains)
            
            if best_r_new != r:
                # Check if it actually improves the global score
                # new_uncovered = ~(other_covered | masks[m][best_r_new])
                # old_uncovered = ~(other_covered | masks[m][r])
                # We can just compute the new score
                new_covered = other_covered | masks[m][best_r_new]
                new_score = base - np.count_nonzero(new_covered)
                
                if new_score < current_score:
                    classes[i] = [m, best_r_new]
                    current_score = new_score
                    improved = True
                    # Update global uncovered
                    # Recalculate uncovered from scratch or update incrementally?
                    # Incremental is safer:
                    # uncovered = ~(OR of all masks)
                    # Let's just recalculate uncovered to be safe, it's O(N * classes)
                    # Or better:
                    # old_mask = masks[m][r]
                    # new_mask = masks[m][best_r_new]
                    # covered = OR of all
                    # This is getting complex. Let's just recalculate uncovered at the end of the loop iteration if improved.
                    
        # Recalculate uncovered if improved
        if improved:
            covered = np.zeros(base, dtype=bool)
            for m, r in classes:
                covered |= masks[m][r]
            uncovered = ~covered
            current_score = np.count_nonzero(uncovered)
            continue # Restart the loop to try more optimizations or swaps
        
        # --- Step 4b: Swap Moduli ---
        # Try swapping a selected modulus m_out with an unselected modulus m_in
        # For a given m_out, r_out and m_in, r_in:
        # new_uncovered = (uncovered | mask_out) & ~mask_in
        # We want to minimize count(new_uncovered)
        
        # To speed up, for each m_in, find the best r_in that maximizes coverage of (uncovered | mask_out)
        # Let target = uncovered | mask_out
        # We want to maximize count(mask_in & target)
        
        candidates_in = []
        for m_in in divisors:
            if m_in in selected_moduli:
                continue
            # We will evaluate this against each m_out later, or find a generic best?
            # The best r_in depends on m_out. So we must do this inside the m_out loop.
            pass
        
        for i, (m_out, r_out) in enumerate(classes):
            if time.time() > deadline - 5.0:
                break
                
            mask_out = masks[m_out][r_out]
            # The set of residues that would become uncovered if we remove m_out
            # is those covered ONLY by m_out.
            # Actually, let's use the formula:
            # new_uncovered = (uncovered | mask_out) & ~mask_in
            # Let potential_uncovered = uncovered | mask_out
            # We want to choose m_in, r_in to minimize count(potential_uncovered & ~mask_in)
            # This is equivalent to maximizing count(potential_uncovered & mask_in)
            
            potential = uncovered | mask_out
            
            best_gain = 0
            best_m_in = None
            best_r_in = None
            
            for m_in in divisors:
                if m_in in selected_moduli:
                    continue
                
                # Maximize count(potential & masks[m_in])
                gains = np.count_nonzero(potential[None, :] & masks[m_in], axis=1)
                max_gain = np.max(gains)
                
                if max_gain > best_gain:
                    best_gain = max_gain
                    best_m_in = m_in
                    best_r_in = np.argmax(gains)
            
            if best_m_in is not None and best_gain > 0:
                # Calculate new score
                # new_uncovered = potential & ~masks[best_m_in][best_r_in]
                new_uncovered = potential & ~masks[best_m_in][best_r_in]
                new_score = np.count_nonzero(new_uncovered)
                
                if new_score < current_score:
                    # Accept swap
                    classes[i] = [best_m_in, best_r_in]
                    selected_moduli.remove(m_out)
                    selected_moduli.add(best_m_in)
                    current_score = new_score
                    improved = True
                    # Update global uncovered
                    covered = np.zeros(base, dtype=bool)
                    for m, r in classes:
                        covered |= masks[m][r]
                    uncovered = ~covered
                    break # Restart loop to re-evaluate with new state

    return classes
