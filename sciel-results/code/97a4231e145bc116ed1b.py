"""Optimized odd covering system search for moduli dividing 51975."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (evaluator kills at 150s)
    
    base = 51975
    # 51975 = 3^3 * 5^2 * 7 * 11
    # Get all odd divisors of base greater than 1
    # Since base is odd, all divisors are odd
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Precompute residue classes for each divisor
    # For divisor m, residues r, the indices are np.arange(r, base, m)
    # We'll store these as precomputed arrays for speed
    
    # Greedy initialization: add classes in order of increasing modulus
    # For each modulus, choose the residue that maximizes newly covered residues
    counts = np.zeros(base, dtype=np.int32)
    classes = []
    
    for m in divisors:
        if time.time() > deadline - 5:
            return classes
        
        best_r = 0
        best_gain = -1
        
        # For each residue r, count how many positions x ≡ r (mod m) are uncovered
        # This is sum of (counts[x] == 0) for x in {r, r+m, r+2m, ...}
        # Using vectorized operations
        # The positions for residue r are r, r+m, r+2m, ..., r+k*m where r+k*m < base
        # We can compute this efficiently
        
        # Number of terms: floor((base-1-r)/m) + 1 if r < base, else 0
        # Since r < m <= base, r is always < base
        
        # Efficient computation: for each r, we want sum of (counts[r::m] == 0)
        # This is equivalent to m - sum of (counts[r::m] > 0)
        # Or we can just count directly
        
        # To speed up, we can try residues in a smart order
        # But let's first try all residues for small m, and sample for large m
        
        if m <= 1000:
            # Try all residues
            for r in range(m):
                gain = np.count_nonzero(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
        else:
            # For large m, the number of positions per residue is small
            # base // m is small, so each residue class has few elements
            # We can try all residues since m * (base/m) = base operations total
            for r in range(m):
                gain = np.count_nonzero(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
        
        classes.append([m, best_r])
        counts[best_r::m] += 1
    
    # Local search: try to improve by changing residues of individual classes
    # Objective: maximize total covered residues (minimize uncovered)
    
    # First, compute current total covered
    total_covered = np.count_nonzero(counts > 0)
    
    improved = True
    max_passes = 100
    pass_count = 0
    
    while improved and pass_count < max_passes and time.time() < deadline - 10:
        improved = False
        pass_count += 1
        
        # Try to improve each class
        # Process classes in random order to escape local optima
        indices = list(range(len(classes)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5:
                return classes
            
            m, old_r = classes[idx]
            
            # Temporarily remove this class
            counts[old_r::m] -= 1
            base_total = np.count_nonzero(counts > 0)
            
            best_r = old_r
            best_total = base_total
            
            # Try alternative residues
            # For efficiency, we only need to check residues that might improve coverage
            # A residue r is beneficial if it covers positions that are currently uncovered
            # and removing old_r doesn't uncover too many positions
            
            # Key insight: changing from old_r to new_r
            # - Positions in old_r class that had count 1 become uncovered (loss)
            # - Positions in new_r class that had count 0 become covered (gain)
            # Net change = gain - loss
            
            # Let's compute this efficiently
            # loss = number of positions x ≡ old_r (mod m) where counts[x] == 1 (after removal, so before removal counts[x] == 2? No...)
            # Wait, we already removed old_r. So counts reflects state without this class.
            # If we add new_r, positions x ≡ new_r (mod m) get +1.
            # Gain = number of positions x ≡ new_r (mod m) where counts[x] == 0 (currently uncovered)
            # But we also need to consider that some of these might be in old_r class too
            # Since old_r and new_r are different residues mod m, the classes are disjoint!
            # So gain and loss are independent.
            
            # Actually, since we already removed old_r, the current counts don't include this class.
            # Adding new_r: gain = count of positions x ≡ new_r (mod m) with counts[x] == 0
            # The total would be base_total + gain (since old_r positions that were at count 1 
            # are now at count 0, but they were already counted in base_total as uncovered)
            
            # Wait, I need to be more careful.
            # Before removal: total_covered = T
            # After removal of old_r: base_total = T - (number of positions in old_r class that went from >=1 to 0)
            #   = T - (number of positions x ≡ old_r (mod m) where original counts[x] == 1)
            
            # If we add new_r: new_total = base_total + (number of positions x ≡ new_r (mod m) where counts_after_removal[x] == 0)
            
            # Since old_r != new_r, the classes are disjoint, so:
            # positions in new_r class are unaffected by the removal of old_r
            # So counts_after_removal[x] for x ≡ new_r (mod m) is the same as original counts[x]
            
            # Therefore: new_total = base_total + (number of positions x ≡ new_r (mod m) where original counts[x] == 0)
            
            # And base_total = total_covered - (number of positions x ≡ old_r (mod m) where original counts[x] == 1)
            
            # So: new_total = total_covered - loss_old + gain_new
            # where loss_old = count of x ≡ old_r (mod m) with counts_original[x] == 1
            # and gain_new = count of x ≡ new_r (mod m) with counts_original[x] == 0
            
            # But we've already modified counts (removed old_r). Let's just work with current counts.
            # Current counts = original - indicator(old_r class)
            
            # For new_r (new_r != old_r), the current counts at positions in new_r class are unchanged.
            # So gain = np.count_nonzero(counts[new_r::m] == 0)
            # And the new total = base_total + gain
            
            # We want to maximize base_total + gain, which is equivalent to maximizing gain
            # since base_total is constant for this class.
            
            # So we just need to find the residue r that maximizes gain = count of uncovered positions in that class.
            # But wait, we also need to account for the fact that we're replacing old_r.
            # The net change from current state (with old_r removed) is:
            #   If we keep old_r: total = base_total + loss_old (where loss_old = positions in old_r that are now 0 but were 1)
            #     Actually, base_total already has them as 0. Adding old_r back would cover positions where counts==0 in old_r class.
            #     So total_with_old_r = base_total + (count of x ≡ old_r (mod m) where current_counts[x] == 0)
            #     But current_counts[x] for x ≡ old_r (mod m) = original_counts[x] - 1
            #     So current_counts[x] == 0 iff original_counts[x] == 1
            #     So total_with_old_r = base_total + loss_old = total_covered (as expected)
            
            #   If we switch to new_r: total = base_total + gain_new
            #     where gain_new = count of x ≡ new_r (mod m) where current_counts[x] == 0
            #     = count of x ≡ new_r (mod m) where original_counts[x] == 0 (since new_r != old_r)
            
            # So we want to find new_r that maximizes gain_new, and compare with loss_old (the gain from keeping old_r).
            # If max_gain_new > loss_old, we should switch to the best new_r.
            
            # Let's compute loss_old first
            loss_old = np.count_nonzero(counts[old_r::m] == 0)  # This is the number of positions in old_r class that are now uncovered
            
            # Now find best new_r
            # For efficiency, we can skip old_r itself
            best_gain = loss_old  # At least we can keep old_r
            best_r = old_r
            
            if m <= 2000:
                for r in range(m):
                    if r == old_r:
                        continue
                    gain = np.count_nonzero(counts[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
            else:
                # For large m, sample residues or try all (since base/m is small)
                # Number of positions per class is base//m + 1 or so
                # If base//m is small, trying all m residues is m * (base/m) = base ops, which is fine
                for r in range(m):
                    if r == old_r:
                        continue
                    gain = np.count_nonzero(counts[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
            
            # Apply the best choice
            if best_r != old_r:
                # We need to add best_r to counts (old_r is already removed)
                counts[best_r::m] += 1
                classes[idx][1] = best_r
                improved = True
                total_covered = base_total + best_gain
            else:
                # Keep old_r, add it back
                counts[old_r::m] += 1
                # total_covered stays the same
    
    return classes
# EVOLVE-BLOCK-END
