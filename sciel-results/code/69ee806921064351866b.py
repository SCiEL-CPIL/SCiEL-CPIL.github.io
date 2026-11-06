import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    base = 51975
    
    # Find all divisors of base
    divisors = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    
    # Filter: m > 1, odd (all divisors of 51975 are odd)
    divisors = [m for m in divisors if m > 1]
    divisors.sort()
    
    # Precompute bitset masks for each (m, r) pair
    # mask[m][r] is a Python int where bit i is set iff i % m == r
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask = 0
            for i in range(r, base, m):
                mask |= (1 << i)
            masks[m][r] = mask
    
    # Full mask: all bits set for positions 0..base-1
    full_mask = (1 << base) - 1
    
    best_uncovered = base
    best_classes = []
    
    def count_uncovered(covered_mask):
        """Count number of uncovered residues."""
        # Popcount of complement within full_mask
        return bin(full_mask & ~covered_mask).count('1')
    
    def greedy_search(run_idx, deadline):
        """Run one greedy search with randomization."""
        if time.time() > deadline - 3:
            return None, 0
        
        # Randomize the order of moduli processing
        moduli_order = divisors[:]
        if run_idx > 0:
            random.shuffle(moduli_order)
        
        covered_mask = 0
        used_moduli = set()
        classes = []
        
        for m in moduli_order:
            if time.time() > deadline - 2:
                break
            
            # Find best residue for this modulus
            best_r = 0
            best_gain = -1
            for r in range(m):
                mask = masks[m][r]
                gain = bin(mask & ~covered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain <= 0:
                continue
            
            covered_mask |= masks[m][best_r]
            used_moduli.add(m)
            classes.append([m, best_r])
        
        # Local search: try to improve by removing redundant classes
        # and trying alternative residues
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try removing each class and see if it's redundant
            for i in range(len(classes)):
                if time.time() > deadline - 2:
                    break
                m, r = classes[i]
                # Recompute coverage without this class
                temp_mask = 0
                for j, (mj, rj) in enumerate(classes):
                    if j != i:
                        temp_mask |= masks[mj][rj]
                
                # Check if removing this class increases uncovered
                unc_with = count_uncovered(temp_mask)
                unc_without = count_uncovered(covered_mask)
                
                if unc_with <= unc_without:
                    # Removing is neutral or better, try to find better residue for m
                    best_r_new = r
                    best_gain_new = -1
                    for r2 in range(m):
                        # What would coverage be if we use r2 instead?
                        test_mask = temp_mask | masks[m][r2]
                        gain = bin(masks[m][r2] & ~temp_mask).count('1')
                        if gain > best_gain_new:
                            best_gain_new = gain
                            best_r_new = r2
                    
                    if best_gain_new > 0:
                        classes[i] = [m, best_r_new]
                        covered_mask = temp_mask | masks[m][best_r_new]
                        improved = True
                    else:
                        # Class is truly redundant, remove it
                        classes.pop(i)
                        covered_mask = temp_mask
                        improved = True
                        break  # Restart loop since indices changed
                else:
                    # Try to find a better residue for this class
                    best_r_new = r
                    best_unc = unc_without
                    for r2 in range(m):
                        if r2 == r:
                            continue
                        test_mask = 0
                        for j, (mj, rj) in enumerate(classes):
                            if j == i:
                                test_mask |= masks[m][r2]
                            else:
                                test_mask |= masks[mj][rj]
                        unc = count_uncovered(test_mask)
                        if unc < best_unc:
                            best_unc = unc
                            best_r_new = r2
                    
                    if best_r_new != r:
                        classes[i] = [m, best_r_new]
                        # Recompute covered_mask
                        covered_mask = 0
                        for mj, rj in classes:
                            covered_mask |= masks[mj][rj]
                        improved = True
                        break  # Restart loop
            
            # Try adding back a removed modulus with a different residue
            for m in divisors:
                if m in used_moduli:
                    continue
                if time.time() > deadline - 2:
                    break
                
                best_r_new = None
                best_unc = count_uncovered(covered_mask)
                for r in range(m):
                    test_mask = covered_mask | masks[m][r]
                    unc = count_uncovered(test_mask)
                    if unc < best_unc:
                        best_unc = unc
                        best_r_new = r
                
                if best_r_new is not None:
                    classes.append([m, best_r_new])
                    used_moduli.add(m)
                    covered_mask |= masks[m][best_r_new]
                    improved = True
                    break  # Restart loop
        
        final_unc = count_uncovered(covered_mask)
        return classes, final_unc
    
    # Run multiple greedy searches
    num_runs = 30
    for run in range(num_runs):
        if time.time() > deadline - 5:
            break
        
        # Vary randomization strategy
        if run == 0:
            # Pure greedy: largest moduli first (cover more per class)
            moduli_order = sorted(divisors, reverse=True)
        elif run == 1:
            # Smallest moduli first
            moduli_order = sorted(divisors)
        elif run == 2:
            # Prime factors first
            prime_factors = [3, 5, 7, 11]
            rest = [m for m in divisors if m not in prime_factors]
            moduli_order = prime_factors + rest
        else:
            moduli_order = divisors[:]
            random.shuffle(moduli_order)
        
        # Temporarily override the moduli order for this run
        # We'll do a custom greedy here
        if time.time() > deadline - 3:
            break
        
        covered_mask = 0
        used_moduli = set()
        classes = []
        
        for m in moduli_order:
            if time.time() > deadline - 2:
                break
            
            best_r = 0
            best_gain = -1
            for r in range(m):
                mask = masks[m][r]
                gain = bin(mask & ~covered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain <= 0:
                continue
            
            covered_mask |= masks[m][best_r]
            used_moduli.add(m)
            classes.append([m, best_r])
        
        # Local search polish
        improved = True
        polish_iterations = 0
        while improved and polish_iterations < 20 and time.time() < deadline - 2:
            improved = False
            polish_iterations += 1
            
            # Try removing redundant classes
            for i in range(len(classes)):
                if time.time() > deadline - 2:
                    break
                m, r = classes[i]
                temp_mask = 0
                for j, (mj, rj) in enumerate(classes):
                    if j != i:
                        temp_mask |= masks[mj][rj]
                
                unc_with = count_uncovered(temp_mask)
                unc_without = count_uncovered(covered_mask)
                
                if unc_with <= unc_without:
                    # Try to find better residue for m
                    best_r_new = r
                    best_gain_new = -1
                    for r2 in range(m):
                        gain = bin(masks[m][r2] & ~temp_mask).count('1')
                        if gain > best_gain_new:
                            best_gain_new = gain
                            best_r_new = r2
                    
                    if best_gain_new > 0:
                        classes[i] = [m, best_r_new]
                        covered_mask = temp_mask | masks[m][best_r_new]
                        improved = True
                    else:
                        classes.pop(i)
                        covered_mask = temp_mask
                        improved = True
                        break
                else:
                    # Try alternative residues
                    best_r_new = r
                    best_unc = unc_without
                    for r2 in range(m):
                        if r2 == r:
                            continue
                        test_mask = 0
                        for j, (mj, rj) in enumerate(classes):
                            if j == i:
                                test_mask |= masks[m][r2]
                            else:
                                test_mask |= masks[mj][rj]
                        unc = count_uncovered(test_mask)
                        if unc < best_unc:
                            best_unc = unc
                            best_r_new = r2
                    
                    if best_r_new != r:
                        classes[i] = [m, best_r_new]
                        covered_mask = 0
                        for mj, rj in classes:
                            covered_mask |= masks[mj][rj]
                        improved = True
                        break
            
            # Try adding back unused moduli
            if not improved:
                for m in divisors:
                    if m in used_moduli:
                        continue
                    if time.time() > deadline - 2:
                        break
                    
                    best_r_new = None
                    best_unc = count_uncovered(covered_mask)
                    for r in range(m):
                        test_mask = covered_mask | masks[m][r]
                        unc = count_uncovered(test_mask)
                        if unc < best_unc:
                            best_unc = unc
                            best_r_new = r
                    
                    if best_r_new is not None:
                        classes.append([m, best_r_new])
                        used_moduli.add(m)
                        covered_mask |= masks[m][best_r_new]
                        improved = True
                        break
        
        final_unc = count_uncovered(covered_mask)
        if final_unc < best_uncovered:
            best_uncovered = final_unc
            best_classes = [c[:] for c in classes]
        
        if best_uncovered == 0:
            return best_classes
    
    return best_classes
