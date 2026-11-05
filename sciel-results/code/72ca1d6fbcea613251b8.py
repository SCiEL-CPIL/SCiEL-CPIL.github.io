"""Optimized odd covering system search for moduli dividing 51975."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin for 150s kill limit
    
    base = 51975
    # 51975 = 3^3 * 5^2 * 7 * 11
    # Get all odd divisors of base greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Precompute indices for fast residue access? 
    # We will use vectorized operations for gain calculation.
    
    # Greedy initialization: process moduli in order of increasing size
    counts = np.zeros(base, dtype=np.int32)
    classes = []
    used_moduli = set()
    
    for m in divisors:
        if time.time() > deadline - 8:
            break
        
        # Vectorized gain calculation for all residues of m
        # Gain[r] = number of x in [0, base) such that x % m == r and counts[x] == 0
        # We can use bincount on (x % m) with weights (counts[x] == 0)
        residues = np.arange(base) % m
        weights = (counts == 0).astype(np.int32)
        gains = np.bincount(residues, weights=weights, minlength=m)
        
        best_r = np.argmax(gains)
        classes.append([m, int(best_r)])
        counts[best_r::m] += 1
        used_moduli.add(m)
    
    def compute_total_covered():
        return np.count_nonzero(counts > 0)
    
    total_covered = compute_total_covered()
    best_total_covered = total_covered
    best_classes = [c[:] for c in classes]
    
    improved = True
    max_passes = 500
    pass_count = 0
    no_improve_count = 0
    
    while improved and pass_count < max_passes and time.time() < deadline - 10:
        improved = False
        pass_count += 1
        no_improve_count += 1
        
        # Try residue changes for each class
        indices = list(range(len(classes)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 8:
                break
            
            m, old_r = classes[idx]
            
            # Temporarily remove this class
            counts[old_r::m] -= 1
            
            # Calculate gains for all residues if class idx is removed
            residues = np.arange(base) % m
            weights = (counts == 0).astype(np.int32)
            gains = np.bincount(residues, weights=weights, minlength=m)
            
            best_r = int(np.argmax(gains))
            
            # Check if best_r is different and provides improvement
            # The gain of best_r is gains[best_r].
            # The gain of old_r (if we kept it) would be gains[old_r].
            # But we are comparing total coverage.
            # If we choose best_r, the new total covered is:
            # current_total_covered - (positions in old_r that are now unique) + (positions in best_r that are now unique)
            # Actually, it's easier to just apply and check.
            
            if best_r != old_r:
                # Apply best_r
                counts[best_r::m] += 1
                new_total = compute_total_covered()
                
                if new_total > total_covered:
                    total_covered = new_total
                    classes[idx][1] = best_r
                    improved = True
                    no_improve_count = 0
                    if total_covered > best_total_covered:
                        best_total_covered = total_covered
                        best_classes = [c[:] for c in classes]
                else:
                    # Revert
                    counts[best_r::m] -= 1
                    counts[old_r::m] += 1
            else:
                # Revert
                counts[old_r::m] += 1
        
        if time.time() > deadline - 8:
            break
        
        # Try class removal
        for idx in range(len(classes) - 1, -1, -1):
            if time.time() > deadline - 8:
                break
            
            m, r = classes[idx]
            counts[r::m] -= 1
            new_total = compute_total_covered()
            
            if new_total >= total_covered:
                classes.pop(idx)
                used_moduli.discard(m)
                total_covered = new_total
                improved = True
                no_improve_count = 0
                if total_covered > best_total_covered:
                    best_total_covered = total_covered
                    best_classes = [c[:] for c in classes]
            else:
                counts[r::m] += 1
        
        if time.time() > deadline - 8:
            break
        
        # Try class addition
        if len(classes) < len(divisors):
            unused = [m for m in divisors if m not in used_moduli]
            if unused:
                best_m = None
                best_r = None
                best_gain = 0
                
                for m in unused:
                    if time.time() > deadline - 8:
                        break
                    
                    residues = np.arange(base) % m
                    weights = (counts == 0).astype(np.int32)
                    gains = np.bincount(residues, weights=weights, minlength=m)
                    
                    max_gain = np.max(gains)
                    if max_gain > best_gain:
                        best_gain = max_gain
                        best_m = m
                        best_r = int(np.argmax(gains))
                
                if best_m is not None and best_gain > 0:
                    classes.append([best_m, best_r])
                    counts[best_r::best_m] += 1
                    used_moduli.add(best_m)
                    total_covered = compute_total_covered()
                    improved = True
                    no_improve_count = 0
                    if total_covered > best_total_covered:
                        best_total_covered = total_covered
                        best_classes = [c[:] for c in classes]
        
        if no_improve_count >= 10:
            break
    
    return best_classes if best_classes else classes
# EVOLVE-BLOCK-END
