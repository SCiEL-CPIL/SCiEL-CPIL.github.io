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
    
    # Precompute index arrays for each divisor's residue classes
    # For speed, we'll compute gains vectorized where possible
    
    # Greedy initialization: process moduli in order of increasing size
    # For each modulus, choose the residue that maximizes newly covered residues
    counts = np.zeros(base, dtype=np.int32)
    classes = []
    used_moduli = set()
    
    for m in divisors:
        if time.time() > deadline - 8:
            break
        
        best_r = 0
        best_gain = -1
        
        # Compute gain for each residue efficiently
        # For residue r, gain = number of positions x ≡ r (mod m) with counts[x] == 0
        # We can compute this by summing over all r at once using a clever trick:
        # Create an array where position x contributes to residue x % m
        # But that's not directly vectorizable. Instead, loop over residues but use
        # the fact that for large m, each residue class has few elements.
        
        # For small m, try all residues
        # For large m, the number of positions per residue is small (base // m)
        # So trying all m residues costs about base operations total
        
        for r in range(m):
            gain = np.count_nonzero(counts[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        classes.append([m, best_r])
        counts[best_r::m] += 1
        used_moduli.add(m)
    
    # Local search with multiple move types:
    # 1. Residue change: change residue of an existing class
    # 2. Class removal: remove a class if it covers no unique residues
    # 3. Class addition: add a new class using an unused modulus
    
    # Helper function to compute total covered
    def compute_total_covered():
        return np.count_nonzero(counts > 0)
    
    total_covered = compute_total_covered()
    best_total_covered = total_covered
    best_classes = [c[:] for c in classes]
    
    improved = True
    max_passes = 200
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
            
            # Compute loss: positions in old_r class that are now uncovered
            loss_old = np.count_nonzero(counts[old_r::m] == 0)
            
            # Find best new residue
            best_gain = loss_old  # Gain from keeping old_r
            best_r = old_r
            
            # Try all residues
            for r in range(m):
                if r == old_r:
                    continue
                gain = np.count_nonzero(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            # Apply the best choice
            if best_r != old_r:
                counts[best_r::m] += 1
                classes[idx][1] = best_r
                improved = True
                no_improve_count = 0
            else:
                counts[old_r::m] += 1
            
            # Check if we improved total covered
            current_total = compute_total_covered()
            if current_total > best_total_covered:
                best_total_covered = current_total
                best_classes = [c[:] for c in classes]
        
        if time.time() > deadline - 8:
            break
        
        # Try class removal: remove a class if it doesn't cover any unique residues
        # A class is redundant if all positions it covers are also covered by other classes
        # i.e., removing it doesn't decrease total covered
        for idx in range(len(classes) - 1, -1, -1):
            if time.time() > deadline - 8:
                break
            
            m, r = classes[idx]
            
            # Check if removing this class reduces total covered
            counts[r::m] -= 1
            new_total = compute_total_covered()
            
            if new_total >= total_covered:
                # Removing this class doesn't hurt (or helps if there was overcounting)
                # Remove it
                classes.pop(idx)
                used_moduli.discard(m)
                total_covered = new_total
                improved = True
                no_improve_count = 0
            else:
                # Put it back
                counts[r::m] += 1
        
        if time.time() > deadline - 8:
            break
        
        # Try class addition: add a new class using an unused modulus
        if len(classes) < len(divisors):
            unused = [m for m in divisors if m not in used_moduli]
            if unused:
                # Pick the unused modulus that gives the best gain
                best_m = None
                best_r = None
                best_gain = 0
                
                for m in unused:
                    if time.time() > deadline - 8:
                        break
                    
                    # Find best residue for this modulus
                    for r in range(m):
                        gain = np.count_nonzero(counts[r::m] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_m = m
                            best_r = r
                
                if best_m is not None and best_gain > 0:
                    classes.append([best_m, best_r])
                    counts[best_r::best_m] += 1
                    used_moduli.add(best_m)
                    total_covered = compute_total_covered()
                    improved = True
                    no_improve_count = 0
        
        # Update best solution if improved
        if total_covered > best_total_covered:
            best_total_covered = total_covered
            best_classes = [c[:] for c in classes]
        
        # Early termination if no improvement for several passes
        if no_improve_count >= 5:
            break
    
    # Return the best solution found
    return best_classes if best_classes else classes
# EVOLVE-BLOCK-END
