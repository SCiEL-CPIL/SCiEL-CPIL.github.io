import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    base = 51975
    # 51975 = 3^3 * 5^2 * 7 * 11
    # Get all odd divisors of base greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Pre-compute residues modulo m for each m?
    # np.arange(base) % m is fast enough.
    
    def compute_gains(counts, m):
        """
        Returns an array of gains for each residue r in [0, m).
        gain[r] = number of indices i such that i % m == r and counts[i] == 0
        """
        residues = np.arange(base) % m
        weights = (counts == 0).astype(np.int32)
        gains = np.bincount(residues, weights=weights, minlength=m)
        return gains

    def get_best_r(counts, m):
        gains = compute_gains(counts, m)
        best_r = int(np.argmax(gains))
        return best_r, int(gains[best_r])

    def greedy_init(seed):
        rng = np.random.RandomState(seed)
        counts = np.zeros(base, dtype=np.int32)
        classes = []
        used_moduli = set()
        remaining = list(divisors)
        rng.shuffle(remaining) # Randomize order to break symmetry
        
        while remaining and time.time() < deadline - 10:
            best_m = None
            best_r = None
            best_gain = 0
            
            # Find the modulus with the highest potential gain
            # To speed up, we can sample or just iterate all
            for m in remaining:
                r, gain = get_best_r(counts, m)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
            
            if best_m is None or best_gain == 0:
                break
            
            classes.append([best_m, best_r])
            counts[best_r::best_m] += 1
            used_moduli.add(best_m)
            remaining.remove(best_m)
            
        return classes, counts, used_moduli

    def local_search(classes, counts, used_moduli, max_passes=2000):
        total_covered = np.count_nonzero(counts > 0)
        best_total_covered = total_covered
        best_classes = [c[:] for c in classes]
        best_counts = counts.copy()
        
        no_improve_count = 0
        rng = np.random.RandomState(123)
        
        for p in range(max_passes):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # 1. Try to optimize residues of existing classes
            indices = list(range(len(classes)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5:
                    break
                m, old_r = classes[idx]
                
                # Temporarily remove
                counts[old_r::m] -= 1
                
                # Find best new residue
                new_r, gain = get_best_r(counts, m)
                
                if new_r != old_r:
                    # Apply new residue
                    counts[new_r::m] += 1
                    new_total = np.count_nonzero(counts > 0)
                    
                    if new_total > total_covered:
                        total_covered = new_total
                        classes[idx][1] = new_r
                        improved = True
                        no_improve_count = 0
                        if total_covered > best_total_covered:
                            best_total_covered = total_covered
                            best_classes = [c[:] for c in classes]
                            best_counts = counts.copy()
                    else:
                        # Revert
                        counts[new_r::m] -= 1
                        counts[old_r::m] += 1
                else:
                    # Revert
                    counts[old_r::m] += 1
            
            # 2. Try to add new classes
            unused = [m for m in divisors if m not in used_moduli]
            if unused:
                best_add_m = None
                best_add_r = None
                best_add_gain = 0
                
                # Shuffle unused to randomize
                rng.shuffle(unused)
                
                for m in unused:
                    if time.time() > deadline - 5:
                        break
                    r, gain = get_best_r(counts, m)
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_m = m
                        best_add_r = r
                
                if best_add_m is not None and best_add_gain > 0:
                    classes.append([best_add_m, best_add_r])
                    counts[best_add_r::best_add_m] += 1
                    used_moduli.add(best_add_m)
                    total_covered = np.count_nonzero(counts > 0)
                    improved = True
                    no_improve_count = 0
                    if total_covered > best_total_covered:
                        best_total_covered = total_covered
                        best_classes = [c[:] for c in classes]
                        best_counts = counts.copy()

            # 3. Try to remove redundant classes
            # A class is redundant if all its residues are covered by others
            for idx in range(len(classes) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                m, r = classes[idx]
                # Check if removing it increases uncovered
                # i.e., if any count becomes 0
                # We can check min count on the slice
                if np.min(counts[r::m]) > 1:
                    # Safe to remove
                    counts[r::m] -= 1
                    classes.pop(idx)
                    used_moduli.discard(m)
                    total_covered = np.count_nonzero(counts > 0)
                    improved = True
                    no_improve_count = 0
                    if total_covered > best_total_covered:
                        best_total_covered = total_covered
                        best_classes = [c[:] for c in classes]
                        best_counts = counts.copy()

            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            else:
                no_improve_count = 0

        return best_classes, best_total_covered

    best_overall_classes = None
    best_overall_covered = -1
    
    # Multiple restarts
    # Allocate time for ~5 restarts
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        seed = i * 100 + 42
        classes, counts, used_moduli = greedy_init(seed)
        
        if not classes:
            continue
        
        best_classes, best_covered = local_search(classes, counts, used_moduli)
        
        if best_covered > best_overall_covered:
            best_overall_covered = best_covered
            best_overall_classes = best_classes

    if best_overall_classes is None:
        # Fallback
        best_overall_classes = []

    return best_overall_classes
