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
    
    def compute_total_covered(counts):
        return np.count_nonzero(counts > 0)
    
    def greedy_init_randomized(seed, time_budget):
        """Randomized greedy initialization with adaptive look-ahead."""
        rng = np.random.RandomState(seed)
        counts = np.zeros(base, dtype=np.int32)
        classes = []
        used_moduli = set()
        remaining = list(divisors)
        
        while remaining and time.time() < deadline - time_budget:
            # Calculate marginal gain for each remaining modulus
            gains = []
            for m in remaining:
                residues = np.arange(base) % m
                weights = (counts == 0).astype(np.int32)
                g = np.bincount(residues, weights=weights, minlength=m)
                gains.append(np.max(g))
            
            if not gains or max(gains) == 0:
                break
            
            # Adaptive look-ahead: probabilistically select based on gain
            # Use softmax-like selection with temperature
            gains_arr = np.array(gains, dtype=np.float64)
            if gains_arr.max() == 0:
                break
            
            # Temperature decreases as we use more time
            elapsed = time.time() - start_time
            temp = max(0.1, 1.0 - elapsed / 30.0)
            
            # Normalize gains for probability
            gains_norm = gains_arr / (gains_arr.sum() + 1e-10)
            
            # Apply temperature for exploration
            if temp < 0.5:
                # More exploitative
                probs = gains_norm ** (1.0 / temp)
            else:
                # More exploratory
                probs = gains_norm
            
            probs = probs / (probs.sum() + 1e-10)
            
            # Select a modulus
            idx = rng.choice(len(remaining), p=probs)
            m = remaining[idx]
            remaining.pop(idx)
            
            # Find best residue for this modulus
            residues = np.arange(base) % m
            weights = (counts == 0).astype(np.int32)
            gains_m = np.bincount(residues, weights=weights, minlength=m)
            best_r = int(np.argmax(gains_m))
            
            classes.append([m, best_r])
            counts[best_r::m] += 1
            used_moduli.add(m)
        
        return classes, counts, used_moduli
    
    def local_search(classes, counts, time_budget):
        """Local search with residue changes, removals, and additions."""
        total_covered = compute_total_covered(counts)
        best_total_covered = total_covered
        best_classes = [c[:] for c in classes]
        used_moduli = set(m for m, r in classes)
        
        improved = True
        max_passes = 200
        pass_count = 0
        no_improve_count = 0
        rng = np.random.RandomState(42)
        
        while improved and pass_count < max_passes and time.time() < deadline - time_budget:
            improved = False
            pass_count += 1
            no_improve_count += 1
            
            # Try residue changes for each class
            indices = list(range(len(classes)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - time_budget - 5:
                    break
                
                m, old_r = classes[idx]
                
                # Temporarily remove this class
                counts[old_r::m] -= 1
                
                # Calculate gains for all residues if class idx is removed
                residues = np.arange(base) % m
                weights = (counts == 0).astype(np.int32)
                gains = np.bincount(residues, weights=weights, minlength=m)
                
                best_r = int(np.argmax(gains))
                
                if best_r != old_r:
                    # Apply best_r
                    counts[best_r::m] += 1
                    new_total = compute_total_covered(counts)
                    
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
            
            if time.time() > deadline - time_budget - 5:
                break
            
            # Try class removal
            for idx in range(len(classes) - 1, -1, -1):
                if time.time() > deadline - time_budget - 5:
                    break
                
                m, r = classes[idx]
                counts[r::m] -= 1
                new_total = compute_total_covered(counts)
                
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
            
            if time.time() > deadline - time_budget - 5:
                break
            
            # Try class addition
            if len(classes) < len(divisors):
                unused = [m for m in divisors if m not in used_moduli]
                if unused:
                    best_m = None
                    best_r = None
                    best_gain = 0
                    
                    for m in unused:
                        if time.time() > deadline - time_budget - 5:
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
                        total_covered = compute_total_covered(counts)
                        improved = True
                        no_improve_count = 0
                        if total_covered > best_total_covered:
                            best_total_covered = total_covered
                            best_classes = [c[:] for c in classes]
            
            if no_improve_count >= 8:
                break
        
        return best_classes, best_total_covered
    
    # Multiple restarts with different seeds
    best_overall_classes = None
    best_overall_covered = 0
    
    # Allocate time: 40s for initialization phase, rest for local search
    init_budget = 40
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        seed = restart * 1000 + 42
        local_budget = max(10, (deadline - time.time()) - 10)
        
        classes, counts, used_moduli = greedy_init_randomized(seed, init_budget)
        
        if not classes:
            continue
        
        best_classes, best_covered = local_search(classes, counts, local_budget)
        
        if best_covered > best_overall_covered:
            best_overall_covered = best_covered
            best_overall_classes = best_classes
    
    if best_overall_classes is None:
        # Fallback: simple greedy
        counts = np.zeros(base, dtype=np.int32)
        classes = []
        for m in divisors:
            residues = np.arange(base) % m
            weights = (counts == 0).astype(np.int32)
            gains = np.bincount(residues, weights=weights, minlength=m)
            best_r = int(np.argmax(gains))
            classes.append([m, best_r])
            counts[best_r::m] += 1
        best_overall_classes = classes
    
    return best_overall_classes
# EVOLVE-BLOCK-END
