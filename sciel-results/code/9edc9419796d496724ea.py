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
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Precompute for each divisor m the list of positions for each residue
    # pos_table[m_idx][r] = array of positions x where x ≡ r (mod m)
    # This is expensive in memory, so we'll compute on the fly but cache small ones
    
    # Randomized greedy with adaptive look-ahead
    # Model as maximum coverage: universe = base residues, each (m,r) is a set
    # Use probabilistic selection based on marginal gain
    
    def compute_gains_for_modulus(m, counts):
        """Compute gain for each residue r of modulus m."""
        gains = np.zeros(m, dtype=np.int32)
        for r in range(m):
            gains[r] = np.count_nonzero(counts[r::m] == 0)
        return gains
    
    def local_search_polish(classes, counts, deadline):
        """Local search to improve residue assignments."""
        improved = True
        max_passes = 50
        pass_count = 0
        
        while improved and pass_count < max_passes and time.time() < deadline - 10:
            improved = False
            pass_count += 1
            
            indices = list(range(len(classes)))
            np.random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5:
                    return classes, counts
                
                m, old_r = classes[idx]
                
                # Temporarily remove this class
                counts[old_r::m] -= 1
                
                # Compute loss: positions in old_r class now uncovered
                loss_old = np.count_nonzero(counts[old_r::m] == 0)
                
                # Find best new residue
                best_gain = loss_old
                best_r = old_r
                
                # For efficiency, compute gains for all residues
                # For small m, try all; for large m, sample
                if m <= 5000:
                    for r in range(m):
                        if r == old_r:
                            continue
                        gain = np.count_nonzero(counts[r::m] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                else:
                    # Sample a subset of residues for very large m
                    n_sample = min(m, 200)
                    if m <= 200:
                        sample = list(range(m))
                    else:
                        sample = np.random.choice(m, size=n_sample, replace=False)
                        sample = [int(x) for x in sample]
                    for r in sample:
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
                else:
                    counts[old_r::m] += 1
        
        return classes, counts
    
    # Multiple restarts with different random seeds for diversity
    best_classes = None
    best_uncovered = base + 1
    
    n_restarts = 3  # Number of randomized restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(restart * 12345 + 67890)
        
        counts = np.zeros(base, dtype=np.int32)
        classes = []
        used_moduli = set()
        remaining_moduli = list(divisors)
        
        # Randomized greedy with adaptive look-ahead
        # At each step, compute marginal gain for each available modulus
        # Use softmax-like probabilistic selection based on gain
        # Adaptive: as we use more moduli, reduce temperature to be more greedy
        
        n_moduli = len(remaining_moduli)
        for step in range(n_moduli):
            if time.time() > deadline - 15:
                break
            
            # Compute gains for each remaining modulus
            # For each modulus, the best residue's gain
            moduli_to_consider = [m for m in remaining_moduli if m not in used_moduli]
            
            if not moduli_to_consider:
                break
            
            # Compute best gain for each modulus
            gains = []
            for m in moduli_to_consider:
                # Find best residue for this modulus
                best_r = 0
                best_gain = -1
                for r in range(m):
                    gain = np.count_nonzero(counts[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                gains.append((m, best_r, best_gain))
            
            if not gains:
                break
            
            # Adaptive temperature: start high (exploratory), decrease over time
            # temperature = max_gain * (1 - step/n_moduli) * 0.5 + 1
            max_gain = max(g[2] for g in gains)
            if max_gain <= 0:
                break
            
            # Use softmax weighting based on gain
            # weight_i = exp(gain_i / temperature)
            temperature = max_gain * 0.3 + 1.0  # Adaptive temperature
            
            weights = []
            for m, r, g in gains:
                w = np.exp(g / temperature)
                weights.append(w)
            
            total_w = sum(weights)
            if total_w == 0:
                break
            
            # Probabilistic selection
            probs = [w / total_w for w in weights]
            idx = np.random.choice(len(gains), p=probs)
            
            m, best_r, _ = gains[idx]
            classes.append([m, best_r])
            counts[best_r::m] += 1
            used_moduli.add(m)
            remaining_moduli.remove(m)
        
        # Local search polish
        classes, counts = local_search_polish(classes, counts, deadline)
        
        # Evaluate
        uncovered = np.count_nonzero(counts == 0)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [list(c) for c in classes]
        
        if best_uncovered == 0:
            break
    
    if best_classes is None:
        # Fallback: return empty
        return []
    
    return best_classes
# EVOLVE-BLOCK-END
