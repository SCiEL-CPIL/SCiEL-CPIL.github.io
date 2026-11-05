"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time

def construct():
    base = 51975
    # Generate all divisors of base that are > 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort divisors (order doesn't matter much as we evaluate all)
    divisors.sort()
    
    best_score = base
    best_classes = []
    
    # Multi-start with randomized greedy
    start_time = time.time()
    max_time = 110.0
    
    trial = 0
    while time.time() - start_time < max_time:
        trial += 1
        random.seed(trial)
        
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        remaining_moduli = set(divisors)
        
        # Greedy loop
        while remaining_moduli:
            # Check time
            if time.time() - start_time > max_time:
                break
            
            # Evaluate ALL remaining moduli
            # Since there are only ~47 divisors, this is fast.
            gains_list = []
            
            for m in remaining_moduli:
                # Find best residue r for this m
                # Gain for residue r is sum(1 - covered[r::m])
                
                if m <= 200:
                    # Vectorized calculation for small m
                    try:
                        cov_reshaped = covered.reshape(m, base // m)
                        gains = (1 - cov_reshaped).sum(axis=1)
                        best_r_for_m = int(np.argmax(gains))
                        best_gain_for_m = int(gains[best_r_for_m])
                    except:
                        best_r_for_m = 0
                        best_gain_for_m = 0
                        for r in range(m):
                            g = (1 - covered[r::m]).sum()
                            if g > best_gain_for_m:
                                best_gain_for_m = g
                                best_r_for_m = r
                else:
                    # Large m: sample residues
                    num_samples = min(m, 20)
                    sample_rs = random.sample(range(m), num_samples)
                    best_r_for_m = sample_rs[0]
                    best_gain_for_m = -1
                    for r in sample_rs:
                        g = (1 - covered[r::m]).sum()
                        if g > best_gain_for_m:
                            best_gain_for_m = g
                            best_r_for_m = r
                
                if best_gain_for_m > 0:
                    gains_list.append((best_gain_for_m, m, best_r_for_m))
            
            if not gains_list:
                break
            
            # Probabilistic selection based on marginal gain
            # Sort by gain descending
            gains_list.sort(key=lambda x: x[0], reverse=True)
            max_g = gains_list[0][0]
            
            # Temperature T
            # Scale T with the maximum gain to make it adaptive
            T = max_g * 0.2 if max_g > 0 else 1.0
            
            # Compute probabilities using Boltzmann distribution
            probs = []
            for g, m, r in gains_list:
                # Avoid underflow/overflow by subtracting max_g
                prob = np.exp((g - max_g) / T)
                probs.append(prob)
            
            total_prob = sum(probs)
            if total_prob == 0:
                break
                
            # Normalize
            probs = [p / total_prob for p in probs]
            
            # Select candidate
            rand_val = random.random()
            cumulative = 0
            selected = gains_list[0]
            for i, p in enumerate(probs):
                cumulative += p
                if rand_val <= cumulative:
                    selected = gains_list[i]
                    break
            
            best_gain, best_m, best_r = selected
            
            classes.append([best_m, best_r])
            remaining_moduli.discard(best_m)
            
            # Update covered
            covered[best_r::best_m] = 1
        
        # Compute score
        uncovered = (1 - covered).sum()
        if uncovered < best_score:
            best_score = int(uncovered)
            best_classes = classes
    
    return best_classes

# EVOLVE-BLOCK-END
