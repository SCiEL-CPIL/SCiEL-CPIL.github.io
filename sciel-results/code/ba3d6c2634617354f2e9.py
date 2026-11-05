"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import math

def construct():
    base = 51975
    
    # Generate all divisors of base that are > 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort divisors to have a consistent order
    divisors.sort()
    
    best_score = base
    best_classes = []
    
    # Increase trials to explore more of the search space
    # The heuristic is now adaptive (softmax with cooling)
    num_trials = 500 
    
    for trial in range(num_trials):
        random.seed(trial * 7919 + 13)
        
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        remaining_moduli = set(divisors)
        
        # Adaptive temperature for softmax selection
        # Start high to explore, decrease to exploit
        # Total steps is len(divisors)
        total_steps = len(divisors)
        current_step = 0
        
        while remaining_moduli:
            # Calculate marginal gains for all remaining moduli
            uncovered_view = 1 - covered
            candidates = []
            
            for m in remaining_moduli:
                try:
                    reshaped = uncovered_view.reshape(-1, m)
                    gains = reshaped.sum(axis=0)
                    best_r = int(np.argmax(gains))
                    max_gain = int(gains[best_r])
                    
                    if max_gain > 0:
                        candidates.append((m, best_r, max_gain))
                except ValueError:
                    continue
            
            if not candidates:
                break
            
            # Sort candidates by gain descending to facilitate softmax and top-k
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Adaptive temperature
            # T starts at 1.0 and decays exponentially
            # This encourages exploration early and exploitation later
            progress = current_step / max(1, total_steps)
            temperature = 1.0 * (1.0 - progress) + 0.1
            
            # Calculate softmax probabilities
            # To avoid overflow, subtract max gain
            max_gain_val = candidates[0][2]
            gains_array = np.array([c[2] for c in candidates], dtype=np.float64)
            # Normalize gains to be around 0-1 for stable softmax
            # If max_gain is large, divide by max_gain
            if max_gain_val > 0:
                normalized_gains = gains_array / max_gain_val
            else:
                normalized_gains = gains_array
            
            exponents = np.exp(normalized_gains / temperature)
            sum_exponents = np.sum(exponents)
            probabilities = exponents / sum_exponents
            
            # Sample a candidate
            idx = np.random.choice(len(candidates), p=probabilities)
            chosen = candidates[idx]
            
            m, r, gain = chosen
            classes.append([m, r])
            remaining_moduli.discard(m)
            current_step += 1
            
            # Update covered array
            covered[r::m] = 1
        
        # Calculate final score
        uncovered = (1 - covered).sum()
        if uncovered < best_score:
            best_score = int(uncovered)
            best_classes = classes
    
    return best_classes

# EVOLVE-BLOCK-END
