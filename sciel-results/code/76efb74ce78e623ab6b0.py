"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    base = 51975
    
    # Generate all divisors of base that are > 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort divisors to have a consistent order, though we will iterate over a set
    divisors.sort()
    
    best_score = base
    best_classes = []
    
    # Multi-start randomized greedy
    # We increase trials to explore more of the search space
    # Each trial is relatively fast due to vectorized gain calculation
    num_trials = 100 
    
    for trial in range(num_trials):
        random.seed(trial * 7919 + 13)
        
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        remaining_moduli = set(divisors)
        
        while remaining_moduli:
            # Calculate marginal gains for all remaining moduli
            # We want to find the best (m, r) pair
            # For each m, the best r is the one that maximizes the sum of uncovered residues in that class
            
            uncovered_view = 1 - covered
            candidates = []
            
            for m in remaining_moduli:
                # Reshape uncovered_view to (-1, m) to sum columns
                # This works because base is divisible by m
                try:
                    reshaped = uncovered_view.reshape(-1, m)
                    gains = reshaped.sum(axis=0)
                    best_r = int(np.argmax(gains))
                    max_gain = int(gains[best_r])
                    
                    if max_gain > 0:
                        candidates.append((m, best_r, max_gain))
                except ValueError:
                    # Should not happen for divisors, but safe fallback
                    continue
            
            if not candidates:
                break
            
            # Sort candidates by gain descending
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Probabilistic selection to escape local optima
            # With high probability, pick the best.
            # With lower probability, pick from the top few.
            if random.random() < 0.7:
                # Pick the best
                chosen = candidates[0]
            else:
                # Pick from top 5 (or all if fewer)
                k = min(5, len(candidates))
                # Use softmax-like weighting or just uniform over top k
                # Uniform over top k is a simple effective heuristic
                chosen = random.choice(candidates[:k])
            
            m, r, gain = chosen
            classes.append([m, r])
            remaining_moduli.discard(m)
            
            # Update covered array
            # Mark all residues congruent to r mod m as covered
            covered[r::m] = 1
        
        # Calculate final score
        uncovered = (1 - covered).sum()
        if uncovered < best_score:
            best_score = int(uncovered)
            best_classes = classes
    
    return best_classes

# EVOLVE-BLOCK-END
