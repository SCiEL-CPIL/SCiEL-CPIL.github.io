"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    base = 51975
    # Generate all divisors of base that are > 1 and odd (all divisors of 51975 are odd)
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Precompute coverage sets for each divisor (as sorted lists of residues)
    # For efficiency, we'll work with bitmasks or boolean arrays
    # Since base is 51975, a bytearray is fine
    
    # We'll use a randomized greedy with look-ahead
    # Key idea: at each step, consider multiple candidate moduli and pick the one
    # that maximizes expected marginal gain, with some randomization to escape local optima
    
    # Precompute residue sets for each divisor
    residue_sets = {}
    for m in divisors:
        # For each modulus m, the residues are 0, 1, ..., m-1
        # We need to pick one residue r for modulus m
        # The set covered by (m, r) is {r, r+m, r+2m, ...} mod base
        residue_sets[m] = list(range(m))
    
    # We'll use a multi-start approach with randomized greedy
    best_score = base  # worst case: nothing covered
    best_classes = None
    
    num_trials = 20  # multiple randomized trials
    
    for trial in range(num_trials):
        random.seed(trial * 1000 + 42)
        
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        remaining_moduli = set(divisors)
        
        # Shuffle moduli to introduce randomness in ordering
        moduli_list = list(remaining_moduli)
        random.shuffle(moduli_list)
        
        # Adaptive look-ahead: at each step, evaluate a subset of candidates
        # and pick the best marginal gain with some noise
        
        while remaining_moduli:
            # Consider a subset of candidates for look-ahead
            # Use adaptive size: more candidates when few are left
            if len(remaining_moduli) <= 10:
                candidates = list(remaining_moduli)
            else:
                # Sample a fraction of candidates
                sample_size = max(5, len(remaining_moduli) // 5)
                candidates = random.sample(list(remaining_moduli), sample_size)
            
            best_m = None
            best_r = None
            best_gain = -1
            
            for m in candidates:
                # For this modulus, find the residue that covers the most uncovered residues
                # Compute marginal gain for each residue
                uncovered_count = 0
                best_r_for_m = 0
                best_gain_for_m = 0
                
                # Efficiently compute coverage for each residue r in [0, m)
                # The set covered by (m, r) is {r + k*m for k >= 0 such that r + k*m < base}
                # We can compute the number of uncovered residues in this set
                
                # Use numpy for speed
                # residues = np.arange(r, base, m)
                # gain = (1 - covered[residues]).sum()
                
                # Vectorized approach: compute for all r at once
                # For modulus m, create an array of length m where entry r is the gain
                # This is expensive for large m, so we sample residues for large m
                
                if m <= 100:
                    # Compute gain for all residues
                    gains = np.zeros(m, dtype=np.int32)
                    for r in range(m):
                        residues = np.arange(r, base, m)
                        gains[r] = (1 - covered[residues]).sum()
                    best_r_for_m = int(np.argmax(gains))
                    best_gain_for_m = int(gains[best_r_for_m])
                else:
                    # Sample residues for large moduli
                    num_samples = min(m, 50)
                    if num_samples < m:
                        sample_rs = random.sample(range(m), num_samples)
                    else:
                        sample_rs = list(range(m))
                    
                    best_r_for_m = 0
                    best_gain_for_m = 0
                    for r in sample_rs:
                        residues = np.arange(r, base, m)
                        gain = (1 - covered[residues]).sum()
                        if gain > best_gain_for_m:
                            best_gain_for_m = gain
                            best_r_for_m = r
                
                if best_gain_for_m > best_gain:
                    best_gain = best_gain_for_m
                    best_m = m
                    best_r = best_r_for_m
            
            # If no positive gain, break
            if best_m is None or best_gain <= 0:
                break
            
            # Add the class
            classes.append([best_m, best_r])
            remaining_moduli.discard(best_m)
            
            # Update covered array
            residues = np.arange(best_r, base, best_m)
            covered[residues] = 1
        
        # Compute final score
        uncovered = (1 - covered).sum()
        if uncovered < best_score:
            best_score = int(uncovered)
            best_classes = classes
    
    # Fallback: if no classes found, return empty (shouldn't happen)
    if best_classes is None:
        best_classes = []
    
    return best_classes

# EVOLVE-BLOCK-END
