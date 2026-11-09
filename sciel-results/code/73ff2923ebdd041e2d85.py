"""Strategy: Global Randomized Support and Coefficient Search."""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # We need to return a list of coefficients in ascending degree.
    # Max length 41 (degree 0 to 40).
    # Exactly 12 nonzero terms.
    # Coefficients in [-64, 64].
    # Final coefficient (highest degree) must be nonzero.
    # Minimize nonzero terms in the square.
    
    # The square of a degree-40 polynomial has degree 80.
    # We don't need to represent the whole square, just count nonzeros.
    # However, numpy convolution is efficient enough for small arrays.
    
    def count_nonzero_sq(coeffs_list):
        # coeffs_list is a list of length <= 41
        # Convert to numpy array for fast convolution
        arr = np.array(coeffs_list, dtype=np.int64)
        # Square is convolution of arr with itself
        sq = np.convolve(arr, arr)
        return np.count_nonzero(sq)
    
    best_score = float('inf')
    best_poly = None
    
    # Helper to generate a valid random polynomial
    def generate_random_poly():
        # We need exactly 12 nonzero terms.
        # Indices must be within 0..40.
        # The highest index (degree) must be one of the nonzero indices.
        
        # Choose 12 distinct indices from 0..40
        indices = random.sample(range(41), 12)
        # Ensure the max index is the highest degree? 
        # Actually, the "final coefficient" in the list returned must be nonzero.
        # If we return a list of length L, the last element coeffs[L-1] must be nonzero.
        # So we must ensure that the maximum index chosen is the length of the list - 1.
        # Or we can pad the list to 41 length? 
        # "Return coefficient list in ascending degree, length<=41... nonzero final coefficient"
        # If I return a list of length 12, the last element is index 11.
        # If my polynomial is 1 + x^10, and I return [1, 0, ..., 0, 1], length 11, last is 1.
        # It is safer to return a list of length 41 always, or trim trailing zeros?
        # "length<=41, nonzero final coefficient" implies if I return length 5, coeffs[4] != 0.
        # To maximize flexibility for cancellation, I should probably fix the "window" or let it vary.
        # However, the square's sparsity depends on the *relative* distances between terms.
        # Shifting the whole polynomial by x^k doesn't change the number of nonzeros in the square.
        # So we can fix one term at index 0 to reduce symmetry? 
        # But the constraint is on the *list* returned.
        # If I shift, the list length changes or trailing zeros appear.
        # If I have terms at 0 and 40, list length 41.
        # If I have terms at 0 and 10, list length 11.
        # The score is just the count of nonzeros in the square.
        
        # Let's just pick 12 indices, sort them.
        indices.sort()
        max_idx = indices[-1]
        
        # The list length will be max_idx + 1
        length = max_idx + 1
        coeffs = [0] * length
        
        # Assign random coefficients to the chosen indices
        for i in indices:
            # Coefficients in [-64, 64], non-zero
            c = random.randint(1, 64)
            if random.random() < 0.5:
                c = -c
            coeffs[i] = c
            
        return coeffs

    # Initial guess: Baseline [1]*12
    baseline = [1] * 12
    best_score = count_nonzero_sq(baseline)
    best_poly = baseline
    
    # Search loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Strategy: 
        # 1. Random restart: Generate a completely new random polynomial.
        # 2. Local perturbation: Take the current best, change one coefficient or one index.
        
        if random.random() < 0.3 or best_poly is None:
            # Random Restart
            candidate = generate_random_poly()
        else:
            # Local Perturbation of best_poly
            candidate = list(best_poly)
            
            # Pick a random mutation type
            mut_type = random.randint(0, 2)
            
            if mut_type == 0:
                # Change a coefficient
                # Find nonzero indices
                nz_indices = [i for i, c in enumerate(candidate) if c != 0]
                if nz_indices:
                    idx = random.choice(nz_indices)
                    new_c = random.randint(1, 64)
                    if random.random() < 0.5:
                        new_c = -new_c
                    candidate[idx] = new_c
                    
            elif mut_type == 1:
                # Move a term
                nz_indices = [i for i, c in enumerate(candidate) if c != 0]
                if len(nz_indices) >= 2:
                    # Pick a term to move (not the last one, to keep length stable? 
                    # Actually, moving might change length if we move the last term to a lower index,
                    # or if we move a term to a higher index.
                    # To keep it simple and valid, let's just pick a nonzero index and a new index.
                    # We must ensure exactly 12 nonzeros remain.
                    # If we move idx_i to idx_j:
                    # 1. Set candidate[idx_i] = 0
                    # 2. If candidate[idx_j] was 0, set to candidate[idx_i]'s value.
                    # 3. If candidate[idx_j] was non-zero, we have a conflict (would become 11 or 13 terms).
                    # So we should only move to a zero index.
                    zero_indices = [i for i, c in enumerate(candidate) if c == 0]
                    if zero_indices:
                        src = random.choice(nz_indices)
                        dst = random.choice(zero_indices)
                        val = candidate[src]
                        candidate[src] = 0
                        candidate[dst] = val
                        
            elif mut_type == 2:
                # Swap coefficients between two nonzero terms
                nz_indices = [i for i, c in enumerate(candidate) if c != 0]
                if len(nz_indices) >= 2:
                    i1, i2 = random.sample(nz_indices, 2)
                    candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            
            # Ensure final coefficient is nonzero
            if candidate and candidate[-1] == 0:
                # This can happen if we moved the last term away.
                # We need to fix it by moving a nonzero term to the last position?
                # Or just discard this candidate if it violates constraints?
                # Let's check constraints:
                # 1. Length <= 41
                # 2. Nonzero final coefficient
                # 3. Exactly 12 nonzero terms
                # 4. Entries in [-64, 64]
                
                # If candidate[-1] is 0, it's invalid.
                # Let's just skip evaluation for this candidate.
                continue
        
        # Validate candidate
        if len(candidate) > 41:
            continue
        if not candidate or candidate[-1] == 0:
            continue
        if sum(1 for c in candidate if c != 0) != 12:
            continue
        if any(abs(c) > 64 for c in candidate):
            continue
            
        # Evaluate
        score = count_nonzero_sq(candidate)
        
        if score < best_score:
            best_score = score
            best_poly = candidate
            
            # If we found a very good score, we might want to do more local search around it
            # But the loop structure handles this via the 70% local perturbation bias.
            
    return best_poly
