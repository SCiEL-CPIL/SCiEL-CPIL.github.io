"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _count_nonzero(coeffs):
    """Count nonzero entries in a coefficient list."""
    return sum(1 for c in coeffs if c != 0)

def _get_sparsity(pairs):
    """Compute the number of nonzero terms in the square of the polynomial defined by pairs.
    pairs: list of (exponent, coefficient) tuples."""
    if not pairs:
        return 0
    max_deg = max(e for e, c in pairs if c != 0)
    arr = np.zeros(2 * max_deg + 1, dtype=int)
    for e, c in pairs:
        if c != 0:
            arr[e] += c
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_sparsity = float('inf')
    best_result_pairs = None
    
    # Candidate exponent sets
    # 1. Arithmetic Progression (0..11)
    exps_ap = list(range(12))
    # 2. Arithmetic Progression with step 2 (0, 2, ..., 22)
    exps_ap2 = [2*i for i in range(12)]
    # 3. Clumped blocks
    exps_clump = [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32]
    
    candidate_exps = [exps_ap, exps_ap2, exps_clump]
    
    # Generate random exponent sets with high collision potential
    # We use a small range to ensure many sum collisions
    random_exps = []
    for _ in range(2000):
        base = random.randint(0, 20)
        exps = set()
        while len(exps) < 12:
            # Pick a random offset in a small range
            e = base + random.randint(0, 25)
            if e < 40:
                exps.add(e)
        random_exps.append(sorted(exps))
    
    candidate_exps.extend(random_exps)
    
    # Step 1: For each exponent set, perform a local search on coefficients
    coeff_choices = [-2, -1, 1, 2]
    
    for exps in candidate_exps:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random coefficients
        current_coeffs = [random.choice(coeff_choices) for _ in range(12)]
        current_pairs = list(zip(exps, current_coeffs))
        current_sparsity = _get_sparsity(current_pairs)
        
        # Local search
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            # Try changing each coefficient
            for i in range(12):
                old_c = current_coeffs[i]
                for new_c in coeff_choices:
                    if new_c == old_c:
                        continue
                    new_coeffs = current_coeffs[:]
                    new_coeffs[i] = new_c
                    new_pairs = list(zip(exps, new_coeffs))
                    next_sparsity = _get_sparsity(new_pairs)
                    
                    if next_sparsity < current_sparsity:
                        current_coeffs = new_coeffs
                        current_sparsity = next_sparsity
                        improved = True
                        if current_sparsity < best_sparsity:
                            best_sparsity = current_sparsity
                            best_result_pairs = list(zip(exps, current_coeffs))
                        break # Restart loop as state changed
                    
        # Also try all 1s and all -1s as specific cases
        for test_coeffs in [[1]*12, [-1]*12]:
            test_pairs = list(zip(exps, test_coeffs))
            test_sparsity = _get_sparsity(test_pairs)
            if test_sparsity < best_sparsity:
                best_sparsity = test_sparsity
                best_result_pairs = test_pairs[:]
                
    if best_result_pairs is None:
        # Fallback
        best_result_pairs = list(zip(range(12), [1]*12))
        best_sparsity = _get_sparsity(best_result_pairs)

    # Convert best_result_pairs to coefficient list
    max_deg = max(e for e, c in best_result_pairs)
    result = [0] * (max_deg + 1)
    for e, c in best_result_pairs:
        result[e] = c
        
    # Ensure length <= 41
    if len(result) > 41:
        result = result[:41]
        
    # Ensure nonzero final coefficient
    if result[-1] == 0:
        last_nonzero = -1
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                last_nonzero = i
                break
        if last_nonzero != -1:
            result = result[:last_nonzero + 1]
        if result[-1] == 0:
            result[-1] = 1
            
    # Ensure exactly 12 nonzero terms
    nonzero_count = sum(1 for c in result if c != 0)
    if nonzero_count != 12:
        # Reconstruct from pairs to be sure
        max_deg = max(e for e, c in best_result_pairs)
        result = [0] * (max_deg + 1)
        for e, c in best_result_pairs:
            result[e] = c
        if len(result) > 41:
             result = result[:41]
        if result[-1] == 0:
             last_nonzero = -1
             for i in range(len(result) - 1, -1, -1):
                 if result[i] != 0:
                     last_nonzero = i
                     break
             if last_nonzero != -1:
                 result = result[:last_nonzero + 1]
             if result[-1] == 0:
                 result[-1] = 1
                 
    return result

# EVOLVE-BLOCK-END
