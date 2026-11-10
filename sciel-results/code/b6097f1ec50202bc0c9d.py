import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def evaluate(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if int(np.count_nonzero(coeffs)) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    def try_poly(poly_list):
        nonlocal best_score, best_poly
        arr = np.array(poly_list, dtype=np.int64)
        if not validate(arr):
            return
        score = evaluate(arr)
        if score < best_score:
            best_score = score
            best_poly = arr.copy()

    random.seed(12345)

    # Strategy: Use structured exponent patterns that create systematic cross-term overlaps,
    # then optimize signs and magnitudes to zero out as many square coefficients as possible.
    
    # Key insight: If P(x) has terms at exponents that form arithmetic progressions or other
    # structured patterns, the convolution will have predictable overlap patterns.
    # We can then solve for signs/magnitudes that cancel those overlaps.
    
    # Phase 1: Try known good structured patterns with optimized signs
    structured_patterns = [
        # Consecutive blocks
        [0,1,2,3,4,5,6,7,8,9,10,11],
        # Two blocks
        [0,1,2,3,4,5,10,11,12,13,14,15],
        [0,1,2,3,4,5,8,9,10,11,12,13],
        # Three blocks
        [0,1,2,3,6,7,8,9,12,13,14,15],
        # Arithmetic progression with gap
        [0,2,4,6,8,10,12,14,16,18,20,22],
        # Mixed: dense core + sparse tail
        [0,1,2,3,4,5,6,7,8,15,20,25],
        # Symmetric
        [0,1,2,3,4,5,10,15,16,17,18,19],
        # Power-of-2 like (but bounded)
        [0,1,2,4,8,16,17,18,19,20,21,22],
        # Clusters
        [0,1,2,3,4,5,10,11,12,13,14,15],
        [0,1,2,3,4,5,6,7,12,13,14,15],
    ]
    
    for exps in structured_patterns:
        if time.time() > deadline - 10:
            break
        # Try all 2^12 sign combinations for this exponent set
        for mask in range(4096):
            if time.time() > deadline - 10:
                break
            coeffs = [0] * 41
            for i, e in enumerate(exps):
                if 0 <= e < 41:
                    coeffs[e] = 1 if (mask >> i) & 1 else -1
            # Trim
            while len(coeffs) > 0 and coeffs[-1] == 0:
                coeffs.pop()
            if len(coeffs) > 0:
                try_poly(coeffs)
    
    # Phase 2: Local search on exponents and signs for the best found so far
    if best_poly is None:
        best_poly = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0], dtype=np.int64)
        best_score = evaluate(best_poly)
    
    # Extract current best parameters
    best_arr = np.array(best_poly, dtype=np.int64)
    curr_exps = [i for i in range(len(best_arr)) if best_arr[i] != 0]
    curr_signs = [1 if best_arr[i] > 0 else -1 for i in curr_exps]
    curr_vals = [abs(int(best_arr[i])) for i in curr_exps]
    
    curr_score = best_score
    no_improve = 0
    max_no_improve = 20000
    
    def get_poly_from_params(exps, signs, vals):
        poly = [0] * 41
        for e, s, v in zip(exps, signs, vals):
            if 0 <= e < 41:
                poly[e] = s * v
        while len(poly) > 0 and poly[-1] == 0:
            poly.pop()
        if len(poly) == 0:
            return None
        if len([x for x in poly if x != 0]) != 12:
            return None
        return poly
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        mod_type = random.choice(['exp', 'sign', 'val', 'exp_sign', 'swap_sign'])
        
        new_exps = curr_exps.copy()
        new_signs = curr_signs.copy()
        new_vals = curr_vals.copy()
        
        idx = random.randint(0, 11)
        
        if mod_type == 'exp':
            old_e = new_exps[idx]
            if random.random() < 0.6:
                new_e = max(0, min(40, old_e + random.choice([-3,-2,-1,1,2,3])))
            else:
                new_e = random.randint(0, 40)
            if new_e in new_exps and new_exps[idx] != new_e:
                no_improve += 1
                continue
            new_exps[idx] = new_e
            new_exps.sort()
            
        elif mod_type == 'sign':
            new_signs[idx] *= -1
            
        elif mod_type == 'val':
            candidates = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 40, 50, 64]
            new_vals[idx] = random.choice(candidates)
            
        elif mod_type == 'exp_sign':
            old_e = new_exps[idx]
            new_e = max(0, min(40, old_e + random.choice([-2,-1,1,2])))
            if new_e in new_exps and new_exps[idx] != new_e:
                no_improve += 1
                continue
            new_exps[idx] = new_e
            new_exps.sort()
            new_signs[idx] *= -1
            
        elif mod_type == 'swap_sign':
            j = random.randint(0, 11)
            new_signs[idx], new_signs[j] = new_signs[j], new_signs[idx]

        new_poly = get_poly_from_params(new_exps, new_signs, new_vals)
        if new_poly is None:
            no_improve += 1
            continue
            
        new_score = evaluate(new_poly)
        
        if new_score < curr_score:
            curr_score = new_score
            no_improve = 0
            curr_exps = new_exps
            curr_signs = new_signs
            curr_vals = new_vals
            try_poly(new_poly)
            if best_score <= 14:
                break
        else:
            no_improve += 1

    if best_poly is None:
        best_poly = np.array([1] * 12, dtype=np.int64)

    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
