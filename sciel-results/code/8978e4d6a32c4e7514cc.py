import itertools
import random

def count_nonzero(lst):
    return sum(1 for x in lst if x != 0)

def poly_square(coeffs):
    """Compute exact square of polynomial given as coeff list (ascending degree)."""
    n = len(coeffs)
    if n == 0:
        return []
    result = [0] * (2*n - 1)
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    for k, (i, ci) in enumerate(nz):
        for l in range(k, len(nz)):
            j, cj = nz[l]
            val = ci * cj
            if i == j:
                result[i+j] += val
            else:
                result[i+j] += 2 * val
    return result

def valid_candidate(coeffs):
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    if count_nonzero(coeffs) != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def construct():
    best_score = float('inf')
    best_cand = None
    
    # Generate candidate exponent sets
    # Each set is a list of 12 distinct integers in [0, 40]
    exp_sets = []
    
    # 1. Consecutive
    exp_sets.append(list(range(12)))
    
    # 2. Two blocks
    exp_sets.append([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23])
    
    # 3. Arithmetic progression step 2
    exp_sets.append([2*i for i in range(12)])
    
    # 4. Arithmetic progression step 3
    exp_sets.append([3*i for i in range(12)])
    
    # 5. Dense cluster then sparse
    exp_sets.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30])
    
    # 6. Deterministic random sets (fixed seed)
    random.seed(123)
    for _ in range(50):
        exps = sorted(random.sample(range(41), 12))
        exp_sets.append(exps)
    
    # For each exponent set, try all 2^12 sign combinations
    # Coefficients are +/- 1
    for exps in exp_sets:
        # Sort exponents to ensure consistent indexing
        exps = sorted(exps)
        
        # Iterate over all sign combinations
        # Using bit manipulation for speed
        for mask in range(1 << 12):
            coeffs = [0] * 41
            for i in range(12):
                # If bit i is set, coeff is 1, else -1
                if (mask >> i) & 1:
                    coeffs[exps[i]] = 1
                else:
                    coeffs[exps[i]] = -1
            
            # Trim trailing zeros (though exps are within 0-40, len is 41)
            # Ensure last coeff is nonzero (it is, since exps[i] < 41)
            # But we need to return a list of length <= 41.
            # The problem says "Return coefficient list in ascending degree, length<=41".
            # It doesn't say the list must be padded to 41. It can be shorter.
            # But "nonzero final coefficient" means the highest degree term is nonzero.
            
            # Find max exponent used
            max_exp = max(exps)
            cand = coeffs[:max_exp+1]
            
            if not valid_candidate(cand):
                continue
                
            sq = poly_square(cand)
            score = count_nonzero(sq)
            
            if score < best_score:
                best_score = score
                best_cand = cand[:]
                
    # If best_cand is None (should not happen), fallback
    if best_cand is None:
        best_cand = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
        
    return best_cand
