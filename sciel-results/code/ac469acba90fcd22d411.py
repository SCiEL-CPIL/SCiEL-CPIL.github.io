"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin
    
    N = 30
    LCM_VAL = 2329089562800
    L_over_i = [LCM_VAL // i for i in range(1, N + 1)]
    
    half1 = list(range(15))
    half2 = list(range(15, 30))
    
    def enumerate_with_coeffs(indices):
        """Return dict: sum -> coefficients tuple"""
        n = len(indices)
        # Start: {0: (0,0,...,0)}
        current = {0: tuple([0] * n)}
        
        for pos in range(n):
            idx = indices[pos]
            step = L_over_i[idx]
            new_current = {}
            for s, coeffs in current.items():
                for c in (-1, 0, 1):
                    new_s = s + c * step
                    new_coeffs = coeffs[:pos] + (c,) + coeffs[pos+1:]
                    # Only keep if not already seen (first occurrence)
                    if new_s not in new_current:
                        new_current[new_s] = new_coeffs
            current = new_current
            if time.time() > deadline:
                break
        
        return current
    
    # Enumerate first half
    dict1 = enumerate_with_coeffs(half1)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Enumerate second half
    dict2 = enumerate_with_coeffs(half2)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Get sorted keys from dict2
    keys2 = sorted(dict2.keys())
    
    best_abs = float('inf')
    best_S1 = None
    best_S2 = None
    
    for S1 in dict1.keys():
        target = -S1
        pos = bisect.bisect_left(keys2, target)
        candidates = []
        if pos < len(keys2):
            candidates.append(keys2[pos])
        if pos > 0:
            candidates.append(keys2[pos - 1])
        
        for S2 in candidates:
            total = S1 + S2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_abs:
                    best_abs = abs_total
                    best_S1 = S1
                    best_S2 = S2
    
    if best_S1 is None:
        return [0] * 28 + [1, -1]
    
    coeffs1 = dict1[best_S1]
    coeffs2 = dict2[best_S2]
    
    result = list(coeffs1) + list(coeffs2)
    
    # Verify
    total = sum(result[i] * L_over_i[i] for i in range(N))
    if total == 0:
        return [0] * 28 + [1, -1]
    
    return result
# EVOLVE-BLOCK-END
