"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_sparsity(pairs):
    """Compute number of nonzero terms in square of polynomial defined by (exp, coeff) pairs."""
    if not pairs:
        return 0
    max_deg = max(e for e, c in pairs if c != 0)
    if max_deg < 0:
        return 0
    arr = np.zeros(2 * max_deg + 1, dtype=np.int64)
    for e, c in pairs:
        if c != 0:
            arr[e] = c
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_sparsity = float('inf')
    best_pairs = None
    
    def try_pattern(exponents, coeff_pattern):
        """Try a specific exponent set with a coefficient pattern. Returns sparsity."""
        if len(exponents) != 12:
            return float('inf')
        pairs = list(zip(exponents, coeff_pattern))
        sp = _compute_sparsity(pairs)
        return sp
    
    def search_signs_for_exponents(exponents):
        """For given exponents, search over sign patterns (±1) to minimize sparsity."""
        best_sp = float('inf')
        best_signs = None
        
        # Try all 2^12 = 4096 sign patterns
        for mask in range(4096):
            signs = []
            for i in range(12):
                if (mask >> i) & 1:
                    signs.append(1)
                else:
                    signs.append(-1)
            sp = try_pattern(exponents, signs)
            if sp < best_sp:
                best_sp = sp
                best_signs = signs[:]
        
        return best_sp, best_signs
    
    # Generate structured exponent patterns
    candidate_exponents = []
    
    # Arithmetic progressions
    for start in range(0, 30):
        for step in range(1, 8):
            exps = [start + i * step for i in range(12)]
            if max(exps) < 41:
                candidate_exponents.append(tuple(exps))
    
    # Symmetric patterns
    for center in range(10, 35):
        for spread in range(1, 15):
            exps = []
            for i in range(6):
                e1 = center - spread * (i + 1)
                e2 = center + spread * (i + 1)
                if 0 <= e1 <= 40 and 0 <= e2 <= 40:
                    exps.append(e1)
                    exps.append(e2)
            if len(exps) == 12:
                exps = sorted(exps)
                candidate_exponents.append(tuple(exps))
    
    # Clustered patterns
    for base in range(0, 25):
        for gap in range(5, 20):
            exps1 = [base + i for i in range(6)]
            exps2 = [base + gap + i for i in range(6)]
            if max(exps2) < 41 and min(exps1) >= 0:
                candidate_exponents.append(tuple(sorted(exps1 + exps2)))
    
    # Power-of-2 patterns
    for shift in range(0, 5):
        exps = [shift + (1 << i) for i in range(12)]
        if max(exps) < 41:
            candidate_exponents.append(tuple(exps))
    
    # Deduplicate
    candidate_exponents = list(set(candidate_exponents))
    
    # Search over structured patterns
    for exps in candidate_exponents:
        if time.time() - start_time > time_limit:
            break
        
        sp, signs = search_signs_for_exponents(list(exps))
        
        if sp < best_sparsity:
            best_sparsity = sp
            best_pairs = list(zip(list(exps), signs))
    
    # Add random structured patterns
    random.seed(42)
    for _ in range(300):
        if time.time() - start_time > time_limit:
            break
        
        pattern_type = random.choice([0, 1, 2])
        
        if pattern_type == 0:
            start = random.randint(0, 25)
            step = random.randint(1, 5)
            exps = [start + i * step for i in range(12)]
        elif pattern_type == 1:
            center = random.randint(15, 30)
            offsets = sorted(random.sample(range(1, 20), 6))
            exps = []
            for off in offsets:
                e1 = center - off
                e2 = center + off
                if 0 <= e1 <= 40:
                    exps.append(e1)
                if 0 <= e2 <= 40:
                    exps.append(e2)
            if len(exps) < 12:
                continue
            exps = sorted(exps[:12])
        else:
            base = random.randint(0, 20)
            gap = random.randint(8, 15)
            exps1 = [base + random.randint(0, 3) for _ in range(6)]
            exps2 = [base + gap + random.randint(0, 3) for _ in range(6)]
            exps = sorted(exps1 + exps2)
        
        if len(exps) != 12 or min(exps) < 0 or max(exps) >= 41:
            continue
        if len(set(exps)) != 12:
            continue
        
        sp, signs = search_signs_for_exponents(exps)
        if sp < best_sparsity:
            best_sparsity = sp
            best_pairs = list(zip(exps, signs))
    
    # Fallback
    if best_pairs is None:
        exps = list(range(12))
        signs = [1 if i % 2 == 0 else -1 for i in range(12)]
        best_pairs = list(zip(exps, signs))
    
    # Convert to coefficient list
    max_deg = max(e for e, c in best_pairs)
    result = [0] * (max_deg + 1)
    for e, c in best_pairs:
        result[e] = c
    
    if len(result) > 41:
        result = result[:41]
    
    while len(result) > 0 and result[-1] == 0:
        result.pop()
    if len(result) == 0:
        result = [1]
    elif result[-1] == 0:
        result[-1] = 1
    
    nonzero_count = sum(1 for c in result if c != 0)
    if nonzero_count != 12:
        result = [1] * 12
    
    return result

# EVOLVE-BLOCK-END
