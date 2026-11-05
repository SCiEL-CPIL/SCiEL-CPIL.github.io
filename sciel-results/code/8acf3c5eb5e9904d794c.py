import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Helper to evaluate a polynomial defined by degrees and coefficients
    def get_score(degs, coeffs):
        max_deg = max(degs)
        if max_deg > 40:
            return 10000  # Penalty for invalid degree
        # Ensure distinct degrees
        if len(set(degs)) != 12:
            return 10000
        
        vec = np.zeros(max_deg + 1, dtype=np.int64)
        for d, c in zip(degs, coeffs):
            vec[d] = c
        
        # Exact integer convolution
        sq = np.convolve(vec, vec)
        return np.count_nonzero(sq)
    
    # Initial candidates
    # 1. All ones
    # 2. Alternating ones
    # 3. Random
    
    candidates = []
    
    # Candidate 1: [1]*12
    degs1 = list(range(12))
    coeffs1 = [1] * 12
    candidates.append((degs1, coeffs1))
    
    # Candidate 2: [1, -1, 1, -1...]
    degs2 = list(range(12))
    coeffs2 = [1 if i % 2 == 0 else -1 for i in range(12)]
    candidates.append((degs2, coeffs2))
    
    # Candidate 3: Random
    degs3 = list(range(12))
    coeffs3 = [np.random.randint(-64, 65) for _ in range(12)]
    # Ensure non-zero
    for i in range(12):
        if coeffs3[i] == 0:
            coeffs3[i] = 1
    candidates.append((degs3, coeffs3))
    
    # Evaluate initial candidates
    best_degs = None
    best_coeffs = None
    best_score = 10000
    
    for degs, coeffs in candidates:
        score = get_score(degs, coeffs)
        if score < best_score:
            best_score = score
            best_degs = degs[:]
            best_coeffs = coeffs[:]
            
    current_degs = best_degs[:]
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Generate a mutation
        new_degs = current_degs[:]
        new_coeffs = current_coeffs[:]
        
        # 80% coeff change, 20% degree change
        if np.random.random() < 0.8:
            idx = np.random.randint(12)
            new_c = np.random.randint(-64, 65)
            if new_c == 0:
                new_c = 1
            new_coeffs[idx] = new_c
        else:
            idx = np.random.randint(12)
            # Try to shift degree
            delta = np.random.choice([-3, -2, -1, 1, 2, 3])
            new_d = new_degs[idx] + delta
            
            if 0 <= new_d <= 40:
                # Check collision with other degrees
                other_degs = [d for i, d in enumerate(new_degs) if i != idx]
                if new_d not in other_degs:
                    new_degs[idx] = new_d
                else:
                    # Collision, try random
                    new_d = np.random.randint(0, 41)
                    if new_d not in other_degs:
                        new_degs[idx] = new_d
                    else:
                        new_degs[idx] = current_degs[idx] # Revert
            
        new_score = get_score(new_degs, new_coeffs)
        
        if new_score < current_score:
            current_degs = new_degs
            current_coeffs = new_coeffs
            current_score = new_score
            no_improve_count = 0
            
            if current_score < best_score:
                best_score = current_score
                best_degs = current_degs[:]
                best_coeffs = current_coeffs[:]
        else:
            no_improve_count += 1
            
        # If stuck, do a larger perturbation
        if no_improve_count > 3000:
            # Randomize a few coeffs
            for _ in range(4):
                idx = np.random.randint(12)
                c = np.random.randint(-64, 65)
                if c == 0:
                    c = 1
                current_coeffs[idx] = c
            
            # Randomize a degree
            idx = np.random.randint(12)
            d = np.random.randint(0, 41)
            other_degs = [dd for i, dd in enumerate(current_degs) if i != idx]
            if d not in other_degs:
                current_degs[idx] = d
            else:
                current_degs[idx] = np.random.randint(0, 41)
                
            current_score = get_score(current_degs, current_coeffs)
            no_improve_count = 0

    # Construct final list
    max_deg = max(best_degs)
    result = [0] * (max_deg + 1)
    for d, c in zip(best_degs, best_coeffs):
        result[d] = c
        
    # Ensure constraints
    if len(result) > 41:
        # Fallback to seed if something went wrong with degrees
        return [1] * 12
        
    # Ensure final coefficient is nonzero (it should be by construction)
    if result[-1] == 0:
        # Trim trailing zeros
        while result and result[-1] == 0:
            result.pop()
        if not result:
            return [1] * 12
            
    return result
