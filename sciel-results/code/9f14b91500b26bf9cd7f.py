import time
import random
import numpy as np
import itertools

def _square_coeffs(c):
    """Return coefficient list of c*c (convolution), trimmed to remove trailing zeros."""
    c_arr = np.array(c, dtype=np.int64)
    result = np.convolve(c_arr, c_arr)
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _nnz(c):
    return sum(1 for x in c if x != 0)

def _is_valid(poly):
    if len(poly) > 41:
        return False
    if len(poly) == 0 or poly[-1] == 0:
        return False
    if sum(1 for x in poly if x != 0) != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _evaluate_poly(poly):
    if not _is_valid(poly):
        return float('inf')
    sq = _square_coeffs(poly)
    return _nnz(sq)

def _local_search(exps, coeffs, max_steps=5000, time_limit=5.0):
    """
    Local search on exponents and coefficients.
    exps: list of 12 exponents
    coeffs: list of 12 coefficients
    """
    start = time.time()
    current_exps = exps[:]
    current_coeffs = coeffs[:]
    
    # Build initial poly
    poly = [0] * (max(current_exps) + 1)
    for e, c in zip(current_exps, current_coeffs):
        if e < len(poly):
            poly[e] = c
    # Ensure length <= 41 and valid
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    if len(poly) > 41:
        return exps, coeffs, float('inf')
        
    current_score = _evaluate_poly(poly)
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        
        improved = False
        
        # 1. Try changing coefficients
        for i in range(12):
            old_c = current_coeffs[i]
            for delta in [-1, 1, -2, 2]:
                new_c = old_c + delta
                if new_c == 0 or abs(new_c) > 64:
                    continue
                new_coeffs = current_coeffs[:]
                new_coeffs[i] = new_c
                # Rebuild poly
                p = [0] * (max(current_exps) + 1)
                for e, c in zip(current_exps, new_coeffs):
                    if e < len(p):
                        p[e] = c
                while len(p) > 1 and p[-1] == 0:
                    p.pop()
                if len(p) > 41:
                    continue
                score = _evaluate_poly(p)
                if score < current_score:
                    current_score = score
                    current_coeffs = new_coeffs
                    poly = p
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 2. Try swapping exponents
        i, j = random.sample(range(12), 2)
        if current_exps[i] != current_exps[j]:
            new_exps = current_exps[:]
            new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
            p = [0] * (max(new_exps) + 1)
            for e, c in zip(new_exps, current_coeffs):
                if e < len(p):
                    p[e] = c
            while len(p) > 1 and p[-1] == 0:
                p.pop()
            if len(p) <= 41:
                score = _evaluate_poly(p)
                if score < current_score:
                    current_score = score
                    current_exps = new_exps
                    poly = p
                    improved = True
        
        if improved:
            continue

        # 3. Try changing one exponent
        i = random.randint(0, 11)
        old_e = current_exps[i]
        for delta in [-1, 1, -2, 2]:
            new_e = old_e + delta
            if new_e < 0 or new_e > 40:
                continue
            if new_e in current_exps and new_e != old_e:
                continue # Exponents must be distinct
            
            new_exps = current_exps[:]
            new_exps[i] = new_e
            p = [0] * (max(new_exps) + 1)
            for e, c in zip(new_exps, current_coeffs):
                if e < len(p):
                    p[e] = c
            while len(p) > 1 and p[-1] == 0:
                p.pop()
            if len(p) <= 41:
                score = _evaluate_poly(p)
                if score < current_score:
                    current_score = score
                    current_exps = new_exps
                    poly = p
                    improved = True
                    break
        
        if not improved:
            # Stagnation: try a larger jump in coefficients
            i = random.randint(0, 11)
            new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_c != current_coeffs[i] and abs(new_c) <= 64:
                new_coeffs = current_coeffs[:]
                new_coeffs[i] = new_c
                p = [0] * (max(current_exps) + 1)
                for e, c in zip(current_exps, new_coeffs):
                    if e < len(p):
                        p[e] = c
                while len(p) > 1 and p[-1] == 0:
                    p.pop()
                if len(p) <= 41:
                    score = _evaluate_poly(p)
                    if score < current_score:
                        current_score = score
                        current_coeffs = new_coeffs
                        poly = p
                        improved = True
        
        if not improved:
            # Random restart of local neighborhood? Or just break?
            # Let's try a random perturbation
            if random.random() < 0.1:
                i = random.randint(0, 11)
                current_coeffs[i] = random.choice([-2, -1, 1, 2])
                p = [0] * (max(current_exps) + 1)
                for e, c in zip(current_exps, current_coeffs):
                    if e < len(p):
                        p[e] = c
                while len(p) > 1 and p[-1] == 0:
                    p.pop()
                if len(p) <= 41:
                    current_score = _evaluate_poly(p)
                    poly = p
            else:
                break

    return current_exps, current_coeffs, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    best_score = float('inf')
    best_exps = None
    best_coeffs = None
    
    random.seed(42)
    
    # Phase 1: Structured Initializations
    # Try palindromic and anti-palindromic structures with random coefficients
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        r = random.random()
        if r < 0.5:
            # Palindromic
            exps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            # Make it palindromic in coefficients? No, just random coeffs for now
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        else:
            # Random exponents
            exps = random.sample(range(0, 41), 12)
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        e, c, s = _local_search(exps, coeffs, max_steps=2000, time_limit=1.0)
        if s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c
    
    # Phase 2: Random Search with Local Optimization
    attempts = 0
    while time.time() < deadline - 5 and attempts < 5000:
        attempts += 1
        
        # Generate random initial state
        exps = random.sample(range(0, 41), 12)
        # Use small coefficients
        coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        
        # Run local search
        e, c, s = _local_search(exps, coeffs, max_steps=3000, time_limit=2.0)
        
        if s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c
            print(f"New best: {s} at attempt {attempts}", flush=True)
            
            # If we found a very good score (e.g., <= 6), we can stop or continue to polish
            if best_score <= 6:
                # Continue polishing
                pass

    # Phase 3: Final Polish
    if best_exps and best_coeffs and time.time() < deadline - 2:
        e, c, s = _local_search(best_exps, best_coeffs, max_steps=5000, time_limit=5.0)
        if s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c

    # Construct final polynomial
    if best_exps is None or best_coeffs is None:
        # Fallback
        return [1]*12
        
    poly = [0] * (max(best_exps) + 1)
    for e, c in zip(best_exps, best_coeffs):
        if e < len(poly):
            poly[e] = c
            
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
        
    if len(poly) > 41:
        poly = poly[:41]
        
    if not _is_valid(poly):
        return [1]*12
        
    return poly

# EVOLVE-BLOCK-END
