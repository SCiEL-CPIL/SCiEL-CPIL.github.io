import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safe margin for 60s limit
    
    def count_nz_square(poly):
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return np.count_nonzero(sq)
    
    def valid_poly(p):
        if len(p) != 41:
            return False
        if p[-1] == 0:
            return False
        nz = sum(1 for x in p if x != 0)
        if nz != 12:
            return False
        if any(abs(x) > 64 for x in p):
            return False
        return True
    
    best_score = float('inf')
    best_poly = None
    
    # Strategy: Use product structure P = A * B where A and B are sparse
    # P^2 = A^2 * B^2
    # If A^2 and B^2 are sparse, P^2 will be sparse
    
    # Key insight: If A has k terms and B has m terms, P has at most k*m terms
    # We need exactly 12 terms in P, so we can try k=3, m=4 or k=4, m=3 or k=2, m=6 etc.
    
    # Let's try: A = a0 + a1*x^p + a2*x^q (3 terms)
    #           B = b0 + b1*x^r + b2*x^s + b3*x^t (4 terms)
    # P = A*B has at most 12 terms (if all products have distinct exponents)
    
    # For P^2 to be sparse, we need A^2*B^2 to be sparse
    # A^2 has at most 6 terms (3*2/2 + 3 = 6, or fewer if cancellations)
    # B^2 has at most 10 terms (4*5/2 = 10, or fewer)
    # A^2*B^2 has at most 60 terms, but with structure could be much less
    
    # Better: Try to make A^2 and B^2 have special structure
    # If A = 1 + x^p + x^q, then A^2 = 1 + 2x^p + 2x^q + x^(2p) + 2x^(p+q) + x^(2q)
    # If we choose p, q such that some exponents coincide... but p != q so no coincidence
    
    # Alternative: Use the fact that (1+x+x^2)(1+x^3+x^6) = 1+x+x^2+x^3+x^4+x^5+x^6+x^7+x^8 (9 terms)
    # Not quite 12.
    
    # Let's try a more systematic approach:
    # Generate random sparse A and B, compute P = A*B, check if P has exactly 12 terms
    # Then evaluate P^2
    
    # Coefficient values to try
    coeff_vals = [-2, -1, 1, 2]
    
    # Try various splits: A has k terms, B has m terms, k*m >= 12
    splits = [(3, 4), (4, 3), (2, 6), (6, 2), (3, 5), (5, 3), (4, 4)]
    
    # For each split, try random exponent patterns
    max_exponent = 40
    
    # Pre-generate some structured candidates
    candidates = []
    
    # Candidate type 1: A = sum of 3 terms, B = sum of 4 terms
    for k, m in splits:
        # Try several random exponent patterns
        for trial in range(50):
            # Generate k distinct exponents for A
            exp_a = sorted(random.sample(range(0, max_exponent + 1), k))
            # Generate m distinct exponents for B
            exp_b = sorted(random.sample(range(0, max_exponent + 1), m))
            
            # Check that all k*m products have distinct exponents (so P has exactly k*m terms)
            products = set()
            valid = True
            for ea in exp_a:
                for eb in exp_b:
                    prod = ea + eb
                    if prod in products:
                        valid = False
                        break
                    products.add(prod)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check max exponent
            if max(products) > 40:
                continue
            
            # Try random coefficients
            coeffs_a = [random.choice(coeff_vals) for _ in range(k)]
            coeffs_b = [random.choice(coeff_vals) for _ in range(m)]
            
            # Build A and B
            A = [0] * 41
            B = [0] * 41
            for i, ea in enumerate(exp_a):
                A[ea] = coeffs_a[i]
            for i, eb in enumerate(exp_b):
                B[eb] = coeffs_b[i]
            
            # Compute P = A * B
            P = np.convolve(np.array(A, dtype=np.int64), np.array(B, dtype=np.int64))
            
            if len(P) > 41:
                continue
            
            # Pad P to length 41
            P_padded = [0] * 41
            for i, val in enumerate(P):
                if i < 41:
                    P_padded[i] = int(val)
            
            # Check validity
            if not valid_poly(P_padded):
                continue
            
            # Check exactly 12 terms
            nz = sum(1 for x in P_padded if x != 0)
            if nz != 12:
                continue
            
            # Evaluate
            score = count_nz_square(P_padded)
            if score < best_score:
                best_score = score
                best_poly = P_padded[:]
            
            candidates.append(P_padded[:])
    
    # Also try: P = A + B where A and B have disjoint supports and special structure
    # A has 6 terms at even positions, B has 6 terms at odd positions
    # Then A^2 and B^2 have only even degrees, 2AB has only odd degrees
    # No cancellation between A^2+B^2 and 2AB
    
    # Try: A = a0 + a1*x^2 + a2*x^4 + a3*x^6 + a4*x^8 + a5*x^10
    #      B = b0*x + b1*x^3 + b2*x^5 + b3*x^7 + b4*x^9 + b5*x^11
    # P = A + B
    # P^2 = A^2 + 2AB + B^2
    # A^2: degrees 0,2,4,...,20 (11 terms max)
    # B^2: degrees 2,4,6,...,22 (11 terms max)
    # 2AB: degrees 1,3,5,...,21 (11 terms max)
    # Total: up to 33 terms, but A^2 and B^2 overlap at even degrees 2,4,...,20
    
    # Try random coefficients
    for trial in range(100):
        A = [0] * 41
        B = [0] * 41
        for i in range(6):
            A[2*i] = random.choice(coeff_vals)
            B[2*i + 1] = random.choice(coeff_vals)
        
        P = [a + b for a, b in zip(A, B)]
        
        if not valid_poly(P):
            continue
        
        score = count_nz_square(P)
        if score < best_score:
            best_score = score
            best_poly = P[:]
        
        candidates.append(P[:])
    
    # If no valid candidate found, create a default
    if best_poly is None:
        best_poly = [0] * 41
        # Use a simple 12-term polynomial
        for i in range(12):
            best_poly[i] = 1
        best_score = count_nz_square(best_poly)
        candidates.append(best_poly[:])
    
    # Local search with simulated annealing
    current_poly = best_poly[:]
    current_score = best_score
    
    T = 2.0
    T_min = 0.05
    cooling_rate = 0.999
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            if T > T_min:
                T = max(T_min, T * cooling_rate)
            else:
                # Random restart
                if candidates:
                    current_poly = random.choice(candidates).copy()
                else:
                    current_poly = best_poly.copy()
                current_score = count_nz_square(current_poly)
                no_improve_count = 0
                T = 2.0
            continue
        
        # Generate neighbor
        move_type = random.random()
        
        if move_type < 0.6:
            # Change coefficient of a nonzero term
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            
            # Try values that might create cancellations
            new_val = random.choice([-3, -2, -1, 1, 2, 3])
            if new_val == old_val:
                new_val = old_val + random.choice([-1, 1])
                if new_val == 0:
                    new_val = 1
            if abs(new_val) > 64:
                new_val = 64 if new_val > 0 else -64
            
            current_poly[idx] = new_val
            new_score = count_nz_square(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                delta_score = new_score - current_score
                if delta_score > 0 and random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                else:
                    current_poly[idx] = old_val
                    no_improve_count += 1
        elif move_type < 0.85:
            # Move a term to a new position
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            z_indices = [i for i in range(41) if current_poly[i] == 0]
            if not z_indices:
                no_improve_count += 1
                continue
                
            src_idx = random.choice(nz_indices)
            dst_idx = random.choice(z_indices)
            
            if src_idx == 40:
                no_improve_count += 1
                continue
            
            val = current_poly[src_idx]
            current_poly[src_idx] = 0
            current_poly[dst_idx] = val
            
            new_score = count_nz_square(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                delta_score = new_score - current_score
                if delta_score > 0 and random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                else:
                    current_poly[dst_idx] = 0
                    current_poly[src_idx] = val
                    no_improve_count += 1
        else:
            # Swap two terms
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            if len(nz_indices) < 2:
                no_improve_count += 1
                continue
                
            i1, i2 = random.sample(nz_indices, 2)
            if i1 == 40 or i2 == 40:
                no_improve_count += 1
                continue
            
            v1 = current_poly[i1]
            v2 = current_poly[i2]
            current_poly[i1] = v2
            current_poly[i2] = v1
            
            new_score = count_nz_square(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                delta_score = new_score - current_score
                if delta_score > 0 and random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                else:
                    current_poly[i1] = v1
                    current_poly[i2] = v2
                    no_improve_count += 1
        
        # Cool down
        if no_improve_count % 100 == 0 and T > T_min:
            T = max(T_min, T * cooling_rate)
    
    return best_poly
