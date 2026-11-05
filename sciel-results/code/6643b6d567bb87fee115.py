import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    best_count = float('inf')
    best_coeffs = None

    def eval_square(coeffs_list):
        """
        coeffs_list: list of (exp, coef) with len 12, coef != 0
        Returns: (count_nonzero_in_square, square_list, max_exp)
        """
        if not coeffs_list:
            return 0, [0], 0
        max_e = max(e for e, c in coeffs_list)
        square = [0] * (2 * max_e + 1)
        
        cl = sorted(coeffs_list)
        
        for i in range(12):
            ei, ci = cl[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, 12):
                ej, cj = cl[j]
                square[ei + ej] += 2 * ci * cj
        
        count = sum(1 for x in square if x != 0)
        return count, square, max_e

    def to_full(coeffs_list):
        if not coeffs_list:
            return None
        max_e = max(e for e, c in coeffs_list)
        if max_e >= 41:
            return None
        
        full = [0] * (max_e + 1)
        for e, c in coeffs_list:
            full[e] = c
        
        if len(full) > 41:
            return None
        if full[-1] == 0:
            return None
        
        nz = sum(1 for x in full if x != 0)
        if nz != 12:
            return None
        if any(abs(x) > 64 for x in full):
            return None
            
        return full

    def try_update(count, coeffs_list):
        nonlocal best_count, best_coeffs
        if count < best_count:
            full = to_full(coeffs_list)
            if full is not None:
                best_count = count
                best_coeffs = full
                return True
        return False

    def simulated_annealing(start_cl, time_budget=5.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = [tuple(x) for x in start_cl]
        if len(cl) != 12:
            return cl, float('inf')
        
        count, _, _ = eval_square(cl)
        try_update(count, cl)
        
        current_temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        # Expanded coefficient values for better cancellation
        vals = [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16, -32, 32]
        
        while time.time() - t0 < time_budget and current_temp > min_temp:
            move_type = random.random()
            
            if move_type < 0.6:
                # Change coefficient
                idx = random.randrange(12)
                e, c = cl[idx]
                new_c = random.choice(vals)
                if new_c != c and abs(new_c) <= 64:
                    cl2 = list(cl)
                    cl2[idx] = (e, new_c)
                    c2, _, _ = eval_square(cl2)
                    delta = c2 - count
                    if delta < 0 or random.random() < np.exp(-delta / current_temp):
                        cl = cl2
                        count = c2
                        try_update(count, cl)
            elif move_type < 0.9:
                # Shift exponent
                idx = random.randrange(12)
                e, c = cl[idx]
                shift = random.choice([-2, -1, 1, 2])
                new_e = e + shift
                if 0 <= new_e <= 40:
                    exps = [cl[i][0] for i in range(12)]
                    exps.remove(e)
                    if new_e not in exps:
                        cl2 = list(cl)
                        cl2[idx] = (new_e, c)
                        c2, _, _ = eval_square(cl2)
                        delta = c2 - count
                        if delta < 0 or random.random() < np.exp(-delta / current_temp):
                            cl = cl2
                            count = c2
                            try_update(count, cl)
            else:
                # Swap coefficients of two terms
                i, j = random.sample(range(12), 2)
                e_i, c_i = cl[i]
                e_j, c_j = cl[j]
                if c_i != c_j:
                    cl2 = list(cl)
                    cl2[i] = (e_i, c_j)
                    cl2[j] = (e_j, c_i)
                    c2, _, _ = eval_square(cl2)
                    delta = c2 - count
                    if delta < 0 or random.random() < np.exp(-delta / current_temp):
                        cl = cl2
                        count = c2
                        try_update(count, cl)
                        
            current_temp *= cooling
            
            if best_count <= 12:
                break
                
        return cl, count

    # =====================================================================
    # Phase 1: Structural Search via Factorization (Retained)
    # =====================================================================
    
    def convolve_sparse(poly1, poly2):
        res = {}
        for e1, c1 in poly1:
            for e2, c2 in poly2:
                e = e1 + e2
                res[e] = res.get(e, 0) + c1 * c2
        return [(e, c) for e, c in res.items() if c != 0]
    
    candidate_factors_pool = []
    for k in range(1, 11):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                if a == 0 or b == 0: continue
                candidate_factors_pool.append([(0, a), (k, b)])
                
    for k in range(1, 8):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                for c in [-2, -1, 1, 2]:
                    if a==0 or b==0 or c==0: continue
                    candidate_factors_pool.append([(0, a), (k, b), (2*k, c)])
                    if k + 1 <= 15:
                        candidate_factors_pool.append([(0, a), (k, b), (k+1, c)])
    
    pool_2 = [f for f in candidate_factors_pool if len(f) == 2]
    pool_3 = [f for f in candidate_factors_pool if len(f) == 3]
    
    pair_products = []
    for i in range(len(pool_2)):
        for j in range(i, len(pool_2)):
            prod = convolve_sparse(pool_2[i], pool_2[j])
            if len(prod) <= 4:
                pair_products.append(prod)
                
    seen_polys = set()
    
    for pp in pair_products:
        if time.time() > deadline - 20:
            break
        for c in pool_3:
            max_e_pp = max(e for e, _ in pp)
            max_e_c = max(e for e, _ in c)
            if max_e_pp + max_e_c > 40:
                continue
            
            p_poly = convolve_sparse(pp, c)
            if len(p_poly) != 12:
                continue
            
            p_sorted = sorted(p_poly)
            key = tuple(p_sorted)
            
            if key in seen_polys:
                continue
            seen_polys.add(key)
            
            count, _, _ = eval_square(p_sorted)
            try_update(count, p_sorted)
            
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 2: Brute Force Signs for Consecutive Exponents (from Inspiration 1)
    # This is fast and establishes a strong baseline
    # =====================================================================
    
    for bits in range(1 << 12):
        if time.time() > deadline - 25:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        cl = [(i, c) for i, c in enumerate(coeffs)]
        count, _, _ = eval_square(cl)
        try_update(count, cl)
        
        if best_count == 12:
            return best_coeffs

    # =====================================================================
    # Phase 3: Simulated Annealing from Best Found and Structured Seeds
    # =====================================================================
    
    random.seed(42)
    
    # Start from best found so far
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        simulated_annealing(start_cl, time_budget=10.0)
        if best_count == 12:
            return best_coeffs
    
    # Start from various structured patterns
    structured_starts = []
    
    # Consecutive
    structured_starts.append([(i, 1) for i in range(12)])
    
    # Arithmetic Progressions with various steps
    for step in [2, 3, 4]:
        if 11 * step <= 40:
            structured_starts.append([(i * step, 1) for i in range(12)])
    
    # Alternating signs on APs
    for step in [2, 3]:
        if 11 * step <= 40:
            structured_starts.append([(i * step, (-1)**i) for i in range(12)])
    
    for start in structured_starts:
        if time.time() > deadline - 15:
            break
        simulated_annealing(start, time_budget=3.0)
        if best_count == 12:
            return best_coeffs

    # Random starts with diverse exponent patterns
    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        
        r = random.random()
        if r < 0.2:
            # Consecutive
            exps = list(range(12))
        elif r < 0.4:
            # Arithmetic Progression
            step = random.randint(1, 3)
            if 11 * step > 40:
                step = 3
            exps = [i * step for i in range(12)]
        elif r < 0.6:
            # Dense random (from 0..15)
            exps = sorted(random.sample(range(16), 12))
        else:
            # Random
            exps = sorted(random.sample(range(41), 12))
        
        cl = [(exps[i], random.choice(vals)) for i in range(12)]
        
        simulated_annealing(cl, time_budget=2.0)
        if best_count == 12:
            return best_coeffs

    # =====================================================================
    # Phase 4: Final Polish on Best Found
    # =====================================================================
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        simulated_annealing(start_cl, time_budget=8.0)
        if best_count == 12:
            return best_coeffs

    # Fallback
    if best_coeffs is None:
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        if best_coeffs[-1] == 0:
            for i in range(len(best_coeffs)-1, -1, -1):
                if best_coeffs[i] != 0:
                    best_coeffs = best_coeffs[:i+1]
                    break
            if not best_coeffs:
                best_coeffs = [1]
        
        nz = [i for i, c in enumerate(best_coeffs) if c != 0]
        if len(nz) != 12:
            best_coeffs = [0]*41
            for i in range(12):
                best_coeffs[i] = 1
            
    return best_coeffs
