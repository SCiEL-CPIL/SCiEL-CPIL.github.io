import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    best_count = float('inf')
    best_coeffs = None

    def eval_square_fast(coeffs_list):
        """
        coeffs_list: list of (exp, coef) with len 12, coef != 0
        Returns: (count_nonzero_in_square, max_exp)
        Uses dictionary for sparse computation.
        """
        if not coeffs_list:
            return 0, 0
        max_e = max(e for e, c in coeffs_list)
        square = {}
        n = len(coeffs_list)
        for i in range(n):
            ei, ci = coeffs_list[i]
            # Square term
            key = 2 * ei
            square[key] = square.get(key, 0) + ci * ci
            # Cross terms
            for j in range(i + 1, n):
                ej, cj = coeffs_list[j]
                key = ei + ej
                square[key] = square.get(key, 0) + 2 * ci * cj
        count = sum(1 for x in square.values() if x != 0)
        return count, max_e

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

    def hill_climb_pairwise(start_cl, time_budget=3.0):
        """
        Local search with pairwise coefficient updates.
        This is the key improvement: updating two coefficients simultaneously
        allows finding cancellation patterns that single-coordinate moves miss.
        """
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = [tuple(x) for x in start_cl]
        if len(cl) != 12:
            return cl, float('inf')
        
        count, _ = eval_square_fast(cl)
        try_update(count, cl)
        
        no_improve_count = 0
        vals_small = [-4, -3, -2, -1, 1, 2, 3, 4]
        vals_med = [-8, -4, -2, -1, 1, 2, 4, 8]
        vals_large = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
        all_vals = [-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64]
        
        while time.time() - t0 < time_budget:
            improved = False
            
            # Try several random moves per iteration
            for _ in range(8):
                move_type = random.random()
                
                if move_type < 0.35:
                    # Single coefficient change
                    idx = random.randrange(12)
                    e, c = cl[idx]
                    new_c = random.choice(vals_med if count > 15 else vals_small)
                    if new_c != c and abs(new_c) <= 64:
                        cl2 = list(cl)
                        cl2[idx] = (e, new_c)
                        c2, _ = eval_square_fast(cl2)
                        if c2 < count:
                            cl = cl2
                            count = c2
                            try_update(count, cl)
                            improved = True
                            no_improve_count = 0
                            break
                            
                elif move_type < 0.65:
                    # PAIRWISE coefficient change - KEY IMPROVEMENT
                    i, j = random.sample(range(12), 2)
                    ei, ci = cl[i]
                    ej, cj = cl[j]
                    new_ci = random.choice(vals_med if count > 15 else vals_small)
                    new_cj = random.choice(vals_med if count > 15 else vals_small)
                    if new_ci != ci and new_cj != cj and abs(new_ci) <= 64 and abs(new_cj) <= 64:
                        cl2 = list(cl)
                        cl2[i] = (ei, new_ci)
                        cl2[j] = (ej, new_cj)
                        c2, _ = eval_square_fast(cl2)
                        if c2 < count:
                            cl = cl2
                            count = c2
                            try_update(count, cl)
                            improved = True
                            no_improve_count = 0
                            break
                            
                elif move_type < 0.85:
                    # Exponent shift
                    idx = random.randrange(12)
                    e, c = cl[idx]
                    shift = random.choice([-2, -1, 1, 2])
                    new_e = e + shift
                    if 0 <= new_e <= 40:
                        exps = [cl[k][0] for k in range(12)]
                        exps.remove(e)
                        if new_e not in exps:
                            cl2 = list(cl)
                            cl2[idx] = (new_e, c)
                            c2, _ = eval_square_fast(cl2)
                            if c2 < count:
                                cl = cl2
                                count = c2
                                try_update(count, cl)
                                improved = True
                                no_improve_count = 0
                                break
                else:
                    # Pairwise exponent + coefficient change
                    i, j = random.sample(range(12), 2)
                    ei, ci = cl[i]
                    ej, cj = cl[j]
                    # Change coefficient of i, shift exponent of j
                    new_ci = random.choice(vals_small)
                    shift = random.choice([-1, 1])
                    new_ej = ej + shift
                    if new_ci != ci and abs(new_ci) <= 64 and 0 <= new_ej <= 40:
                        exps = [cl[k][0] for k in range(12)]
                        exps.remove(ej)
                        if new_ej not in exps:
                            cl2 = list(cl)
                            cl2[i] = (ei, new_ci)
                            cl2[j] = (new_ej, cj)
                            c2, _ = eval_square_fast(cl2)
                            if c2 < count:
                                cl = cl2
                                count = c2
                                try_update(count, cl)
                                improved = True
                                no_improve_count = 0
                                break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 20:
                    # Simulated annealing style perturbation
                    # Accept worse moves with some probability to escape local minima
                    for _ in range(3):
                        idx = random.randrange(12)
                        e, c = cl[idx]
                        new_c = random.choice(vals_large)
                        if new_c != c and abs(new_c) <= 64:
                            cl[idx] = (e, new_c)
                    no_improve_count = 0
                    count, _ = eval_square_fast(cl)
                    try_update(count, cl)
                    
            if best_count == 12:
                break
        return cl, count

    def anneal_search(start_cl, time_budget=5.0):
        """
        Simulated annealing for coefficient optimization.
        Uses a smooth surrogate: minimize count + small penalty for large coefficients.
        """
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = [tuple(x) for x in start_cl]
        if len(cl) != 12:
            return cl, float('inf')
        
        count, _ = eval_square_fast(cl)
        try_update(count, cl)
        
        # Objective: primarily count, secondarily sum of abs coefficients
        def objective(cl):
            c, _ = eval_square_fast(cl)
            coeff_penalty = sum(abs(ci) for _, ci in cl) * 0.01
            return c + coeff_penalty
        
        T = 2.0  # initial temperature
        T_min = 0.01
        cooling = 0.98
        
        current_obj = objective(cl)
        best_local_obj = current_obj
        best_local_cl = list(cl)
        
        while time.time() - t0 < time_budget and T > T_min:
            # Generate neighbor
            cl2 = list(cl)
            move_type = random.random()
            
            if move_type < 0.4:
                idx = random.randrange(12)
                e, c = cl2[idx]
                new_c = c + random.choice([-4, -2, -1, 1, 2, 4])
                if new_c != 0 and abs(new_c) <= 64:
                    cl2[idx] = (e, new_c)
            elif move_type < 0.7:
                i, j = random.sample(range(12), 2)
                ei, ci = cl2[i]
                ej, cj = cl2[j]
                new_ci = ci + random.choice([-2, -1, 1, 2])
                new_cj = cj + random.choice([-2, -1, 1, 2])
                if new_ci != 0 and new_cj != 0 and abs(new_ci) <= 64 and abs(new_cj) <= 64:
                    cl2[i] = (ei, new_ci)
                    cl2[j] = (ej, new_cj)
            else:
                idx = random.randrange(12)
                e, c = cl2[idx]
                shift = random.choice([-1, 1])
                new_e = e + shift
                if 0 <= new_e <= 40:
                    exps = [cl2[k][0] for k in range(12)]
                    exps.remove(e)
                    if new_e not in exps:
                        cl2[idx] = (new_e, c)
            
            new_obj = objective(cl2)
            delta_obj = new_obj - current_obj
            
            # Metropolis criterion
            if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
                cl = cl2
                current_obj = new_obj
                if new_obj < best_local_obj:
                    best_local_obj = new_obj
                    best_local_cl = list(cl)
                    # Update global best
                    c_count, _ = eval_square_fast(cl)
                    try_update(c_count, cl)
            
            T *= cooling
            
            if best_count == 12:
                break
        
        return best_local_cl, best_local_obj

    # =====================================================================
    # Phase 1: Structural Search via Factorization
    # =====================================================================
    
    def convolve_sparse(poly1, poly2):
        res = {}
        for e1, c1 in poly1:
            for e2, c2 in poly2:
                e = e1 + e2
                res[e] = res.get(e, 0) + c1 * c2
        return [(e, c) for e, c in res.items() if c != 0]

    # Generate candidate factor sets for 2*2*3 = 12 term products
    pool_2 = []
    for k in range(1, 11):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                pool_2.append([(0, a), (k, b)])
    
    pool_3 = []
    for k in range(1, 8):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                for c in [-2, -1, 1, 2]:
                    if a == 0 or b == 0 or c == 0:
                        continue
                    pool_3.append([(0, a), (k, b), (2*k, c)])
                    if k + 1 <= 15:
                        pool_3.append([(0, a), (k, b), (k+1, c)])
    
    # Pre-compute products of pairs from pool_2
    pair_products = []
    for i in range(len(pool_2)):
        for j in range(i, len(pool_2)):
            prod = convolve_sparse(pool_2[i], pool_2[j])
            if len(prod) <= 4:
                pair_products.append(prod)
    
    seen_polys = set()
    
    for pp in pair_products:
        if time.time() > deadline - 30:
            break
        max_e_pp = max(e for e, _ in pp)
        for c in pool_3:
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
            count, _ = eval_square_fast(p_sorted)
            try_update(count, p_sorted)
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 2: Local Search from Structural Seeds and Structured Starts
    # =====================================================================
    
    random.seed(42)
    
    # Start from best found so far
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        hill_climb_pairwise(start_cl, time_budget=8.0)
        if best_count == 12:
            return best_coeffs
        anneal_search(start_cl, time_budget=5.0)
        if best_count == 12:
            return best_coeffs
    
    # Structured random starts - prioritize patterns known to give cancellations
    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        
        # Generate structured exponent patterns
        r = random.random()
        if r < 0.25:
            # Consecutive
            exps = list(range(12))
        elif r < 0.40:
            # Arithmetic progression with step 2
            start = random.randint(0, 2)
            exps = [start + 2*i for i in range(12)]
            if exps[-1] > 40:
                exps = list(range(12))
        elif r < 0.55:
            # Arithmetic progression with step 3
            start = random.randint(0, 1)
            exps = [start + 3*i for i in range(12)]
            if exps[-1] > 40:
                exps = list(range(12))
        elif r < 0.70:
            # Two interleaved arithmetic progressions
            base = random.randint(0, 1)
            exps = sorted([base + 4*i for i in range(6)] + [base+1 + 4*i for i in range(6)])
            if exps[-1] > 40:
                exps = list(range(12))
        elif r < 0.85:
            # Random but with small gaps
            exps = sorted(random.sample(range(0, 35), 12))
        else:
            # Fully random
            exps = sorted(random.sample(range(40), 12))
        
        # Assign coefficients with some structure
        if random.random() < 0.5:
            # Alternating signs
            cl = [(exps[i], random.choice(vals) * (1 if i % 2 == 0 else -1)) for i in range(12)]
        else:
            cl = [(exps[i], random.choice(vals)) for i in range(12)]
        
        # Ensure non-zero
        for i in range(12):
            if cl[i][1] == 0:
                cl[i] = (cl[i][0], 1)
        
        hill_climb_pairwise(cl, time_budget=1.5)
        if best_count == 12:
            return best_coeffs
        
        # Occasionally try annealing from good candidates
        if attempts % 5 == 0:
            anneal_search(cl, time_budget=2.0)
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 3: Final High-Precision Polish
    # =====================================================================
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        
        # Multiple polish passes with different strategies
        for polish_round in range(3):
            if time.time() > deadline - 3:
                break
            remaining = deadline - time.time()
            budget = min(remaining * 0.3, 5.0)
            if budget < 1.0:
                break
            
            if polish_round == 0:
                # Fine-grained coefficient search
                hill_climb_pairwise(start_cl, time_budget=budget)
            elif polish_round == 1:
                # Annealing
                anneal_search(start_cl, time_budget=budget)
            else:
                # More aggressive search
                hill_climb_pairwise(start_cl, time_budget=budget)
            
            if best_count == 12:
                return best_coeffs
            
            # Update start_cl from best
            start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]

    # Fallback
    if best_coeffs is None:
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1] + [0]*29
        if best_coeffs[-1] == 0:
            for i in range(len(best_coeffs)-1, -1, -1):
                if best_coeffs[i] != 0:
                    best_coeffs = best_coeffs[:i+1]
                    break

    return best_coeffs
