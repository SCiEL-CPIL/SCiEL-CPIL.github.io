import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0
    
    # Fast evaluation
    def square_nonzero(exponents, coeffs):
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        # minlength 81 covers max degree 40+40=80
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=81)
        return int(np.count_nonzero(result))

    # Validation
    def is_valid(exp, coef):
        if len(exp) != 12 or len(coef) != 12:
            return False
        if len(set(exp)) != 12:
            return False
        if any(e < 0 or e >= 41 for e in exp):
            return False
        if any(c == 0 or abs(c) > 64 for c in coef):
            return False
        return True

    def make_poly(exp, coef):
        max_deg = max(exp)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exp, coef):
            poly[e] = int(c)
        return poly

    # Generate initial candidates (Inspiration strategy)
    def generate_initial_candidates():
        candidates = []
        random.seed(42)
        
        # 1. Arithmetic Progressions
        for d in range(1, 7):
            for a in range(0, 41 - 11*d):
                exps = [a + i*d for i in range(12)]
                # Try a few coefficient patterns
                for _ in range(5):
                    coeffs = [random.choice([-1, 1]) for _ in range(12)]
                    candidates.append((exps, coeffs))
                    coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                    candidates.append((exps, coeffs))
        
        # 2. Symmetric Structures
        for center in range(6, 35):
            if random.random() < 0.5:
                offsets = sorted(random.sample(range(1, 20), 6))
                exps = sorted([center - o for o in offsets] + [center + o for o in offsets])
                if 0 <= exps[0] and exps[-1] < 41 and len(set(exps)) == 12:
                    coeffs = [random.choice([-1, 1]) for _ in range(12)]
                    candidates.append((exps, coeffs))
                    coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                    candidates.append((exps, coeffs))
        
        # 3. Random
        for _ in range(50):
            exps = sorted(random.sample(range(41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            candidates.append((exps, coeffs))
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            candidates.append((exps, coeffs))
            
        return candidates

    # Simulated Annealing (Current program strategy, refined)
    def simulated_annealing(init_exps, init_coeffs, time_budget):
        cur_exps = list(init_exps)
        cur_coeffs = list(init_coeffs)
        cur_score = square_nonzero(cur_exps, cur_coeffs)
        
        best_exps = list(cur_exps)
        best_coeffs = list(cur_coeffs)
        best_score = cur_score
        
        T = 10.0
        cooling = 0.995
        
        while time.time() < time_budget:
            new_exps = list(cur_exps)
            new_coeffs = list(cur_coeffs)
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Shift one exponent
                idx = random.randint(0, 11)
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_e = new_exps[idx] + shift
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue
            elif move_type < 0.6:
                # Change one coefficient
                idx = random.randint(0, 11)
                new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 12, 16, 32])
                if new_c != new_coeffs[idx]:
                    new_coeffs[idx] = new_c
                else:
                    continue
            elif move_type < 0.8:
                # Flip sign of one coefficient
                idx = random.randint(0, 11)
                new_coeffs[idx] = -new_coeffs[idx]
            else:
                # Shift two exponents
                i, j = random.sample(range(12), 2)
                shift1 = random.choice([-2, -1, 1, 2])
                shift2 = random.choice([-2, -1, 1, 2])
                new_ei = new_exps[i] + shift1
                new_ej = new_exps[j] + shift2
                if (0 <= new_ei < 41 and 0 <= new_ej < 41 and 
                    new_ei not in new_exps and new_ej not in new_exps and 
                    new_ei != new_ej):
                    new_exps[i] = new_ei
                    new_exps[j] = new_ej
                    new_exps.sort()
                else:
                    continue
            
            if not is_valid(new_exps, new_coeffs):
                continue
            
            new_score = square_nonzero(new_exps, new_coeffs)
            delta = new_score - cur_score
            
            if delta < 0:
                cur_exps = new_exps
                cur_coeffs = new_coeffs
                cur_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_exps = list(new_exps)
                    best_coeffs = list(new_coeffs)
            else:
                if random.random() < np.exp(-delta / T):
                    cur_exps = new_exps
                    cur_coeffs = new_coeffs
                    cur_score = new_score
            
            T *= cooling
            if T < 0.1:
                T = 1.0
                
        return best_exps, best_coeffs, best_score

    # Local Polish (Hill Climbing)
    def local_polish(exp, coef, time_budget):
        cur_exp = list(exp)
        cur_coef = list(coef)
        cur_nz = square_nonzero(cur_exp, cur_coef)
        
        while time.time() < time_budget:
            improved = False
            # Try all single moves
            for i in range(12):
                # Try changing coefficient
                for c in range(-10, 11):
                    if c == 0: continue
                    if abs(c) > 64: continue
                    if c == cur_coef[i]: continue
                    new_coef = list(cur_coef)
                    new_coef[i] = c
                    if not is_valid(cur_exp, new_coef): continue
                    nz = square_nonzero(cur_exp, new_coef)
                    if nz < cur_nz:
                        cur_coef = new_coef
                        cur_nz = nz
                        improved = True
                        break
                if improved: break
                
                # Try shifting exponent
                for shift in [-3, -2, -1, 1, 2, 3]:
                    new_e = cur_exp[i] + shift
                    if 0 <= new_e < 41 and new_e not in cur_exp:
                        new_exp = list(cur_exp)
                        new_exp[i] = new_e
                        new_exp.sort()
                        if not is_valid(new_exp, cur_coef): continue
                        nz = square_nonzero(new_exp, cur_coef)
                        if nz < cur_nz:
                            cur_exp = new_exp
                            cur_nz = nz
                            improved = True
                            break
                if improved: break
                
            if not improved:
                break
                
        return cur_exp, cur_coef, cur_nz

    # Main Logic
    best_exp = None
    best_coef = None
    best_nz = float('inf')
    
    candidates = generate_initial_candidates()
    
    # Evaluate initial candidates
    for exp, coef in candidates:
        if time.time() > deadline - 10:
            break
        if not is_valid(exp, coef):
            continue
        nz = square_nonzero(exp, coef)
        if nz < best_nz:
            best_nz = nz
            best_exp = list(exp)
            best_coef = list(coef)
            
    # Run SA from best and some diverse starts
    if best_exp is not None:
        e, c, nz = simulated_annealing(best_exp, best_coef, deadline - 5)
        if nz < best_nz:
            best_nz = nz
            best_exp = e
            best_coef = c
            
    # Run SA from a few other good starts
    for i in range(5):
        if time.time() > deadline - 5:
            break
        # Pick a random candidate that wasn't the best
        idx = random.randint(0, len(candidates)-1)
        exp, coef = candidates[idx]
        e, c, nz = simulated_annealing(exp, coef, deadline - 5)
        if nz < best_nz:
            best_nz = nz
            best_exp = e
            best_coef = c
            
    # Final Polish
    if best_exp is not None:
        e, c, nz = local_polish(best_exp, best_coef, deadline - 1)
        if nz < best_nz:
            best_nz = nz
            best_exp = e
            best_coef = c

    # Finalize
    if best_exp is None:
        # Fallback
        best_exp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        best_coef = [1]*12
        
    poly = make_poly(best_exp, best_coef)
    
    # Validation
    if not poly or len(poly) > 41 or poly[-1] == 0:
        # Try a safe fallback
        poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        
    return poly
