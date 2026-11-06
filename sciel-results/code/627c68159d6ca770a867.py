import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    # Max degree is 40, so exponents are in [0, 40]
    # Length of square is up to 80, so array size 81
    
    best_score = 1000
    best_exps = None
    best_coeffs = None
    
    def evaluate(exps, coeffs):
        sq = np.zeros(81, dtype=np.int64)
        for i in range(12):
            ei = exps[i]
            ci = coeffs[i]
            for j in range(12):
                ej = exps[j]
                cj = coeffs[j]
                idx = ei + ej
                sq[idx] += ci * cj
        return np.count_nonzero(sq)
    
    def random_candidate():
        exps = random.sample(range(41), 12)
        exps.sort()
        coeffs = [random.randint(1, 64) * random.choice([1, -1]) for _ in range(12)]
        return exps, coeffs

    # Phase 1: Random Search
    for _ in range(50000):
        if time.time() - start_time > time_limit: break
        exps, coeffs = random_candidate()
        score = evaluate(exps, coeffs)
        if score < best_score:
            best_score = score
            best_exps = exps
            best_coeffs = coeffs
            if best_score == 12:
                break

    # Phase 2: Local Search
    if best_exps is not None:
        current_exps = best_exps.copy()
        current_coeffs = best_coeffs.copy()
        current_score = best_score
        
        no_improve = 0
        while time.time() - start_time < time_limit:
            if random.random() < 0.5:
                # Perturb coefficient
                idx = random.randint(0, 11)
                old_c = current_coeffs[idx]
                # Pick a new non-zero coefficient
                new_c = random.randint(1, 64) * random.choice([1, -1])
                if new_c == old_c:
                    continue
                current_coeffs[idx] = new_c
                new_score = evaluate(current_exps, current_coeffs)
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_exps = current_exps.copy()
                        best_coeffs = current_coeffs.copy()
                        if best_score == 12:
                            break
                else:
                    current_coeffs[idx] = old_c
                    no_improve += 1
            else:
                # Perturb exponent
                idx = random.randint(0, 11)
                old_e = current_exps[idx]
                # Find available exponents
                # Optimization: precompute set of used exponents?
                # 12 items, 41 total. Set lookup is O(1).
                used = set(current_exps)
                available = [e for e in range(41) if e not in used]
                if not available:
                    continue
                new_e = random.choice(available)
                current_exps[idx] = new_e
                current_exps.sort()
                new_score = evaluate(current_exps, current_coeffs)
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_exps = current_exps.copy()
                        best_coeffs = current_coeffs.copy()
                        if best_score == 12:
                            break
                else:
                    current_exps[idx] = old_e
                    current_exps.sort()
                    no_improve += 1
            
            if no_improve > 1000:
                # Restart
                current_exps, current_coeffs = random_candidate()
                current_score = evaluate(current_exps, current_coeffs)
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_exps = current_exps.copy()
                    best_coeffs = current_coeffs.copy()

    # Construct output
    if best_exps is None:
        return [1] * 12
    
    result = [0] * 41
    for e, c in zip(best_exps, best_coeffs):
        result[e] = c
    
    # Trim trailing zeros
    while result and result[-1] == 0:
        result.pop()
        
    return result
