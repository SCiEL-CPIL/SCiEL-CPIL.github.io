import random
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80
    
    def square_nonzero(coeffs):
        n = len(coeffs)
        square = defaultdict(int)
        for i in range(n):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(i, n):
                cj = coeffs[j]
                if cj == 0:
                    continue
                deg = i + j
                if i == j:
                    square[deg] += ci * cj
                else:
                    square[deg] += 2 * ci * cj
        return sum(1 for v in square.values() if v != 0)
    
    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0 or coeffs[-1] == 0:
            return False
        nz = sum(1 for c in coeffs if c != 0)
        if nz != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True
    
    def coeffs_from_state(exponents, coeffs):
        max_deg = max(exponents) if exponents else 0
        full = [0] * (max_deg + 1)
        for e, c in zip(exponents, coeffs):
            full[e] = c
        return full
    
    def score_of(full_coeffs):
        if not is_valid(full_coeffs):
            return 10**9
        return square_nonzero(full_coeffs)
    
    def extract_state(full):
        exponents = []
        coeffs = []
        for i, c in enumerate(full):
            if c != 0:
                exponents.append(i)
                coeffs.append(c)
        return exponents, coeffs
    
    best_overall = None
    best_overall_score = 10**9
    
    initial_configs = [
        (list(range(12)), [1 if i%2==0 else -1 for i in range(12)]),
        (list(range(12)), [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]),
        (list(range(0, 24, 2)), [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]),
        ([0,1,2,3,4,5,7,8,9,10,11,12], [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]),
        (list(range(0, 40, 3)), [1]*12),
        (list(range(0, 40, 4)), [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]),
    ]
    
    for exponents, coeffs in initial_configs:
        full = coeffs_from_state(exponents, coeffs)
        s = score_of(full)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = full[:]
    
    if best_overall is None:
        best_overall = [1]*12
        best_overall_score = square_nonzero(best_overall)
    
    # Local search
    exponents, coeffs = extract_state(best_overall)
    random.seed(42)
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        move_type = random.random()
        saved_exponents = exponents[:]
        saved_coeffs = coeffs[:]
        
        if move_type < 0.4:
            idx = random.randint(0, 11)
            coeffs[idx] = -coeffs[idx] if coeffs[idx] != 0 else 1
        elif move_type < 0.7:
            idx = random.randint(0, 11)
            delta = random.choice([-1, 1])
            new_exp = exponents[idx] + delta
            if new_exp < 0:
                exponents, coeffs = saved_exponents, saved_coeffs
                continue
            if any(e == new_exp for e in exponents if e != exponents[idx]):
                exponents, coeffs = saved_exponents, saved_coeffs
                continue
            exponents[idx] = new_exp
        else:
            idx = random.randint(0, 11)
            coeffs[idx] = random.choice([-2, -1, 1, 2, -3, 3])
        
        full = coeffs_from_state(exponents, coeffs)
        s = score_of(full)
        
        if s < best_overall_score:
            best_overall_score = s
            best_overall = full[:]
            no_improve = 0
        else:
            no_improve += 1
            exponents, coeffs = saved_exponents, saved_coeffs
    
    if best_overall is None:
        best_overall = [1]*12
    
    if not is_valid(best_overall):
        best_overall = [1]*12
    
    return best_overall
# EVOLVE-BLOCK-END
