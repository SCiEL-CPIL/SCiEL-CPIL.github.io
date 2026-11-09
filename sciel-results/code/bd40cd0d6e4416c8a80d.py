"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n = 30
    
    def eval_sum(coeffs):
        total = Fraction(0, 1)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                total += Fraction(c, i)
        return total
    
    def abs_frac(f):
        return -f if f < 0 else f
    
    best_coeffs = None
    best_val = None
    best_abs = None
    
    def update_best(coeffs, val):
        nonlocal best_coeffs, best_val, best_abs
        if val == 0:
            return
        a = abs_frac(val)
        if best_abs is None or a < best_abs:
            best_abs = a
            best_val = val
            best_coeffs = coeffs[:]
    
    # Diverse initial configurations
    inits = []
    
    # 1. Seed-like: only last two nonzero
    inits.append([0]*28 + [1, -1])
    
    # 2. Alternating signs
    inits.append([(-1)**i for i in range(n)])
    
    # 3. All ones then all -1s
    inits.append([1]*15 + [-1]*15)
    
    # 4. Random with bias toward 0
    random.seed(12345)
    for _ in range(5):
        inits.append([random.choice([-1,0,1]) for _ in range(n)])
    
    # 5. Structured: groups of same sign
    inits.append([1]*10 + [0]*10 + [-1]*10)
    inits.append([-1]*10 + [1]*10 + [0]*10)
    
    # 6. Sparse: few nonzero
    for _ in range(5):
        c = [0]*n
        for _ in range(random.randint(2,8)):
            idx = random.randint(0, n-1)
            c[idx] = random.choice([-1,1])
        inits.append(c)
    
    # 7. Near-cancellation patterns
    # 1/1 - 1/2 - 1/3 + 1/6 ... try small harmonic identities
    inits.append([1, -1, -1, 0, 0, 1] + [0]*24)
    inits.append([1, -1, 0, -1, 0, 1, 0, 0, 0, 1] + [0]*20)
    
    # Simulated annealing parameters
    T_start = 1.0
    T_end = 1e-10
    cooling = 0.995
    max_iter = 500
    
    for init in inits:
        coeffs = init[:]
        current_val = eval_sum(coeffs)
        update_best(coeffs, current_val)
        
        T = T_start
        for it in range(max_iter):
            T *= cooling
            if T < T_end:
                break
            
            # Choose a move
            move_type = random.random()
            if move_type < 0.5:
                # Single flip
                i = random.randint(0, n-1)
                old = coeffs[i]
                options = [v for v in [-1,0,1] if v != old]
                new_val = random.choice(options)
                coeffs[i] = new_val
                new_sum = current_val + Fraction(new_val - old, i+1)
                delta = abs_frac(new_sum) - abs_frac(current_val)
                
                if new_sum != 0 and (delta < 0 or random.random() < (1.0 / (1.0 + T*1000))):
                    current_val = new_sum
                    update_best(coeffs, current_val)
                else:
                    coeffs[i] = old
            elif move_type < 0.8:
                # Pair swap
                i, j = random.sample(range(n), 2)
                old_i, old_j = coeffs[i], coeffs[j]
                new_i = random.choice([-1,0,1])
                new_j = random.choice([-1,0,1])
                if new_i == old_i and new_j == old_j:
                    continue
                coeffs[i] = new_i
                coeffs[j] = new_j
                new_sum = current_val + Fraction(new_i - old_i, i+1) + Fraction(new_j - old_j, j+1)
                delta = abs_frac(new_sum) - abs_frac(current_val)
                
                if new_sum != 0 and (delta < 0 or random.random() < (1.0 / (1.0 + T*1000))):
                    current_val = new_sum
                    update_best(coeffs, current_val)
                else:
                    coeffs[i] = old_i
                    coeffs[j] = old_j
            else:
                # Triple move
                i, j, k = random.sample(range(n), 3)
                old_i, old_j, old_k = coeffs[i], coeffs[j], coeffs[k]
                new_i = random.choice([-1,0,1])
                new_j = random.choice([-1,0,1])
                new_k = random.choice([-1,0,1])
                if new_i == old_i and new_j == old_j and new_k == old_k:
                    continue
                coeffs[i] = new_i
                coeffs[j] = new_j
                coeffs[k] = new_k
                new_sum = current_val + Fraction(new_i - old_i, i+1) + Fraction(new_j - old_j, j+1) + Fraction(new_k - old_k, k+1)
                delta = abs_frac(new_sum) - abs_frac(current_val)
                
                if new_sum != 0 and (delta < 0 or random.random() < (1.0 / (1.0 + T*1000))):
                    current_val = new_sum
                    update_best(coeffs, current_val)
                else:
                    coeffs[i] = old_i
                    coeffs[j] = old_j
                    coeffs[k] = old_k
    
    # Final local refinement on best solution
    if best_coeffs is not None:
        coeffs = best_coeffs[:]
        current_val = eval_sum(coeffs)
        
        # Greedy local search
        improved = True
        while improved:
            improved = False
            for i in range(n):
                old = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    coeffs[i] = new_val
                    new_sum = current_val + Fraction(new_val - old, i+1)
                    if new_sum == 0:
                        coeffs[i] = old
                        continue
                    new_abs = abs_frac(new_sum)
                    cur_abs = abs_frac(current_val)
                    if new_abs < cur_abs:
                        current_val = new_sum
                        improved = True
                    else:
                        coeffs[i] = old
            
            # Pair moves
            for i in range(n):
                for j in range(i+1, n):
                    old_i, old_j = coeffs[i], coeffs[j]
                    for new_i in [-1, 0, 1]:
                        for new_j in [-1, 0, 1]:
                            if new_i == old_i and new_j == old_j:
                                continue
                            coeffs[i] = new_i
                            coeffs[j] = new_j
                            new_sum = current_val + Fraction(new_i - old_i, i+1) + Fraction(new_j - old_j, j+1)
                            if new_sum == 0:
                                coeffs[i] = old_i
                                coeffs[j] = old_j
                                continue
                            new_abs = abs_frac(new_sum)
                            cur_abs = abs_frac(current_val)
                            if new_abs < cur_abs:
                                current_val = new_sum
                                improved = True
                            else:
                                coeffs[i] = old_i
                                coeffs[j] = old_j
        
        update_best(coeffs, current_val)
    
    if best_coeffs is None:
        return [0]*28 + [1, -1]
    
    # Verify
    final_val = eval_sum(best_coeffs)
    if final_val == 0:
        return [0]*28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
