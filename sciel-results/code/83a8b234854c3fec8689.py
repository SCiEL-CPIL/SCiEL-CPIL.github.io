import time
import random
import numpy as np

def _square_nonzero_count(poly):
    """
    Given poly as a list of (exponent, coeff) pairs,
    return the number of nonzero terms in poly^2.
    """
    if not poly:
        return 0
    max_deg = max(exp for exp, _ in poly)
    sq_deg = 2 * max_deg
    # Build dense coefficient vector
    vec = np.zeros(sq_deg + 1, dtype=np.int64)
    for exp, coeff in poly:
        vec[exp] = coeff
    # Convolve
    result = np.convolve(vec, vec)
    return int(np.count_nonzero(result))

def _validate(poly):
    """Check if poly meets all constraints."""
    if len(poly) != 12:
        return False
    if any(c == 0 for _, c in poly):
        return False
    if any(c < -64 or c > 64 for _, c in poly):
        return False
    if any(e < 0 or not isinstance(e, int) for e, _ in poly):
        return False
    if any(e > 40 for e, _ in poly):
        return False
    # Check for duplicate exponents
    exps = [e for e, _ in poly]
    if len(set(exps)) != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    best_poly = None
    best_score = float('inf')
    
    # Initial candidate: consecutive exponents with all 1s
    seed_poly = [(i, 1) for i in range(12)]
    seed_score = _square_nonzero_count(seed_poly)
    if _validate(seed_poly):
        best_poly = seed_poly
        best_score = seed_score
    
    # Simulated annealing with structural moves
    random.seed(42)
    np.random.seed(42)
    
    def _evaluate(poly):
        return _square_nonzero_count(poly)
    
    def _sa_optimize(initial_poly, max_iters=50000):
        """Simulated annealing to minimize nonzero terms in square."""
        current = list(initial_poly)
        current_score = _evaluate(current)
        best = list(current)
        best_score = current_score
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.1
        T = T_start
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Cooled temperature
            progress = it / max_iters
            T = T_start * (T_end / T_start) ** progress
            
            # Generate a neighbor
            move_type = random.random()
            neighbor = list(current)
            
            if move_type < 0.5:
                # Change one exponent
                i = random.randint(0, 11)
                old_exp = neighbor[i][0]
                new_exp = random.randint(0, 40)
                if new_exp == old_exp:
                    continue
                # Check for collision
                if any(neighbor[j][0] == new_exp for j in range(12) if j != i):
                    continue
                neighbor[i] = (new_exp, neighbor[i][1])
            elif move_type < 0.8:
                # Change one coefficient
                i = random.randint(0, 11)
                new_coeff = random.choice([-64, -16, -4, -2, -1, 1, 2, 4, 16, 64])
                if new_coeff == 0 or new_coeff == neighbor[i][1]:
                    continue
                neighbor[i] = (neighbor[i][0], new_coeff)
            else:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                ei, ci = neighbor[i]
                ej, cj = neighbor[j]
                neighbor[i] = (ej, ci)
                neighbor[j] = (ei, cj)
            
            if not _validate(neighbor):
                continue
            
            neighbor_score = _evaluate(neighbor)
            delta = neighbor_score - current_score
            
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current = neighbor
                current_score = neighbor_score
                
                if current_score < best_score:
                    best_score = current_score
                    best = list(current)
            
            if best_score <= 12:
                break
        
        return best, best_score
    
    # Multiple random restarts with simulated annealing
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse initial polynomials
        if r == 0:
            # Start from seed
            initial = list(seed_poly)
        elif r == 1:
            # Arithmetic progression with signs
            step = random.randint(1, 3)
            start = random.randint(0, 40 - 11 * step)
            signs = [random.choice([-1, 1]) for _ in range(12)]
            initial = [(start + i * step, signs[i]) for i in range(12)]
        elif r == 2:
            # Two clusters
            exps = sorted(random.sample(range(0, 41), 12))
            # Make first 6 close together, last 6 close together
            initial = [(exps[i], random.choice([-1, 1])) for i in range(12)]
        else:
            # Random
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-2, -1, 1, 2, 3]) for _ in range(12)]
            initial = list(zip(exps, coeffs))
        
        if not _validate(initial):
            continue
        
        score = _evaluate(initial)
        if score < best_score:
            best_score = score
            best_poly = list(initial)
        
        # Run simulated annealing
        local_poly, local_score = _sa_optimize(initial, max_iters=30000)
        if local_score < best_score:
            best_score = local_score
            best_poly = local_poly
        
        if best_score <= 12:
            break
    
    # Final local improvement on best found
    if best_poly is not None and time.time() < deadline:
        final_poly, final_score = _sa_optimize(best_poly, max_iters=20000)
        if final_score < best_score:
            best_score = final_score
            best_poly = final_poly
    
    # Convert to coefficient list
    if best_poly is None:
        best_poly = seed_poly
    
    max_deg = max(e for e, _ in best_poly)
    coeff_list = [0] * (max_deg + 1)
    for exp, coeff in best_poly:
        coeff_list[exp] = coeff
    
    return coeff_list
