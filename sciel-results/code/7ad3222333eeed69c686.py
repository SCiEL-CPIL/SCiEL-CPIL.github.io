import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    best_score = 21
    best_candidate = [1] * 12  # fallback: (1+x+...+x^11)^2 has 21 nonzero terms

    def compute_square_nonzeros(exps, coeffs):
        """Compute number of nonzero terms in P(x)^2 using a list for speed."""
        square = [0] * 81
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                square[ei + ej] += 2 * ci * cj
        return sum(1 for val in square if val != 0)

    def get_dense(exps, coeffs):
        max_deg = max(exps)
        dense = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            dense[e] = c
        if dense[-1] != 0:
            return dense
        return None

    def try_update(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        
        score = compute_square_nonzeros(exps, coeffs)
        if score < best_score:
            best_score = score
            cand = get_dense(exps, coeffs)
            if cand:
                best_candidate = cand

    def simulate_annealing(initial_exps, initial_coeffs, seed, time_limit):
        random.seed(seed)
        exps = list(initial_exps)
        coeffs = list(initial_coeffs)
        current_score = compute_square_nonzeros(exps, coeffs)
        
        T = 3.0
        cooling = 0.99998
        min_T = 0.005
        
        coeff_pool = [-1, 1, -2, 2, -3, 3]
        
        steps = 0
        stagnant_steps = 0
        local_best_score = current_score
        local_best_exps = list(exps)
        local_best_coeffs = list(coeffs)
        
        while time.time() < time_limit and T > min_T:
            steps += 1
            move_type = random.random()
            
            if move_type < 0.35:
                # Change coefficient - try to find cancellations
                i = random.randint(0, 11)
                old_c = coeffs[i]
                new_c = random.choice(coeff_pool)
                if new_c == old_c:
                    continue
                coeffs[i] = new_c
                new_score = compute_square_nonzeros(exps, coeffs)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < local_best_score:
                        local_best_score = current_score
                        local_best_exps = list(exps)
                        local_best_coeffs = list(coeffs)
                        stagnant_steps = 0
                    else:
                        stagnant_steps += 1
                else:
                    coeffs[i] = old_c
                    stagnant_steps += 1
                    
            elif move_type < 0.75:
                # Move exponent - try to create sum collisions
                i = random.randint(0, 11)
                old_exp = exps[i]
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_exp = old_exp + shift
                if new_exp < 0 or new_exp > 40:
                    continue
                if new_exp in exps:
                    if exps.index(new_exp) != i:
                        continue
                
                exps[i] = new_exp
                new_score = compute_square_nonzeros(exps, coeffs)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < local_best_score:
                        local_best_score = current_score
                        local_best_exps = list(exps)
                        local_best_coeffs = list(coeffs)
                        stagnant_steps = 0
                    else:
                        stagnant_steps += 1
                else:
                    exps[i] = old_exp
                    stagnant_steps += 1
                    
            else:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                exps[i], exps[j] = exps[j], exps[i]
                new_score = compute_square_nonzeros(exps, coeffs)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < local_best_score:
                        local_best_score = current_score
                        local_best_exps = list(exps)
                        local_best_coeffs = list(coeffs)
                        stagnant_steps = 0
                    else:
                        stagnant_steps += 1
                else:
                    exps[i], exps[j] = exps[j], exps[i]
                    stagnant_steps += 1

            T *= cooling
            
            if stagnant_steps > 500:
                exps = list(local_best_exps)
                coeffs = list(local_best_coeffs)
                current_score = local_best_score
                stagnant_steps = 0
                T = max(T, 1.0)  # Reheat

            if steps % 200 == 0:
                if len(set(exps)) == 12 and 0 <= min(exps) and max(exps) <= 40:
                    if current_score < best_score:
                        paired = sorted(zip(exps, coeffs))
                        e_sorted = [p[0] for p in paired]
                        c_sorted = [p[1] for p in paired]
                        try_update(e_sorted, c_sorted)

        if len(set(exps)) == 12 and 0 <= min(exps) and max(exps) <= 40:
            paired = sorted(zip(exps, coeffs))
            e_sorted = [p[0] for p in paired]
            c_sorted = [p[1] for p in paired]
            try_update(e_sorted, c_sorted)

    # Generate diverse starting points
    starting_points = []
    
    # 1. Random sets with ±1 coefficients
    for _ in range(30):
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        starting_points.append((exps, coeffs))
        
    # 2. Arithmetic Progressions
    for start in range(0, 10):
        for step in range(1, 6):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
                starting_points.append((exps, coeffs))
                coeffs2 = [1] * 12
                starting_points.append((exps, coeffs2))
                coeffs3 = [(-1)**i for i in range(12)]
                starting_points.append((exps, coeffs3))
                
    # 3. Symmetric sets around 20
    for _ in range(40):
        left = sorted(random.sample(range(0, 21), 6))
        right = sorted(40 - x for x in left)
        exps = sorted(left + right)
        if len(set(exps)) == 12:
            coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            starting_points.append((exps, coeffs))
            coeffs2 = [1] * 6 + [-1] * 6
            starting_points.append((exps, coeffs2))
            coeffs3 = [-1] * 6 + [1] * 6
            starting_points.append((exps, coeffs3))
    
    # 4. Geometric-like progressions
    for base in [2, 3, 4]:
        exps = [min(40, int(base ** (i/3))) for i in range(12)]
        seen = set()
        unique_exps = []
        for e in exps:
            if e not in seen and e <= 40:
                seen.add(e)
                unique_exps.append(e)
        if len(unique_exps) >= 12:
            unique_exps = unique_exps[:12]
            coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            starting_points.append((sorted(unique_exps), coeffs))

    # Run Simulated Annealing
    time_per_run = 4.0
    num_runs = 10
    
    for i in range(num_runs):
        if time.time() > deadline - 5:
            break
        exps, coeffs = starting_points[i % len(starting_points)]
        
        simulate_annealing(exps, coeffs, seed=42 + i * 7, time_limit=time.time() + time_per_run)
        
        if best_score <= 12:
            break

    # Light Final Polish
    if best_candidate and best_score < 21:
        best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
        best_coeffs = [best_candidate[i] for i in best_exps]
        
        for i in range(12):
            old_c = best_coeffs[i]
            for new_c in [-3, -2, -1, 1, 2, 3]:
                if new_c == 0 or new_c == old_c:
                    continue
                best_coeffs[i] = new_c
                s = compute_square_nonzeros(best_exps, best_coeffs)
                if s < best_score:
                    best_score = s
                else:
                    best_coeffs[i] = old_c
            
            old_e = best_exps[i]
            for delta in [-1, 1]:
                new_e = old_e + delta
                if new_e < 0 or new_e > 40:
                    continue
                if new_e in best_exps:
                    if best_exps.index(new_e) != i:
                        continue
                
                best_exps[i] = new_e
                paired = sorted(zip(best_exps, best_coeffs))
                best_exps = [p[0] for p in paired]
                best_coeffs = [p[1] for p in paired]
                s = compute_square_nonzeros(best_exps, best_coeffs)
                if s < best_score:
                    best_score = s
                else:
                    paired = sorted(zip([e if e != new_e else old_e for e in best_exps], best_coeffs))
                    best_exps = [p[0] for p in paired]
                    best_coeffs = [p[1] for p in paired]
        
        if best_score < 21:
            cand = get_dense(best_exps, best_coeffs)
            if cand:
                best_candidate = cand

    return best_candidate

# EVOLVE-BLOCK-END
