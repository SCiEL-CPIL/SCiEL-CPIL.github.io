import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s kill

    best_count = float('inf')
    best_coeffs = None
    best_exponents = None

    def eval_square(exponents, coeffs):
        """
        exponents: list of 12 ints
        coeffs: list of 12 ints
        Returns: (count_nonzero, square_list)
        """
        max_e = max(exponents)
        square = [0] * (2 * max_e + 1)
        n = len(exponents)
        
        # Optimization: Sort by exponent to potentially improve cache locality? 
        # No, we need specific pairs. 
        # 12 terms -> 12*13/2 = 78 pairs. Very fast.
        for i in range(n):
            ei = exponents[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exponents[j]
                cj = coeffs[j]
                square[ei + ej] += 2 * ci * cj
        
        count = sum(1 for x in square if x != 0)
        return count, square

    def to_full(exponents, coeffs):
        max_e = max(exponents)
        if max_e >= 41:
            return None
        full = [0] * (max_e + 1)
        for e, c in zip(exponents, coeffs):
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

    def try_update(exponents, coeffs, count):
        nonlocal best_count, best_coeffs, best_exponents
        if count < best_count:
            full = to_full(exponents, coeffs)
            if full is not None:
                best_count = count
                best_coeffs = full
                best_exponents = exponents[:]
                return True
        return False

    # =====================================================================
    # Phase 1: Brute Force Signs for Consecutive Exponents
    # =====================================================================
    base_exps = list(range(12))
    for bits in range(1 << 12):
        if time.time() > deadline - 40:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        count, _ = eval_square(base_exps, coeffs)
        try_update(base_exps, coeffs, count)
        if best_count == 12:
            return best_coeffs

    # =====================================================================
    # Phase 2: Generate Structured Seeds
    # =====================================================================
    seeds = []
    
    # 1. Consecutive exponents, various small coefficients
    vals_small = [-2, -1, 1, 2]
    for _ in range(100):
        coeffs = [random.choice(vals_small) for _ in range(12)]
        seeds.append((list(range(12)), coeffs))

    # 2. Geometric Progressions
    for base in [1, 2, 3]:
        exps = [i * base for i in range(12)]
        if max(exps) < 41:
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            seeds.append((exps, coeffs))

    # 3. Palindromic Structures (Symmetric coefficients)
    # P(x) = sum c_i x^i, if c_i = c_{11-i}, P(x) is palindromic.
    # Square of palindromic poly is also palindromic.
    for _ in range(200):
        half_coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        coeffs = half_coeffs + half_coeffs[::-1]
        exps = list(range(12))
        seeds.append((exps, coeffs))
        
        # Try with gaps
        exps_gap = [i * 2 for i in range(12)]
        if max(exps_gap) < 41:
            seeds.append((exps_gap, coeffs))

    # 4. Random Exponents with Small Gaps
    for _ in range(500):
        exps = [0]
        for i in range(11):
            last = exps[-1]
            gap = random.choice([1, 1, 1, 2, 3])
            new_e = last + gap
            if new_e > 40:
                break
            exps.append(new_e)
        if len(exps) == 12:
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            seeds.append((exps, coeffs))

    # Evaluate all seeds
    for exps, coeffs in seeds:
        if time.time() > deadline - 30:
            break
        count, _ = eval_square(exps, coeffs)
        try_update(exps, coeffs, count)
        if best_count == 12:
            return best_coeffs

    # =====================================================================
    # Phase 3: Simulated Annealing
    # =====================================================================
    if best_exponents is None:
        best_exponents = list(range(12))
        best_coeffs = [1]*12
        best_count, _ = eval_square(best_exponents, best_coeffs)

    current_exps = best_exponents[:]
    current_coeffs = best_coeffs[:]
    current_count = best_count

    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    moves_made = 0
    
    # Define move generators
    coeff_vals = [-8, -4, -2, -1, 1, 2, 4, 8]
    
    def make_move(exps, coeffs):
        move_type = random.random()
        
        if move_type < 0.4:
            # Change a coefficient
            idx = random.randint(0, 11)
            old_c = coeffs[idx]
            new_c = random.choice(coeff_vals)
            if new_c != old_c and abs(new_c) <= 64:
                return exps, coeffs[:idx] + [new_c] + coeffs[idx+1:]
        
        elif move_type < 0.8:
            # Shift an exponent
            idx = random.randint(0, 11)
            e = exps[idx]
            shift = random.choice([-2, -1, 1, 2])
            new_e = e + shift
            if 0 <= new_e <= 40:
                # Check distinctness
                if new_e not in exps:
                    new_exps = exps[:]
                    new_exps[idx] = new_e
                    return new_exps, coeffs
        
        else:
            # Swap two exponents (permutation)
            i, j = random.sample(range(12), 2)
            if exps[i] != exps[j]:
                new_exps = exps[:]
                new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
                return new_exps, coeffs

    while time.time() < deadline - 10:
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        new_exps, new_coeffs = make_move(current_exps, current_coeffs)
        
        if new_exps is None or new_coeffs is None:
            continue
            
        # Validate
        if len(set(new_exps)) != 12:
            continue
        if any(abs(c) > 64 or c == 0 for c in new_coeffs):
            continue
        if max(new_exps) > 40:
            continue

        new_count, _ = eval_square(new_exps, new_coeffs)
        delta = new_count - current_count
        
        if delta <= 0:
            current_exps = new_exps
            current_coeffs = new_coeffs
            current_count = new_count
            try_update(current_exps, current_coeffs, current_count)
            if best_count == 12:
                return best_coeffs
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                current_exps = new_exps
                current_coeffs = new_coeffs
                current_count = new_count

    # =====================================================================
    # Phase 4: Final Hill Climb Polish
    # =====================================================================
    # Use the best found so far
    if best_exponents is not None:
        current_exps = best_exponents[:]
        current_coeffs = best_coeffs[:]
        current_count = best_count
        
        polish_deadline = deadline - 2.0
        no_improve = 0
        
        while time.time() < polish_deadline:
            improved = False
            # Try all single coefficient changes
            for idx in range(12):
                old_c = current_coeffs[idx]
                for new_c in coeff_vals:
                    if new_c == old_c:
                        continue
                    new_coeffs = current_coeffs[:]
                    new_coeffs[idx] = new_c
                    c_count, _ = eval_square(current_exps, new_coeffs)
                    if c_count < current_count:
                        current_coeffs = new_coeffs
                        current_count = c_count
                        improved = True
                        break
                if improved:
                    break
            
            # Try exponent shifts
            if not improved:
                for idx in range(12):
                    e = current_exps[idx]
                    for shift in [-1, 1]:
                        new_e = e + shift
                        if 0 <= new_e <= 40:
                            if new_e not in current_exps:
                                new_exps = current_exps[:]
                                new_exps[idx] = new_e
                                c_count, _ = eval_square(new_exps, current_coeffs)
                                if c_count < current_count:
                                    current_exps = new_exps
                                    current_count = c_count
                                    improved = True
                                    break
                    if improved:
                        break
            
            if improved:
                no_improve = 0
                try_update(current_exps, current_coeffs, current_count)
                if best_count == 12:
                    return best_coeffs
            else:
                no_improve += 1
                if no_improve > 50:
                    break

    # Fallback
    if best_coeffs is None:
        best_coeffs = [1]*12

    return best_coeffs
