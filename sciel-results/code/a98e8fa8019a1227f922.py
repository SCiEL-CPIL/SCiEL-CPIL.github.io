import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_poly = None
    best_score = float('inf')
    
    # Fixed random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    def evaluate(exponents, coeffs):
        """
        Returns the number of non-zero terms in the square of the polynomial.
        exponents: list of 12 integers
        coeffs: list of 12 integers
        """
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        
        # Calculate all pair sums and products
        # e_i + e_j for all i, j
        # c_i * c_j for all i, j
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        
        # Bincount to sum coefficients for each degree
        # Max degree is 40+40=80, so minlength=81 is sufficient
        # However, exponents can be up to 40, so max sum is 80.
        # If exponents are small, the array is smaller, but bincount handles it.
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=81)
        
        # Count non-zero entries
        return int(np.count_nonzero(result))

    def make_poly(exponents, coeffs):
        """Convert exponents and coeffs to a list representation."""
        max_deg = max(exponents)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exponents, coeffs):
            poly[e] = c
        return poly

    def validate(poly):
        if not poly or len(poly) > 41:
            return False
        if poly[-1] == 0:
            return False
        for c in poly:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly if c != 0) != 12:
            return False
        return True

    # Generate initial candidates
    def generate_initial_candidates():
        candidates = []
        
        # 1. Arithmetic Progressions
        for d in range(1, 7):
            for a in range(0, 41 - 11*d):
                exps = [a + i*d for i in range(12)]
                # Try a few coefficient patterns
                for _ in range(2):
                    coeffs = [random.choice([-1, 1]) for _ in range(12)]
                    candidates.append((exps, coeffs))
                    coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                    candidates.append((exps, coeffs))
        
        # 2. Symmetric Structures
        for center in range(6, 35):
            if random.random() < 0.5: # Limit number
                offsets = sorted(random.sample(range(1, 20), 6))
                exps = sorted([center - o for o in offsets] + [center + o for o in offsets])
                if 0 <= exps[0] and exps[-1] < 41 and len(set(exps)) == 12:
                    coeffs = [random.choice([-1, 1]) for _ in range(12)]
                    candidates.append((exps, coeffs))
        
        # 3. Random
        for _ in range(20):
            exps = sorted(random.sample(range(41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            candidates.append((exps, coeffs))
            
        return candidates

    def simulated_annealing(init_exps, init_coeffs, time_budget):
        """
        Runs SA starting from the given state.
        Returns the best (exps, coeffs, score) found.
        """
        cur_exps = list(init_exps)
        cur_coeffs = list(init_coeffs)
        cur_score = evaluate(cur_exps, cur_coeffs)
        
        best_exps = list(cur_exps)
        best_coeffs = list(cur_coeffs)
        best_score = cur_score
        
        T = 10.0
        cooling = 0.995
        
        steps = 0
        while time.time() < time_budget:
            # Generate a neighbor
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
                # Shift two exponents (maintain relative structure)
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
            
            # Validate
            if len(set(new_exps)) != 12:
                continue
            if any(e < 0 or e >= 41 for e in new_exps):
                continue
            if any(c == 0 or abs(c) > 64 for c in new_coeffs):
                continue
            
            new_score = evaluate(new_exps, new_coeffs)
            delta = new_score - cur_score
            
            if delta < 0:
                # Accept improvement
                cur_exps = new_exps
                cur_coeffs = new_coeffs
                cur_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_exps = list(new_exps)
                    best_coeffs = list(new_coeffs)
            else:
                # Accept worsening with probability exp(-delta / T)
                if random.random() < np.exp(-delta / T):
                    cur_exps = new_exps
                    cur_coeffs = new_coeffs
                    cur_score = new_score
            
            # Cool down
            T *= cooling
            steps += 1
            
            # If T gets too low, reheat slightly or break
            if T < 0.1:
                T = 1.0 # Reheat to escape deep local minima
                # Optionally reset to best? No, let it wander.
                
        return best_exps, best_coeffs, best_score

    # Main Loop
    # 1. Generate initial candidates
    candidates = generate_initial_candidates()
    
    # 2. Evaluate all initially to find a good starting point
    initial_best_exp = None
    initial_best_coef = None
    initial_best_score = float('inf')
    
    for exps, coeffs in candidates:
        score = evaluate(exps, coeffs)
        if score < initial_best_score:
            initial_best_score = score
            initial_best_exp = exps
            initial_best_coef = coeffs
            if score < best_score:
                best_score = score
                best_poly = make_poly(exps, coeffs)
                
    # 3. Run SA from multiple diverse starts
    # We have ~5s left for setup, so we have ~50s for SA.
    # We will run SA for a few seconds from the best initial, and maybe some others.
    
    # Strategy: Run SA from the single best initial candidate for most of the time,
    # but also run from a few other good candidates to avoid getting stuck.
    
    # Sort candidates by score
    scored_candidates = []
    for exps, coeffs in candidates:
        score = evaluate(exps, coeffs)
        scored_candidates.append((score, exps, coeffs))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 5 unique-ish candidates
    start_points = []
    seen = set()
    for score, exps, coeffs in scored_candidates:
        key = tuple(sorted(exps))
        if key not in seen:
            start_points.append((exps, coeffs))
            seen.add(key)
        if len(start_points) >= 5:
            break
            
    # Time allocation: 8 seconds per SA run
    for exps, coeffs in start_points:
        if time.time() > deadline - 2.0:
            break
            
        sa_end_time = min(time.time() + 8.0, deadline - 1.0)
        b_exp, b_coef, b_score = simulated_annealing(exps, coeffs, sa_end_time)
        
        if b_score < best_score:
            best_score = b_score
            best_poly = make_poly(b_exp, b_coef)
            
    # 4. Final check and return
    if best_poly is None or not validate(best_poly):
        # Fallback: A known valid polynomial
        # e.g., 1 + x^10 + ...
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        
    return best_poly

# For testing purposes, if run directly
if __name__ == "__main__":
    poly = construct()
    print(f"Polynomial: {poly}")
    print(f"Length: {len(poly)}")
    print(f"Non-zero terms: {sum(1 for c in poly if c != 0)}")
    # Calculate square
    import numpy as np
    p = np.array(poly)
    sq = np.convolve(p, p)
    print(f"Square non-zero terms: {np.count_nonzero(sq)}")
