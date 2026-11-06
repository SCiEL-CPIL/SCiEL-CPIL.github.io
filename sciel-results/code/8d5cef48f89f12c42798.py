import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    best_score = 21
    best_candidate = [1] * 12  # fallback: (1+x+...+x^11)^2 has 21 nonzero terms

    def compute_square_nonzeros(exps, coeffs):
        """Compute number of nonzero terms in P(x)^2."""
        square = defaultdict(int)
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                square[ei + ej] += 2 * ci * cj
        count = 0
        for val in square.values():
            if val != 0:
                count += 1
        return count

    def try_candidate(exps, coeffs):
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
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense

    def local_search(exps, coeffs, max_iters=5000):
        """
        Perform local search to minimize square nonzeros.
        Moves:
        1. Change coefficient of one term to another small integer.
        2. Change exponent of one term to another valid integer.
        """
        current_score = compute_square_nonzeros(exps, coeffs)
        exps = list(exps)
        coeffs = list(coeffs)
        
        # Allowed coefficient values
        coeff_vals = [-3, -2, -1, 1, 2, 3]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try changing coefficients
            for i in range(12):
                old_c = coeffs[i]
                for new_c in coeff_vals:
                    if new_c == old_c:
                        continue
                    coeffs[i] = new_c
                    new_score = compute_square_nonzeros(exps, coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        coeffs[i] = old_c
                if improved:
                    break
            
            # 2. Try changing exponents if coefficients didn't improve
            if not improved:
                i = random.randint(0, 11)
                old_e = exps[i]
                # Try a few random new exponents
                for _ in range(5):
                    new_e = random.randint(0, 40)
                    if new_e == old_e:
                        continue
                    if new_e in [exps[j] for j in range(12) if j != i]:
                        continue
                    
                    exps[i] = new_e
                    new_score = compute_square_nonzeros(exps, coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        exps[i] = old_e
                
                if not improved:
                    # If we got stuck, do a random restart of coefficients or exponents
                    # To escape local minima, let's randomly perturb one coefficient
                    i = random.randint(0, 11)
                    coeffs[i] = random.choice(coeff_vals)
                    current_score = compute_square_nonzeros(exps, coeffs)
            
            try_candidate(exps, coeffs)
            
        return exps, coeffs, current_score

    # Phase 1: Generate diverse starting points
    random.seed(42)
    starting_points = []
    
    # Arithmetic progressions
    for start in range(0, 10):
        for step in range(1, 8):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                # Try different coefficient patterns
                coeffs = [1] * 12
                starting_points.append((exps, coeffs))
                
                # Alternating signs
                coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(12)]
                starting_points.append((exps, coeffs_alt))
    
    # Symmetric sets
    for D in range(16, 41):
        for _ in range(5):
            left = sorted(random.sample(range(0, D // 2 + 1), 6))
            right = sorted(D - x for x in random.sample(range(0, D // 2 + 1), 6))
            exps = sorted(left + right)
            if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                coeffs = [1] * 12
                starting_points.append((exps, coeffs))
                coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(12)]
                starting_points.append((exps, coeffs_alt))
    
    # Random sets
    for _ in range(100):
        base = random.randint(0, 10)
        exps = sorted(set([base + random.randint(0, 30) for _ in range(15)]))
        if len(exps) >= 12:
            exps = exps[:12]
            if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
                starting_points.append((exps, coeffs))
    
    # Deduplicate
    seen = set()
    unique_starting_points = []
    for exps, coeffs in starting_points:
        key = (tuple(exps), tuple(coeffs))
        if key not in seen:
            seen.add(key)
            unique_starting_points.append((exps, coeffs))
    
    # Phase 2: Run Local Search from each starting point
    # Prioritize starting points with lower initial score
    unique_starting_points.sort(key=lambda x: compute_square_nonzeros(x[0], x[1]))
    
    max_ls_runs = 50
    ls_iters = 2000
    
    for i, (exps, coeffs) in enumerate(unique_starting_points[:max_ls_runs]):
        if time.time() > deadline - 5:
            break
        
        final_exps, final_coeffs, final_score = local_search(exps, coeffs, ls_iters)
        try_candidate(final_exps, final_coeffs)
        
        # If we found a very good score, stop early?
        if best_score <= 12:
            break

    return best_candidate

# EVOLVE-BLOCK-END
