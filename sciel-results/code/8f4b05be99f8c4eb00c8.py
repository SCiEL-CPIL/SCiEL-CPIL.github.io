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
        # Use a list for speed since max degree is 40, so square degree is 80.
        # Max index 80. List of size 81 is fast.
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
        count = 0
        for val in square:
            if val != 0:
                count += 1
        return count

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
        
        # Initial temperature
        T = 5.0
        cooling = 0.99999
        min_T = 0.01
        
        # Coefficient choices: small integers are best for cancellation
        # We bias towards -1, 1, -2, 2
        coeff_pool = [-1, 1, -2, 2, -3, 3, -4, 4]
        
        steps = 0
        while time.time() < time_limit and T > min_T:
            steps += 1
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.5:
                # Change coefficient
                i = random.randint(0, 11)
                old_c = coeffs[i]
                new_c = random.choice(coeff_pool)
                if new_c == old_c:
                    continue
                coeffs[i] = new_c
                new_score = compute_square_nonzeros(exps, coeffs)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    current_score = new_score
                else:
                    coeffs[i] = old_c
                    
            elif move_type < 0.9:
                # Move exponent
                i = random.randint(0, 11)
                old_exp = exps[i]
                # Try a small shift
                shift = random.choice([-2, -1, 1, 2])
                new_exp = old_exp + shift
                if new_exp < 0 or new_exp > 40:
                    continue
                # Check uniqueness
                if new_exp in [exps[j] for j in range(12) if j != i]:
                    continue
                
                exps[i] = new_exp
                # Keep sorted for canonical representation? 
                # Actually, for SA, order doesn't matter for the math, 
                # but we need to ensure we don't have duplicates.
                # We already checked uniqueness.
                
                new_score = compute_square_nonzeros(exps, coeffs)
                delta = new_score - current_score
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    current_score = new_score
                    # Re-sort to keep indices somewhat consistent for future moves?
                    # Not strictly necessary, but helps if we rely on index proximity.
                    # Let's keep it unsorted to allow "swaps" effectively via moves.
                else:
                    exps[i] = old_exp
                    
            else:
                # Swap two exponents (effectively a larger move if they are far apart)
                i, j = random.sample(range(12), 2)
                exps[i], exps[j] = exps[j], exps[i]
                new_score = compute_square_nonzeros(exps, coeffs)
                delta = new_score - current_score
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    current_score = new_score
                else:
                    exps[i], exps[j] = exps[j], exps[i]

            T *= cooling
            
            # Periodically check for global best
            if steps % 100 == 0:
                # Ensure exponents are valid and unique for the record
                if len(set(exps)) == 12 and 0 <= min(exps) and max(exps) <= 40:
                    if current_score < best_score:
                        # Sort for final output generation
                        paired = sorted(zip(exps, coeffs))
                        e_sorted = [p[0] for p in paired]
                        c_sorted = [p[1] for p in paired]
                        try_update(e_sorted, c_sorted)
                        # Reset local state to best found? No, SA continues from current.
                        # But if we found a global best, we might want to restart from it?
                        # Standard SA doesn't, but we can restart if we find a very good one.
                        if best_score <= 14:
                             # Found a very good one, restart SA from this best to refine?
                             # Or just let it run.
                             pass

        # Final check
        if len(set(exps)) == 12 and 0 <= min(exps) and max(exps) <= 40:
            paired = sorted(zip(exps, coeffs))
            e_sorted = [p[0] for p in paired]
            c_sorted = [p[1] for p in paired]
            try_update(e_sorted, c_sorted)

    # Generate diverse starting points
    starting_points = []
    
    # 1. Random sets
    for _ in range(50):
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        starting_points.append((exps, coeffs))
        
    # 2. Arithmetic Progressions
    for start in range(0, 10):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
                starting_points.append((exps, coeffs))
                coeffs2 = [1] * 12
                starting_points.append((exps, coeffs2))
                
    # 3. Symmetric sets
    for _ in range(50):
        # Pick 6 numbers from 0..20
        left = sorted(random.sample(range(0, 21), 6))
        right = sorted(40 - x for x in left)
        exps = sorted(left + right)
        if len(set(exps)) == 12:
             coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
             starting_points.append((exps, coeffs))
             coeffs2 = [1] * 6 + [-1] * 6
             starting_points.append((exps, coeffs2))

    # Run Simulated Annealing
    # We have ~85 seconds. Let's run 10 parallel-ish (sequential but fast) SA runs.
    # Each run gets ~8 seconds.
    time_per_run = 8.0
    num_runs = 10
    
    for i in range(num_runs):
        if time.time() > deadline - 5:
            break
        exps, coeffs = starting_points[i % len(starting_points)]
        # Add some noise to starting point
        if i > 0 and random.random() < 0.5:
             idx = random.randint(0, 11)
             exps = list(exps)
             coeffs = list(coeffs)
             # Perturb
             if exps[idx] + 1 <= 40 and (exps[idx] + 1 not in exps or exps[idx]+1 == exps[idx]): # basic check
                 # Actually just use the generated point as is, SA will explore.
                 pass
        
        simulate_annealing(exps, coeffs, seed=42 + i, time_limit=start_time + deadline - time.time() - (num_runs - i) * time_per_run if False else time.time() + time_per_run)
        
        # If we found a very low score, we can break early
        if best_score <= 13:
            break

    # Final Polish: Local search on the best candidate found
    if best_candidate and best_score < 21:
        max_deg = len(best_candidate) - 1
        best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
        best_coeffs = [best_candidate[i] for i in best_exps]
        
        # Simple hill climbing on the best found
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                old_c = best_coeffs[i]
                for new_c in range(-4, 5):
                    if new_c == 0 or new_c == old_c:
                        continue
                    best_coeffs[i] = new_c
                    s = compute_square_nonzeros(best_exps, best_coeffs)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_coeffs[i] = old_c
                
                # Try moving exponent
                old_e = best_exps[i]
                for delta in [-1, 1]:
                    new_e = old_e + delta
                    if new_e < 0 or new_e > 40:
                        continue
                    if new_e in [best_exps[j] for j in range(12) if j != i]:
                        continue
                    best_exps[i] = new_e
                    # Re-sort
                    paired = sorted(zip(best_exps, best_coeffs))
                    best_exps = [p[0] for p in paired]
                    best_coeffs = [p[1] for p in paired]
                    s = compute_square_nonzeros(best_exps, best_coeffs)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        # Revert
                        paired = sorted(zip([e if e != new_e else old_e for e in best_exps], best_coeffs))
                        best_exps = [p[0] for p in paired]
                        best_coeffs = [p[1] for p in paired]
            
            if best_score < 21:
                cand = get_dense(best_exps, best_coeffs)
                if cand:
                    best_candidate = cand

    return best_candidate

# EVOLVE-BLOCK-END
