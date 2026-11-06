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

    def optimize_local(exps, coeffs):
        """
        Deterministic local search.
        1. Hill climb on coefficients.
        2. Try moving exponents to nearby positions, then hill climb coefficients again.
        """
        exps = list(exps)
        coeffs = list(coeffs)
        current_score = compute_square_nonzeros(exps, coeffs)
        
        # Coefficient values to try
        coeff_values = [-4, -3, -2, -1, 1, 2, 3, 4]
        
        # Phase 1: Coefficient Hill Climbing
        improved = True
        while improved:
            improved = False
            for i in range(12):
                old_c = coeffs[i]
                for new_c in coeff_values:
                    if new_c == old_c:
                        continue
                    coeffs[i] = new_c
                    new_score = compute_square_nonzeros(exps, coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        old_c = new_c
                    else:
                        coeffs[i] = old_c
                if time.time() > deadline:
                    return exps, coeffs, current_score
        
        # Phase 2: Exponent Perturbation + Coefficient Refinement
        # Try moving each exponent to a nearby valid position
        for i in range(12):
            if time.time() > deadline:
                break
            
            old_exp = exps[i]
            # Try shifting by -1, 1, -2, 2
            for delta in [-1, 1, -2, 2]:
                new_exp = old_exp + delta
                if new_exp < 0 or new_exp > 40:
                    continue
                # Check uniqueness
                if new_exp in [exps[j] for j in range(12) if j != i]:
                    continue
                
                # Apply move
                exps[i] = new_exp
                # Sort to maintain canonical order (exps and coeffs move together)
                paired = sorted(zip(exps, coeffs))
                exps_new = [p[0] for p in paired]
                coeffs_new = [p[1] for p in paired]
                
                # Re-run coefficient hill climbing on this new structure
                # We do a simplified version: just one pass of hill climbing
                # to see if it improves. If it does, we keep it.
                c_score = compute_square_nonzeros(exps_new, coeffs_new)
                
                # Quick hill climb on coeffs for this new exp structure
                temp_exps = exps_new
                temp_coeffs = list(coeffs_new)
                temp_score = c_score
                c_improved = True
                while c_improved:
                    c_improved = False
                    for k in range(12):
                        old_c = temp_coeffs[k]
                        for new_c in coeff_values:
                            if new_c == old_c:
                                continue
                            temp_coeffs[k] = new_c
                            s = compute_square_nonzeros(temp_exps, temp_coeffs)
                            if s < temp_score:
                                temp_score = s
                                c_improved = True
                                old_c = new_c
                            else:
                                temp_coeffs[k] = old_c
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                
                if temp_score < current_score:
                    current_score = temp_score
                    exps = temp_exps
                    coeffs = temp_coeffs
                    # If we improved, we should continue searching from this new state
                    # Break the delta loop and restart the exponent perturbation phase?
                    # For simplicity, we just accept and continue to next i
                    # But since exps changed, indices might have shifted.
                    # To be safe, we break out and restart the whole optimize_local?
                    # No, that's expensive. We just break the delta loop.
                    break
            if time.time() > deadline:
                break
            
            # Re-sort after any changes to keep exps sorted for next iterations
            paired = sorted(zip(exps, coeffs))
            exps = [p[0] for p in paired]
            coeffs = [p[1] for p in paired]
            
        return exps, coeffs, current_score

    # Phase 1: Generate diverse starting points
    random.seed(42)
    starting_points = []
    
    # 1. Arithmetic Progressions (APs)
    for start in range(0, 25):
        for step in range(1, 8):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                coeffs = [1] * 12
                starting_points.append((exps, coeffs))
                
                # Also try alternating signs
                coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(12)]
                starting_points.append((exps, coeffs_alt))
    
    # 2. Symmetric Sets around a center
    for D in range(20, 41):
        for _ in range(10):
            # Pick 6 numbers from [0, D//2]
            # Ensure they are distinct
            candidates = list(range(0, D // 2 + 1))
            if len(candidates) < 6:
                continue
            left = sorted(random.sample(candidates, 6))
            right = sorted(D - x for x in left)
            exps = sorted(left + right)
            if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                coeffs = [1] * 12
                starting_points.append((exps, coeffs))
                
                # Try symmetric signs
                # e.g., + - + - ... on left, mirror on right?
                # Or just random signs
                coeffs_rand = [random.choice([-1, 1]) for _ in range(12)]
                starting_points.append((exps, coeffs_rand))

    # 3. Random Sets with small span
    for _ in range(100):
        base = random.randint(0, 15)
        # Generate 12 distinct numbers in [base, base+25]
        span = random.randint(12, 25)
        candidates = list(range(base, base + span + 1))
        if len(candidates) < 12:
            continue
        exps = sorted(random.sample(candidates, 12))
        if max(exps) <= 40:
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            starting_points.append((exps, coeffs))

    # 4. Specific "Cancellations" patterns
    # e.g., (1+x+x^2+x^3)^2 has 7 terms.
    # Let's try to find sets that decompose into small clusters.
    for _ in range(50):
        # Pick a random base
        base = random.randint(0, 20)
        # Pick 4 offsets
        offs = random.sample(range(0, 10), 4)
        exps1 = [base + o for o in offs]
        # Pick another base and 4 offsets
        base2 = base + random.randint(5, 15)
        offs2 = random.sample(range(0, 10), 4)
        exps2 = [base2 + o for o in offs2]
        exps = sorted(exps1 + exps2)
        if len(set(exps)) == 12 and max(exps) <= 40:
            # Coeffs: 1 for first 4, 1 for next 4
            coeffs = [1]*4 + [1]*4
            starting_points.append((exps, coeffs))
            
            # Try -1 for second group
            coeffs2 = [1]*4 + [-1]*4
            starting_points.append((exps, coeffs2))

    # Deduplicate starting points
    seen = set()
    unique_starting_points = []
    for exps, coeffs in starting_points:
        key = (tuple(exps), tuple(coeffs))
        if key not in seen:
            seen.add(key)
            unique_starting_points.append((exps, coeffs))
    
    # Sort by initial score to prioritize promising structures
    unique_starting_points.sort(key=lambda x: compute_square_nonzeros(x[0], x[1]))
    
    # Phase 2: Run Local Search on top starting points
    # We run local search on a limited number of best candidates
    max_runs = 200
    for i, (exps, coeffs) in enumerate(unique_starting_points[:max_runs]):
        if time.time() > deadline - 2:
            break
        
        final_exps, final_coeffs, final_score = optimize_local(exps, coeffs)
        try_candidate(final_exps, final_coeffs)
        
        # If we found a very good score, we might stop early or break
        if best_score <= 14:
            break

    # Phase 3: Final Polish on the best candidate
    if best_score < 21 and time.time() < deadline - 5:
        if best_candidate:
            max_deg = len(best_candidate) - 1
            best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
            best_coeffs = [best_candidate[i] for i in best_exps]
            
            # Run one more aggressive local search
            # This time, allow larger coefficient changes
            coeff_values_large = [-64, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 64]
            
            improved = True
            while improved:
                improved = False
                for i in range(12):
                    old = best_coeffs[i]
                    for new_val in coeff_values_large:
                        if new_val == old:
                            continue
                        best_coeffs[i] = new_val
                        s = compute_square_nonzeros(best_exps, best_coeffs)
                        if s < best_score:
                            best_score = s
                            improved = True
                        else:
                            best_coeffs[i] = old
                if time.time() > deadline:
                    break
            
            # Update candidate
            if best_score < 21:
                max_deg = max(best_exps)
                dense = [0] * (max_deg + 1)
                for e, c in zip(best_exps, best_coeffs):
                    dense[e] = c
                if dense[-1] != 0:
                    best_candidate = dense

    return best_candidate

# EVOLVE-BLOCK-END
