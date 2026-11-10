import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    def evaluate(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) != 12:
            return False
        if coeffs[-1] == 0:
            return False
        if coeffs[0] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    random.seed(42)

    # Precompute pair structure for fast square evaluation
    # sq[k] = sum_{i+j=k} c[i]*c[j]
    # For k in 0..22, the pairs are (i, k-i) for valid i
    
    # Build a lookup: for each k, list of (i, j) pairs where i+j=k, 0<=i,j<12
    pair_to_degree = {}
    for i in range(12):
        for j in range(12):
            k = i + j
            if k not in pair_to_degree:
                pair_to_degree[k] = []
            pair_to_degree[k].append((i, j))

    def compute_square(coeffs):
        """Compute square coefficients efficiently."""
        sq = [0] * 23
        for i in range(12):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(12):
                cj = coeffs[j]
                if cj == 0:
                    continue
                sq[i + j] += ci * cj
        return sq

    def count_sq_nonzero(sq):
        return sum(1 for x in sq if x != 0)

    def update_best(coeffs, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            best_poly = coeffs.copy()

    # --- Phase 1: Structured search with self-similar patterns ---
    # Try patterns of the form P(x) = A(x) + x^k * A(x) where A has 6 terms
    # This creates collision structure that can lead to cancellations
    
    # Also try arithmetic progression supports
    # For 12 terms with consecutive exponents 0..11, try various sign/magnitude patterns
    
    # Key insight: try to make the square have a specific sparse pattern
    # e.g., square has terms only at even degrees (like (a0 + a2*x^2 + ...)^2)
    # This requires odd-degree coefficients to be zero, but we need 12 nonzero terms
    
    # Better: try to find coefficients where many cross-terms cancel
    # For example, if c[i] = -c[j] for certain pairs, then c[i]*c[j] + c[j]*c[i] = -2*c[i]^2
    # But we need the SUM to be zero, not individual terms
    
    # Let's try a targeted approach:
    # 1. Start with known good structures
    # 2. Use simulated annealing with structured moves
    # 3. Use local search with collision-aware moves

    # Initial candidates: various structured patterns
    initial_candidates = []
    
    # All ones
    initial_candidates.append([1]*12)
    
    # Alternating signs
    initial_candidates.append([(-1)**i for i in range(12)])
    
    # Block patterns
    initial_candidates.append([1,1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    initial_candidates.append([1,1,1,1,-1,-1,-1,-1,-1,-1,1,1])
    initial_candidates.append([1,1,-1,-1,1,1,-1,-1,1,1,-1,-1])
    
    # Small integer variations
    for pattern in [[1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1],
                    [1,2,1,1,1,1,1,1,1,1,1,1],
                    [1,1,2,1,1,1,1,1,1,1,2,1],
                    [1,1,1,2,1,1,1,1,2,1,1,1]]:
        initial_candidates.append(pattern.copy())
        initial_candidates.append([-x for x in pattern])
    
    # Evaluate initial candidates
    for cand in initial_candidates:
        if not validate(cand):
            continue
        score = evaluate(cand)
        update_best(cand, score)

    # --- Phase 2: Simulated Annealing with structured moves ---
    def simulated_annealing(num_restarts=15, iters_per_restart=30000):
        nonlocal best_score, best_poly
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                return
            
            # Initialize
            if best_poly is not None and random.random() < 0.5:
                curr = best_poly.copy()
                # Perturb
                for _ in range(random.randint(1, 4)):
                    idx = random.randint(0, 11)
                    curr[idx] = random.choice([-3, -2, -1, 1, 2, 3])
                    if curr[idx] == 0:
                        curr[idx] = 1
            else:
                curr = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            
            if not validate(curr):
                curr = [1]*12
            
            curr_sq = compute_square(curr)
            curr_score = count_sq_nonzero(curr_sq)
            
            if curr_score < best_score:
                best_score = curr_score
                best_poly = curr.copy()
            
            T = 3.0
            T_min = 0.05
            cooling_rate = 0.9995
            
            for i in range(iters_per_restart):
                if time.time() > deadline - 2:
                    return
                
                # Choose move type
                move_type = random.random()
                
                if move_type < 0.6:
                    # Single coefficient change
                    idx = random.randint(0, 11)
                    old_val = curr[idx]
                    if random.random() < 0.7:
                        new_val = random.choice([-3, -2, -1, 1, 2, 3])
                    else:
                        new_val = random.randint(-64, 64)
                    if new_val == 0:
                        new_val = 1
                    if new_val == old_val:
                        continue
                    curr[idx] = new_val
                    
                elif move_type < 0.8:
                    # Swap two coefficients
                    i1, i2 = random.sample(range(12), 2)
                    curr[i1], curr[i2] = curr[i2], curr[i1]
                    old_i1, old_i2 = curr[i2], curr[i1]
                    
                elif move_type < 0.9:
                    # Sign flip
                    idx = random.randint(0, 11)
                    curr[idx] = -curr[idx]
                    old_val = -curr[idx]
                    
                else:
                    # Block flip: flip signs of a contiguous block
                    start = random.randint(0, 10)
                    length = random.randint(1, 3)
                    end = min(start + length, 12)
                    for idx in range(start, end):
                        curr[idx] = -curr[idx]
                    old_block = [-curr[idx] for idx in range(start, end)]
                
                # Compute new square
                new_sq = compute_square(curr)
                new_score = count_sq_nonzero(new_sq)
                
                delta = new_score - curr_score
                
                # Acceptance criterion
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    curr_score = new_score
                    curr_sq = new_sq
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                else:
                    # Revert
                    if move_type < 0.6:
                        curr[idx] = old_val
                    elif move_type < 0.8:
                        curr[i1], curr[i2] = old_i1, old_i2
                    elif move_type < 0.9:
                        curr[idx] = old_val
                    else:
                        for idx in range(start, end):
                            curr[idx] = -curr[idx]
                
                T *= cooling_rate
                if T < T_min:
                    T = T_min

    simulated_annealing(num_restarts=10, iters_per_restart=50000)

    # --- Phase 3: Targeted local search focusing on collision cancellation ---
    if best_poly is not None and time.time() < deadline - 5:
        curr = best_poly.copy()
        curr_score = best_score
        
        # For each degree k in the square, try to zero it out
        # by adjusting one or two coefficients
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            curr_sq = compute_square(curr)
            
            # Find nonzero terms in the square
            nonzero_degrees = [k for k in range(23) if curr_sq[k] != 0]
            
            if not nonzero_degrees:
                break
            
            # Pick a random nonzero degree to try to zero out
            k = random.choice(nonzero_degrees)
            
            # For each coefficient i that contributes to sq[k], try adjusting it
            # sq[k] = sum_{i+j=k} c[i]*c[j]
            # If we change c[i] to c[i]+d, the change in sq[k] is:
            # d * (sum_{j: i+j=k, j!=i} c[j]) + d^2 * (1 if 2i==k else 0)
            
            found_improvement = False
            for i in range(12):
                j = k - i
                if j < 0 or j >= 12:
                    continue
                
                # Compute the coefficient of c[i] in sq[k]
                # sq[k] = c[i]*c[j] + (other terms not involving c[i])
                # If i != j: sq[k] contains 2*c[i]*c[j] (from (i,j) and (j,i))
                # If i == j: sq[k] contains c[i]^2
                
                other_sum = 0
                for l in range(12):
                    m = k - l
                    if m < 0 or m >= 12:
                        continue
                    if l == i and m == i:
                        continue  # This is c[i]^2
                    if l == i or m == i:
                        # This term involves c[i]
                        if l == i:
                            other_sum += curr[m]
                        elif m == i:
                            other_sum += curr[l]
                
                # sq[k] = A*c[i]^2 + B*c[i] + C
                # where A = 1 if 2i==k else 0
                #       B = sum of c[j] for j != i with i+j=k (counting multiplicity)
                #       C = sq[k] - A*c[i]^2 - B*c[i]
                
                A = 1 if (2*i == k) else 0
                B = 0
                for l in range(12):
                    m = k - l
                    if m < 0 or m >= 12:
                        continue
                    if l == i and m == i:
                        continue
                    if l == i:
                        B += curr[m]
                    elif m == i:
                        B += curr[l]
                
                C = curr_sq[k] - A * curr[i] * curr[i] - B * curr[i]
                
                # We want A*c[i]^2 + B*c[i] + C = 0
                # If A == 0: c[i] = -C/B (if B != 0)
                # If A != 0: solve quadratic
                
                if A == 0:
                    if B != 0 and C % B == 0:
                        new_val = -C // B
                        if -64 <= new_val <= 64 and new_val != 0 and new_val != curr[i]:
                            curr[i] = new_val
                            new_sq = compute_square(curr)
                            new_score = count_sq_nonzero(new_sq)
                            if new_score < curr_score:
                                curr_score = new_score
                                if curr_score < best_score:
                                    best_score = curr_score
                                    best_poly = curr.copy()
                                no_improve = 0
                                found_improvement = True
                                break
                else:
                    # A*c[i]^2 + B*c[i] + C = 0
                    # Discriminant D = B^2 - 4*A*C
                    D = B*B - 4*A*C
                    if D >= 0:
                        sqrt_D = int(np.sqrt(D))
                        if sqrt_D * sqrt_D == D:
                            # Integer solutions
                            for sign in [1, -1]:
                                num = -B + sign * sqrt_D
                                if num % (2*A) == 0:
                                    new_val = num // (2*A)
                                    if -64 <= new_val <= 64 and new_val != 0 and new_val != curr[i]:
                                        curr[i] = new_val
                                        new_sq = compute_square(curr)
                                        new_score = count_sq_nonzero(new_sq)
                                        if new_score < curr_score:
                                            curr_score = new_score
                                            if curr_score < best_score:
                                                best_score = curr_score
                                                best_poly = curr.copy()
                                            no_improve = 0
                                            found_improvement = True
                                            break
                            if found_improvement:
                                break
                
                # Also try nearby values
                for dv in [-2, -1, 1, 2]:
                    new_val = curr[i] + dv
                    if -64 <= new_val <= 64 and new_val != 0:
                        curr[i] = new_val
                        new_sq = compute_square(curr)
                        new_score = count_sq_nonzero(new_sq)
                        if new_score < curr_score:
                            curr_score = new_score
                            if curr_score < best_score:
                                best_score = curr_score
                                best_poly = curr.copy()
                            no_improve = 0
                            found_improvement = True
                            break
                        curr[i] = curr[i] - dv  # restore
                
                if found_improvement:
                    break
            
            if not found_improvement:
                no_improve += 1

    # --- Phase 4: Final polish with random restarts ---
    if time.time() < deadline - 3:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            curr = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            score = evaluate(curr)
            if score < best_score:
                best_score = score
                best_poly = curr.copy()

    if best_poly is None:
        best_poly = [1] * 12

    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly
