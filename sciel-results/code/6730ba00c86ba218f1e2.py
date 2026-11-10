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

    # Precompute convolution structure
    # sq[k] = sum_{i+j=k} c[i]*c[j] for k=0..22
    # For each k, store which (i,j) pairs contribute
    pair_contribs = {k: [] for k in range(23)}
    for i in range(12):
        for j in range(12):
            k = i + j
            pair_contribs[k].append((i, j))

    def compute_square(coeffs):
        """Compute the square coefficients."""
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

    def count_nz(sq):
        return sum(1 for x in sq if x != 0)

    # Strategy 1: Targeted cancellation search
    # For a given pattern of which sq[k] should be zero, try to find coefficients
    # that satisfy those constraints.
    
    # Key insight: if we want sq[k] = 0 for a set Z of k values,
    # we have |Z| quadratic equations in 12 unknowns.
    # We can use iterative refinement: fix some coefficients, solve for others.

    # Let's try a more direct approach: 
    # Start with a random polynomial, then iteratively try to zero out
    # the most "expensive" nonzero square coefficients.

    def try_zero_out(coeffs, k, max_iter=100):
        """Try to zero out sq[k] by modifying one coefficient."""
        sq = compute_square(coeffs)
        if sq[k] == 0:
            return coeffs, count_nz(sq)
        
        current_score = count_nz(sq)
        
        # For each index i that affects sq[k], try to find a new value
        # that zeros out sq[k]
        for i in range(12):
            j = k - i
            if j < 0 or j >= 12:
                continue
            
            # sq[k] = A*c[i]^2 + B*c[i] + C
            # where A = 1 if 2*i == k else 0
            # B = sum of c[l] for l != i such that l + (k-l) = k and l != i, (k-l) != i
            # C = sum of c[l]*c[m] for l,m != i such that l+m=k
            
            A = 1 if (2 * i == k) else 0
            B = 0
            C = 0
            
            for l in range(12):
                m = k - l
                if m < 0 or m >= 12:
                    continue
                if l == i:
                    continue
                if m == i:
                    continue
                # c[l]*c[m] contributes to sq[k]
                # If l == m, it's c[l]^2
                # But we're excluding l==i and m==i
                C += coeffs[l] * coeffs[m]
                
            # B: coefficient of c[i] in sq[k]
            # Terms involving c[i]: c[i]*c[j] where j = k-i
            # If i != j: 2*c[i]*c[j] (both (i,j) and (j,i))
            # If i == j: c[i]^2
            if i != j:
                B = 2 * coeffs[j]
            else:
                # i == j, so 2*i == k, A=1, B=0
                B = 0
            
            # Now solve A*c[i]^2 + B*c[i] + C = 0
            if A == 0:
                # Linear: B*c[i] + C = 0
                if B != 0:
                    new_val = -C // B
                    if B * new_val + C == 0:  # Exact division
                        if -64 <= new_val <= 64 and new_val != 0:
                            new_coeffs = coeffs.copy()
                            new_coeffs[i] = new_val
                            new_sq = compute_square(new_coeffs)
                            new_score = count_nz(new_sq)
                            if new_score < current_score:
                                return new_coeffs, new_score
            else:
                # Quadratic: A*c[i]^2 + B*c[i] + C = 0
                # Discriminant: B^2 - 4*A*C
                disc = B * B - 4 * A * C
                if disc >= 0:
                    sqrt_disc = int(np.sqrt(disc))
                    if sqrt_disc * sqrt_disc == disc:
                        for sign in [1, -1]:
                            num = -B + sign * sqrt_disc
                            den = 2 * A
                            if den != 0 and num % den == 0:
                                new_val = num // den
                                if -64 <= new_val <= 64 and new_val != 0:
                                    new_coeffs = coeffs.copy()
                                    new_coeffs[i] = new_val
                                    new_sq = compute_square(new_coeffs)
                                    new_score = count_nz(new_sq)
                                    if new_score < current_score:
                                        return new_coeffs, new_score
        return coeffs, current_score

    # Strategy 2: Simulated annealing with targeted moves
    def run_sa(num_restarts=15, iters_per_restart=15000):
        nonlocal best_score, best_poly
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random +/- 1
            curr = [random.choice([-1, 1]) for _ in range(12)]
            curr_score = evaluate(curr)
            
            if curr_score < best_score:
                best_score = curr_score
                best_poly = curr.copy()
                
            T = 5.0
            T_min = 0.1
            cooling_rate = 0.9995
            
            for i in range(iters_per_restart):
                if time.time() > deadline - 2:
                    return
                
                # Pick a random index to modify
                idx = random.randint(0, 11)
                old_val = curr[idx]
                
                # Choose a new value
                if random.random() < 0.6:
                    new_val = random.choice([-3, -2, -1, 1, 2, 3])
                elif random.random() < 0.8:
                    new_val = random.randint(-10, 10)
                    if new_val == 0:
                        new_val = 1
                else:
                    new_val = random.randint(-64, 64)
                    if new_val == 0:
                        new_val = 1
                
                if new_val == old_val:
                    continue
                    
                curr[idx] = new_val
                new_score = evaluate(curr)
                
                delta = new_score - curr_score
                
                # Acceptance criterion
                if delta < 0 or random.random() < np.exp(-delta / T):
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                else:
                    curr[idx] = old_val
                    
                T *= cooling_rate
                if T < T_min:
                    T = T_min

    # Strategy 3: Local search with targeted zeroing
    def local_search_with_zeroing(start_coeffs, max_iters=5000):
        nonlocal best_score, best_poly
        
        curr = start_coeffs.copy()
        curr_score = evaluate(curr)
        
        if curr_score < best_score:
            best_score = curr_score
            best_poly = curr.copy()
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 3 and no_improve < max_no_improve:
            # Find the nonzero square coefficients
            sq = compute_square(curr)
            nonzero_ks = [k for k in range(23) if sq[k] != 0]
            
            if not nonzero_ks:
                break
            
            # Try to zero out a random nonzero coefficient
            k = random.choice(nonzero_ks)
            new_curr, new_score = try_zero_out(curr, k)
            
            if new_score < curr_score:
                curr = new_curr
                curr_score = new_score
                no_improve = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr.copy()
            else:
                # Try a random perturbation
                idx = random.randint(0, 11)
                old_val = curr[idx]
                new_val = old_val + random.choice([-3, -2, -1, 1, 2, 3])
                if -64 <= new_val <= 64 and new_val != 0:
                    curr[idx] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        no_improve = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                    else:
                        curr[idx] = old_val
                        no_improve += 1
                else:
                    no_improve += 1
    
    # Run the search strategies
    # First, quick random initialization
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        curr = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        score = evaluate(curr)
        if score < best_score:
            best_score = score
            best_poly = curr.copy()
    
    # Run simulated annealing
    run_sa(num_restarts=10, iters_per_restart=20000)
    
    # Run local search with targeted zeroing from best found
    if best_poly is not None and time.time() < deadline - 10:
        local_search_with_zeroing(best_poly, max_iters=5000)
    
    # Final polish: hill climbing
    if best_poly is not None and time.time() < deadline - 5:
        curr = best_poly.copy()
        curr_score = best_score
        
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            idx = random.randint(0, 11)
            old_val = curr[idx]
            
            candidates = set()
            for dv in [-3, -2, -1, 1, 2, 3]:
                nv = old_val + dv
                if -64 <= nv <= 64 and nv != 0:
                    candidates.add(nv)
            candidates.add(-old_val)
            candidates.add(random.choice([-5, -3, -2, -1, 1, 2, 3, 5]))
            
            improved = False
            for new_val in candidates:
                if new_val == 0 or new_val == old_val:
                    continue
                if not (-64 <= new_val <= 64):
                    continue
                
                curr[idx] = new_val
                new_score = evaluate(curr)
                
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                        no_improve = 0
                    break
            if not improved:
                curr[idx] = old_val
                no_improve += 1
            else:
                no_improve = 0

    if best_poly is None:
        best_poly = [1] * 12

    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly
