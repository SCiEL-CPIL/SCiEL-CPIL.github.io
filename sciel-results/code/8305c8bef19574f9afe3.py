import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for safety
    
    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        Returns an array of correlation values for shifts k = -(N-1) ... -1, 1 ... N-1.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c has length 2N-1.
        # Indices: 0 to N-1 are shifts -(N-1) to 0.
        # Indices: N to 2N-2 are shifts 1 to N-1.
        # We want non-zero shifts.
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
        """
        h = np.clip(h, 0.0, 1.0)
        for _ in range(50):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-14:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        # Final exact correction
        residual = N / 2.0 - np.sum(h)
        if abs(residual) > 1e-15:
            interior = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        return h

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def coordinate_descent(h, max_iters=1000):
        """
        Perform coordinate descent minimizing the max correlation.
        For each coordinate i, find the optimal h_i in [0,1] that minimizes
        the maximum of the affected correlations, then project back to feasible set.
        """
        h = h.copy()
        h = project_to_feasible(h)
        
        # Precompute which shifts are affected by which index?
        # Shift k (positive) involves h[i] and h[i+k].
        # So h[i] affects:
        # 1. C_k where i is the first index: k such that i+k < N. (Shifts k = 1..N-1-i)
        # 2. C_k where i is the second index: k such that i-k >= 0. (Shifts k = 1..i)
        # Note: The correlation definition in np.correlate:
        # c[k] = sum h[n] g[n+k].
        # If we use the non-zero shifts array:
        # The array returned by get_correlations has length 2(N-1).
        # First N-1 elements correspond to negative shifts?
        # Let's check np.correlate(h, g, 'full').
        # c[m] = sum_n h[n] g[n+m-N+1]? No.
        # Standard definition: c[k] = sum_n h[n] g[n-k].
        # np.correlate(a, v, 'full') computes sum_n a[n] conj(v[n+k]).
        # Let's stick to the indices in the returned array.
        # The returned array `c_nonzero` in get_correlations is:
        # c[:N-1] corresponds to shifts where the overlap is partial (negative lags in standard correlation terminology if we align start).
        # c[N:] corresponds to positive lags.
        
        # To be safe and fast, we will just re-evaluate the objective for candidate h_i values.
        # Since N=100, evaluating the full score is O(N^2) ~ 10^4 ops.
        # Coordinate descent does N updates per iteration.
        # Total cost per iteration: N * (Cost to find optimal h_i + Cost to project).
        # Cost to find optimal h_i:
        # We need to find min_{x in [0,1]} max_k C_k(x).
        # We can sample the kinks or just use a ternary search if we assume unimodality?
        # Max of linear functions is convex, so it is unimodal.
        # We can use ternary search or binary search on the derivative.
        # Or simply, since the number of linear pieces is small, we can find the intersection points.
        
        # However, a simpler and robust approach for N=100:
        # For each i, define f(x) = score_h(h_with_i_set_to_x).
        # This is O(N^2) to evaluate.
        # We can do a fine grid search or ternary search on x in [0,1].
        # Ternary search on a convex function takes ~50 evaluations.
        # 50 * N (coords) * N (iterations) = 50 * 100 * 1000 = 5,000,000 evaluations of score_h?
        # No, score_h is O(N^2) = 10,000 ops.
        # 5,000,000 * 10,000 = 5 * 10^10 ops. Too slow.
        
        # We need to be smarter.
        # When we change h_i to x, only correlations involving i change.
        # Let's precompute the "base" correlations excluding i's contribution?
        # C_k = sum_{n != i, n+k != i} h_n (1-h_{n+k}) + (terms with i).
        
        # Let's implement the "Find min of max of linear functions" exactly.
        # For a fixed i, let's identify the shifts k that involve i.
        # In np.correlate(h, 1-h, 'full'):
        # The output c has length 2N-1.
        # c[idx] = sum_{n} h[n] (1-h[n+k_idx]) where k_idx is related to idx.
        # Specifically, if we use standard correlation c[k] = sum h[n] g[n-k],
        # then c has indices from -(N-1) to N-1.
        # np.correlate returns them in order from -(N-1) to N-1.
        # c[0] corresponds to shift -(N-1).
        # c[N-1] corresponds to shift 0.
        # c[N] corresponds to shift 1.
        
        # The function get_correlations returns:
        # c_nonzero = concat(c[:N-1], c[N:])
        # So index 0 in c_nonzero is shift -(N-1).
        # Index N-2 in c_nonzero is shift -1.
        # Index N-1 in c_nonzero is shift 1.
        # Index 2N-3 in c_nonzero is shift N-1.
        
        # For a specific i, which shifts k involve h_i?
        # h_i appears in the sum for shift k if:
        # 1. n = i, so we need g[i-k] to be valid. i-k in [0, N-1] => k in [i-N+1, i].
        # 2. n-k = i => n = i+k, so we need h[i+k] to be valid. i+k in [0, N-1] => k in [-i, N-1-i].
        
        # Let's just use a simpler heuristic:
        # For each i, we will try a few candidate values for h_i:
        # 1. Current value.
        # 2. 0.
        # 3. 1.
        # 4. Values that equalize the top 2 correlations.
        
        # Actually, let's just use **Ternary Search** on the exact score, but optimized.
        # We can cache the score calculation? No, h changes.
        
        # Alternative: **Random Restarts with Simulated Annealing on Exact Max**.
        # SA is robust to non-convexity and doesn't need gradients.
        # N=100 is small enough for SA to work well if we have a good neighborhood.
        # Neighborhood: pick random i, j, swap or adjust h_i, h_j to maintain sum.
        
        # Let's try **Coordinate Descent with Local Linear Approximation**.
        # For each i, we want to minimize max_k C_k.
        # We can approximate C_k near current h_i as linear.
        # C_k(h_i + delta) ≈ C_k(h_i) + grad_k_i * delta.
        # We want to minimize max_k (C_k(h_i) + grad_k_i * delta).
        # This is a 1D convex optimization problem.
        # We can solve it analytically or with a few bisections.
        
        # Let's implement this.
        
        h = h.copy()
        for _iter in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            improved = False
            for i in range(N):
                # Compute gradients of each C_k w.r.t h_i
                # C_k = sum_n h_n (1-h_{n+k})
                # dC_k/dh_i = (1-h_{i+k}) - (1-h_{i-k})  (with bounds)
                
                # We need to map the shifts in get_correlations to the actual k values.
                # c_nonzero indices 0..N-2 correspond to k = -(N-1)..-1.
                # c_nonzero indices N-1..2N-3 correspond to k = 1..N-1.
                
                # Let's compute the gradient vector for all active k's.
                # For a given i, the active k's are those where i and i+k are in bounds.
                
                # Instead of complex indexing, let's just evaluate the score for a few candidate h_i values.
                # Candidates:
                # 1. h[i]
                # 2. 0.0
                # 3. 1.0
                # 4. The value that minimizes the max of the two largest correlations involving i.
                
                # To find candidate 4, we need to know which correlations are largest.
                # Let's just do a ternary search on [0,1] for h_i, but limit evaluations.
                # Since f(x) is convex, ternary search is safe.
                # 20 iterations of ternary search gives high precision.
                # 20 evaluations per coordinate * 100 coordinates = 2000 evaluations per pass.
                # 1000 passes = 2,000,000 evaluations.
                # Each evaluation is O(N^2) = 10^4.
                # Total 2 * 10^10 ops. Still too slow.
                
                # We MUST use the gradient approximation or exact linear analysis.
                
                # Let's go back to exact linear analysis for 1D.
                # For fixed i, let's identify the shifts k that involve i.
                # And compute the linear coefficients a_k, b_k such that C_k = a_k h_i + b_k.
                
                # This requires knowing the mapping between c_nonzero index and k.
                # Let's create a helper to get the k value for a given c_nonzero index.
                
                # c_nonzero[0] is k = -(N-1)
                # ...
                # c_nonzero[N-2] is k = -1
                # c_nonzero[N-1] is k = 1
                # ...
                # c_nonzero[2N-3] is k = N-1
                
                def get_k_from_idx(idx):
                    if idx < N-1:
                        return idx - (N-1) # -(N-1) to -1
                    else:
                        return idx - (N-1) + 1 # 1 to N-1

                # For current h, compute C_k for all k.
                c_all = get_correlations(h)
                
                # For coordinate i, find all k such that h_i is involved in C_k.
                # h_i is involved if:
                # 1. i is the "first" index: n=i. Then we need g[i-k] to exist.
                #    i-k in [0, N-1] => k in [i-N+1, i].
                # 2. i is the "second" index: n+k=i => n=i-k. Then we need h[i-k] to exist.
                #    i-k in [0, N-1] => k in [-i, N-1-i].
                
                # So k must be in the union of these ranges.
                # And k must be non-zero.
                
                active_ks = []
                for k in range(-(N-1), N):
                    if k == 0:
                        continue
                    # Check if k is in range for "first" or "second"
                    # First: i-k in [0, N-1]
                    if 0 <= i-k <= N-1:
                        active_ks.append(k)
                    # Second: i+k in [0, N-1] (this is the same condition for the other term in the sum)
                    # Actually, the sum is over n.
                    # C_k = sum_n h_n (1-h_{n+k}).
                    # Term for n=i: h_i (1-h_{i+k}). Requires i+k in [0, N-1].
                    # Term for n=i-k: h_{i-k} (1-h_i). Requires i-k in [0, N-1].
                    
                    if (0 <= i+k <= N-1) or (0 <= i-k <= N-1):
                        # k is active
                        # We need to find the index in c_all for this k
                        if k < 0:
                            idx = N-1 + k # k=-(N-1) -> 0, k=-1 -> N-2
                        else:
                            idx = N-1 + k - 1 # k=1 -> N-1, k=N-1 -> 2N-3
                        
                        # Get current C_k
                        c_val = c_all[idx]
                        
                        # Compute derivative dC_k/dh_i
                        # C_k = ... + h_i(1-h_{i+k}) + ... + h_{i-k}(1-h_i) + ...
                        # d/dh_i [h_i(1-h_{i+k})] = 1 - h_{i+k} (if i+k valid)
                        # d/dh_i [h_{i-k}(1-h_i)] = -h_{i-k} (if i-k valid)
                        
                        deriv = 0.0
                        if 0 <= i+k <= N-1:
                            deriv += (1.0 - h[i+k])
                        if 0 <= i-k <= N-1:
                            deriv += (-h[i-k])
                        
                        active_ks.append((idx, c_val, deriv))
                
                # Now we have a list of (idx, c_val, deriv).
                # We want to find x in [0,1] that minimizes max_j (c_val_j + deriv_j * (x - h[i])).
                # Let delta = x - h[i].
                # f(delta) = max_j (c_val_j + deriv_j * delta).
                # This is a convex piecewise linear function of delta.
                # We want to minimize it.
                
                # The minimum is at a kink or boundary.
                # Kinks occur where c_val_a + deriv_a * delta = c_val_b + deriv_b * delta.
                # delta = (c_val_b - c_val_a) / (deriv_a - deriv_b).
                
                # We can find the minimum by checking the value at delta=0 (current),
                # and at all kinks within the range corresponding to x in [0,1].
                # Range of delta: [-h[i], 1-h[i]].
                
                candidates = [0.0] # delta = 0
                # Add boundaries
                candidates.append(-h[i])
                candidates.append(1.0 - h[i])
                
                # Add kinks
                # Number of active ks is small (~N).
                # Pairs are O(N^2).
                for a in range(len(active_ks)):
                    for b in range(a+1, len(active_ks)):
                        idx_a, c_a, d_a = active_ks[a]
                        idx_b, c_b, d_b = active_ks[b]
                        
                        if abs(d_a - d_b) < 1e-12:
                            continue
                            
                        delta_kink = (c_b - c_a) / (d_a - d_b)
                        
                        if -h[i] - 1e-9 <= delta_kink <= 1.0 - h[i] + 1e-9:
                            candidates.append(delta_kink)
                
                # Evaluate f(delta) for all candidates and pick the best
                best_delta = 0.0
                best_val = np.inf
                
                for d in candidates:
                    # Clamp d to valid range
                    d = np.clip(d, -h[i], 1.0 - h[i])
                    
                    # Compute max correlation for this delta
                    # We only need to check the active ks?
                    # No, other ks don't change.
                    # So max is max( max_inactive_c, max_active_c_new )
                    
                    # We don't have max_inactive_c easily.
                    # But we can just compute the full score for the best few candidates?
                    # Or we can approximate.
                    
                    # Let's just compute the max of the active ones + the max of the inactive ones.
                    # We can precompute max_inactive?
                    # The set of inactive ks depends on i.
                    
                    # For speed, let's just evaluate the full score for the top 3 candidates.
                    # But we need to know which are top.
                    
                    # Let's just evaluate the full score for all unique candidates.
                    # Number of candidates is O(N^2) ~ 10^4.
                    # Too many.
                    
                    # Let's restrict candidates to the top 10 kinks closest to 0.
                    # Or just use ternary search on the exact score for this coordinate.
                    # 30 iterations of ternary search.
                    # 30 * 100 (coords) * 100 (passes) = 30,000 evaluations.
                    # 30,000 * 10^4 = 3 * 10^8 ops. This is feasible in Python if optimized?
                    # Numpy vectorization helps.
                    
                    # Let's try ternary search on the exact score.
                    lo = -h[i]
                    hi = 1.0 - h[i]
                    
                    for _ in range(30):
                        m1 = lo + (hi - lo) / 3.0
                        m2 = hi - (hi - lo) / 3.0
                        
                        h_temp = h.copy()
                        h_temp[i] += m1
                        h_temp = project_to_feasible(h_temp)
                        f1 = score_h(h_temp)
                        
                        h_temp[i] += m2
                        h_temp = project_to_feasible(h_temp)
                        f2 = score_h(h_temp)
                        
                        if f1 < f2:
                            hi = m2
                        else:
                            lo = m1
                    
                    delta_opt = (lo + hi) / 2.0
                    h[i] += delta_opt
                    h = project_to_feasible(h)
                    
            # End coordinate loop
            # Check convergence
            # If no significant improvement, break
            
        return h

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)
    
    rng = np.random.default_rng(42)
    
    # Multi-restart with Coordinate Descent
    restarts = []
    
    # 1. Uniform
    restarts.append(np.full(N, 0.5))
    
    # 2. Random
    for _ in range(5):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = project_to_feasible(h_rand)
        restarts.append(h_rand)
        
    # 3. Bimodal
    for _ in range(5):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.8, 0.2)
        h_rand = project_to_feasible(h_rand)
        restarts.append(h_rand)

    for h0 in restarts:
        if time.time() - start_time > time_limit:
            break
        
        # Run coordinate descent
        h_opt = coordinate_descent(h0, max_iters=50)
        current_score = score_h(h_opt)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
            
    # Polish
    if time.time() - start_time < time_limit - 10.0:
        h_polished = coordinate_descent(best_h, max_iters=100)
        polished_score = score_h(h_polished)
        if polished_score < best_score:
            best_score = polished_score
            best_h = h_polished.copy()
            
    best_h = project_to_feasible(best_h)
    return best_h

