import numpy as np
import time

def _sup_norm_fast(coeffs, n_samples=1000):
    """
    Estimate the sup-norm of the polynomial with given coefficients.
    coeffs: list of +1/-1
    n_samples: number of points on the unit circle for initial scan
    """
    # Precompute powers of z for efficiency? 
    # P(z) = sum c_k z^k.
    # z = exp(i * theta)
    # We can use FFT or direct summation. Direct summation is O(N^2) per eval if done naively, 
    # but with precomputed exps, it's O(N) per point.
    # N=24, n_samples=1000. 24*1000 = 24000 ops. Very fast.
    
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    # Compute z^k for all k and all thetas
    # z_thetas = exp(1j * thetas)
    # We want P(z) = sum_{k=0}^{23} c_k (z)^k
    
    # Vectorized approach:
    # coeffs is shape (N,)
    # thetas is shape (M,)
    # We want matrix Z of shape (M, N) where Z[m, k] = exp(i * theta_m * k)
    # Then P[m] = dot(coeffs, Z[m, :])
    
    # To save memory and time, we can just loop over k and accumulate.
    # Or use np.exp(np.outer(thetas, np.arange(len(coeffs))))
    # 1000 x 24 matrix is small.
    
    k_vals = np.arange(len(coeffs))
    # angles = np.outer(thetas, k_vals) # shape (M, N)
    # Z = np.exp(1j * angles)
    # P_vals = Z @ coeffs
    
    # Even faster: iterative multiplication
    P_vals = np.zeros(len(thetas), dtype=complex)
    # Start with c_0
    P_vals += coeffs[0]
    z_pow = np.exp(1j * thetas)
    for k in range(1, len(coeffs)):
        P_vals += coeffs[k] * z_pow
        z_pow *= np.exp(1j * thetas) # This is inefficient. z_pow should be z^k.
        
    # Let's stick to the outer product, it's cleaner and N is small.
    k_vals = np.arange(len(coeffs))
    angles = np.outer(thetas, k_vals)
    Z = np.exp(1j * angles)
    P_vals = Z @ coeffs
    
    # Find the index of the maximum magnitude
    mags = np.abs(P_vals)
    max_idx = np.argmax(mags)
    max_val = mags[max_idx]
    
    # Refine around max_idx using golden section search
    # The peak is likely between theta[max_idx-1] and theta[max_idx+1]
    # Let's define the interval [theta_left, theta_right]
    # If max_idx is 0, wrap around.
    
    # For simplicity and robustness, let's just do a few iterations of refinement
    # on the top few peaks.
    
    # Actually, for N=24, the function is smooth enough that a dense grid of 1000
    # followed by a local refinement on the top 5 peaks is sufficient for 1e-9 accuracy
    # in the final score, provided the refinement is good.
    
    # Let's refine the single highest peak for now.
    # Interval for golden section:
    # Thetas are sorted.
    # If max_idx > 0: left_theta = thetas[max_idx-1]
    # Else: left_theta = thetas[-1] - 2*pi (wrap)
    # If max_idx < n_samples-1: right_theta = thetas[max_idx+1]
    # Else: right_theta = thetas[0] + 2*pi
    
    # Let's implement a simple refinement function
    def eval_poly(theta):
        # Reuse the logic? No, let's just compute directly for a single point.
        # P(e^{i theta})
        z = np.exp(1j * theta)
        # Horner's method or direct sum? Direct sum is fine for N=24.
        # coeffs[k] * z^k
        # Pre-compute z_powers?
        # For a single point, loop is fast.
        val = 0.0 + 0.0j
        z_k = 1.0 + 0.0j
        for c in coeffs:
            val += c * z_k
            z_k *= z
        return np.abs(val)

    # Golden section search
    # Interval [a, b]
    def golden_search(a, b, func, tol=1e-12, max_iter=100):
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = func(c)
        fd = func(d)
        for _ in range(max_iter):
            if abs(b - a) < tol:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = func(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = func(d)
        # Return the max of the last few evaluations or the midpoint
        # The function is unimodal in the neighborhood of the peak?
        # |P(z)| is not necessarily unimodal, but near a local max it is.
        # We want to MAXIMIZE the magnitude.
        # Golden section above minimizes. Let's invert the function.
        # Or just implement max version.
        # Let's rewrite golden_search for MAX.
        
    def golden_search_max(a, b, func, tol=1e-12, max_iter=100):
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = func(c)
        fd = func(d)
        for _ in range(max_iter):
            if abs(b - a) < tol:
                break
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = func(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = func(d)
        # The maximum is likely at c or d.
        # Let's evaluate at (a+b)/2 as well just in case
        mid = (a + b) / 2
        fm = func(mid)
        return max(fc, fd, fm)

    # Determine bounds for refinement
    if max_idx == 0:
        a = thetas[-1] - 2 * np.pi
        b = thetas[1]
    elif max_idx == n_samples - 1:
        a = thetas[-2]
        b = thetas[0] + 2 * np.pi
    else:
        a = thetas[max_idx - 1]
        b = thetas[max_idx + 1]
        
    refined_max = golden_search_max(a, b, eval_poly)
    
    # It's possible the global max is not the one we refined if the grid was too coarse
    # and there was a higher peak elsewhere.
    # To be safe, let's refine the top 3 peaks from the grid.
    top_indices = np.argsort(mags)[-3:]
    best_refined = 0.0
    for idx in top_indices:
        if idx == 0:
            a = thetas[-1] - 2 * np.pi
            b = thetas[1] if n_samples > 1 else thetas[0] + 2*np.pi
        elif idx == n_samples - 1:
            a = thetas[-2]
            b = thetas[0] + 2 * np.pi
        else:
            a = thetas[idx - 1]
            b = thetas[idx + 1]
            
        # Ensure a < b for the search? 
        # If a > b, swap.
        if a > b:
            a, b = b, a
            
        val = golden_search_max(a, b, eval_poly)
        if val > best_refined:
            best_refined = val
            
    return max(max_val, best_refined)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial seed: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    coeffs = p[:24]
    
    # Evaluate initial score
    current_norm = _sup_norm_fast(coeffs, 2000)
    best_coeffs = coeffs[:]
    best_norm = current_norm
    
    # Local Search: Single Flips
    # We will iterate until no improvement or time limit.
    # For N=24, single flips are 24 operations.
    # We can also do pair flips (276 operations) if stuck.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single flips
        best_flip_norm = current_norm
        best_flip_idx = -1
        
        for i in range(24):
            # Flip coefficient i
            old_val = coeffs[i]
            coeffs[i] = -old_val
            norm = _sup_norm_fast(coeffs, 1000) # Lower samples for speed during search
            coeffs[i] = old_val
            
            if norm < best_flip_norm:
                best_flip_norm = norm
                best_flip_idx = i
        
        if best_flip_idx != -1:
            # Apply the best flip
            coeffs[best_flip_idx] = -coeffs[best_flip_idx]
            current_norm = best_flip_norm
            best_coeffs = coeffs[:]
            best_norm = current_norm
            improved = True
            continue
            
        # If no single flip improved, try pair flips
        best_pair_norm = current_norm
        best_pair_i = -1
        best_pair_j = -1
        
        for i in range(23):
            for j in range(i + 1, 24):
                # Flip i and j
                old_i = coeffs[i]
                old_j = coeffs[j]
                coeffs[i] = -old_i
                coeffs[j] = -old_j
                
                norm = _sup_norm_fast(coeffs, 500) # Even fewer samples for pair flips
                coeffs[i] = old_i
                coeffs[j] = old_j
                
                if norm < best_pair_norm:
                    best_pair_norm = norm
                    best_pair_i = i
                    best_pair_j = j
                    
        if best_pair_i != -1:
            coeffs[best_pair_i] = -coeffs[best_pair_i]
            coeffs[best_pair_j] = -coeffs[best_pair_j]
            current_norm = best_pair_norm
            best_coeffs = coeffs[:]
            best_norm = current_norm
            improved = True
            continue
            
        # If no pair flip improved, we are at a local minimum.
        # Try a random restart if we have time?
        # Or break and return best found.
        # Given the problem structure, the GRS seed + local search usually finds the global optimum.
        # To be safe, if we have > 10s left, do a few random restarts.
        if time.time() > deadline - 10.0:
            break
        
        # Random restart
        # Generate random coefficients
        random_coeffs = [1 if np.random.rand() < 0.5 else -1 for _ in range(24)]
        coeffs = random_coeffs
        current_norm = _sup_norm_fast(coeffs, 2000)
        if current_norm < best_norm:
            best_coeffs = coeffs[:]
            best_norm = current_norm
            
        # Run local search on this new seed
        # (The loop above will handle it, but we need to re-enter the 'improved' state)
        # Actually, the loop structure above handles the current `coeffs`.
        # So just setting `coeffs` and `current_norm` and continuing the loop is correct.
        # However, `improved` was False, so the `while` loop would exit.
        # We need to force it to continue.
        improved = True 
        # But we should only do a limited number of restarts to avoid infinite loop if stuck.
        # Let's count restarts.
        
    # Final high-precision evaluation of the best candidate
    # The search used lower precision. We need to report the score accurately?
    # No, the evaluator calculates the score. We just return the coefficients.
    # But we should make sure we returned the *best* coefficients found.
    
    return best_coeffs

# Add a restart counter to prevent infinite random restarts if stuck
# Let's refine the loop logic slightly to include restart limit.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial seed
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    coeffs = p[:24]
    
    current_norm = _sup_norm_fast(coeffs, 2000)
    best_coeffs = coeffs[:]
    best_norm = current_norm
    
    restart_count = 0
    max_restarts = 50
    
    while time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Single flips
            best_flip_norm = current_norm
            best_flip_idx = -1
            for i in range(24):
                old_val = coeffs[i]
                coeffs[i] = -old_val
                norm = _sup_norm_fast(coeffs, 1000)
                coeffs[i] = old_val
                if norm < best_flip_norm:
                    best_flip_norm = norm
                    best_flip_idx = i
                    
            if best_flip_idx != -1:
                coeffs[best_flip_idx] = -coeffs[best_flip_idx]
                current_norm = best_flip_norm
                if current_norm < best_norm:
                    best_norm = current_norm
                    best_coeffs = coeffs[:]
                improved = True
                continue
                
            # Pair flips
            best_pair_norm = current_norm
            best_pair_i = -1
            best_pair_j = -1
            for i in range(23):
                for j in range(i + 1, 24):
                    old_i = coeffs[i]
                    old_j = coeffs[j]
                    coeffs[i] = -old_i
                    coeffs[j] = -old_j
                    norm = _sup_norm_fast(coeffs, 500)
                    coeffs[i] = old_i
                    coeffs[j] = old_j
                    if norm < best_pair_norm:
                        best_pair_norm = norm
                        best_pair_i = i
                        best_pair_j = j
                        
            if best_pair_i != -1:
                coeffs[best_pair_i] = -coeffs[best_pair_i]
                coeffs[best_pair_j] = -coeffs[best_pair_j]
                current_norm = best_pair_norm
                if current_norm < best_norm:
                    best_norm = current_norm
                    best_coeffs = coeffs[:]
                improved = True
                continue
                
            # Stuck in local minimum
            if restart_count >= max_restarts:
                break
                
            # Random restart
            restart_count += 1
            coeffs = [1 if np.random.rand() < 0.5 else -1 for _ in range(24)]
            current_norm = _sup_norm_fast(coeffs, 2000)
            if current_norm < best_norm:
                best_norm = current_norm
                best_coeffs = coeffs[:]
            # Continue outer loop to run local search on new seed
            
        if time.time() >= deadline:
            break
            
    return best_coeffs

# Wait, the inner loop `while improved` will exit when `improved` is False.
# Then the outer loop checks time, increments restart, sets new coeffs, and loops back to `while improved` (which is inside the outer loop? No, I structured it as nested).
# Let's trace:
# Outer: while time < deadline
#   Inner: while improved
#     ...
#     if stuck: improved=False (implicitly by not setting it to True) -> Inner loop exits
#   if stuck (inner exited without improvement):
#     restart
#     set coeffs
#     (Loop back to Inner? No, the Inner loop is already exited. The Outer loop continues.)
#     Next iteration of Outer loop enters Inner loop with `improved=True`?
#     No, `improved` is a variable inside the outer loop scope.
#     I need to reset `improved = True` before entering the inner loop in the next iteration.
#     In my code above, `improved = True` is set *inside* the outer loop, before the inner loop.
#     So yes, it will re-enter the inner loop with the new random coeffs.

# One optimization: The `_sup_norm_fast` function creates a new `eval_poly` closure each time.
# This is fine for N=24.

# Let's double check the GRS seed.
# GRS polynomial for N=24.
# p=1, q=1
# 1,1 -> 1,1,1,1 (p=1111, q=1-11-1? No. Q -> P, -Q. P -> P+Q.
# Start P=1, Q=1.
# Step 1: P=1+1=11, Q=1,-1.
# Step 2: P=11 + 1-1 = 111-1. Q=1,-1 + -1,-(-1)=1,1 => 1,-1,-1,1.
# Step 3: P=111-1 + 1,-1,-1,1 = 111-11-1-11. (8 coeffs)
# Step 4: P (16 coeffs)
# Step 5: P (32 coeffs) -> truncate to 24.
# This is a valid starting point.

# The search should converge to the optimum quickly.

# Final check on `_sup_norm_fast`:
# It uses `np.linspace` which is deterministic.
# `np.random.rand` is used for restarts.
# The function returns a list of ints.

# One potential issue: `eval_poly` inside `_sup_norm_fast` uses `coeffs` from the enclosing scope of `_sup_norm_fast`?
# No, `coeffs` is an argument to `_sup_norm_fast`.
# `eval_poly` is defined inside `_sup_norm_fast` and captures `coeffs`.
# This is correct.

# Performance:
# `_sup_norm_fast` with 1000 samples:
# `np.outer` 1000x24. `np.exp`. `@` dot product.
# This is very fast in NumPy.
# 24 single flips * 1000 samples * (some constant) ~ 24k ops.
# 276 pair flips * 500 samples ~ 138k ops.
# Total per iteration ~ 160k ops.
# 100 iterations ~ 16M ops.
# Python overhead for the loops might be significant.
# 24 + 276 = 300 calls to `_sup_norm_fast`.
# Each call does vectorized NumPy ops.
# Should be well within 110s.

# To be safe against time limits, I will add a check inside the flip loops.
# If time is running out, break.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial seed
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    coeffs = p[:24]
    
    current_norm = _sup_norm_fast(coeffs, 2000)
    best_coeffs = coeffs[:]
    best_norm = current_norm
    
    restart_count = 0
    max_restarts = 100
    
    while time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Single flips
            best_flip_norm = current_norm
            best_flip_idx = -1
            for i in range(24):
                if time.time() > deadline - 5.0:
                    break
                old_val = coeffs[i]
                coeffs[i] = -old_val
                norm = _sup_norm_fast(coeffs, 1000)
                coeffs[i] = old_val
                if norm < best_flip_norm:
                    best_flip_norm = norm
                    best_flip_idx = i
                    
            if best_flip_idx != -1:
                coeffs[best_flip_idx] = -coeffs[best_flip_idx]
                current_norm = best_flip_norm
                if current_norm < best_norm:
                    best_norm = current_norm
                    best_coeffs = coeffs[:]
                improved = True
                continue
                
            # Pair flips
            best_pair_norm = current_norm
            best_pair_i = -1
            best_pair_j = -1
            for i in range(23):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, 24):
                    if time.time() > deadline - 5.0:
                        break
                    old_i = coeffs[i]
                    old_j = coeffs[j]
                    coeffs[i] = -old_i
                    coeffs[j] = -old_j
                    norm = _sup_norm_fast(coeffs, 500)
                    coeffs[i] = old_i
                    coeffs[j] = old_j
                    if norm < best_pair_norm:
                        best_pair_norm = norm
                        best_pair_i = i
                        best_pair_j = j
                        
            if best_pair_i != -1:
                coeffs[best_pair_i] = -coeffs[best_pair_i]
                coeffs[best_pair_j] = -coeffs[best_pair_j]
                current_norm = best_pair_norm
                if current_norm < best_norm:
                    best_norm = current_norm
                    best_coeffs = coeffs[:]
                improved = True
                continue
                
            # Stuck
            if restart_count >= max_restarts:
                break
                
            restart_count += 1
            coeffs = [1 if np.random.rand() < 0.5 else -1 for _ in range(24)]
            current_norm = _sup_norm_fast(coeffs, 2000)
            if current_norm < best_norm:
                best_norm = current_norm
                best_coeffs = coeffs[:]
                
    return best_coeffs

# The `_sup_norm_fast` function needs to be defined before `construct` uses it.
# I will place it before `construct`.

# One detail: `np.linspace` endpoint=False is good for periodicity.
# The refinement logic handles wrap-around.

# Let's verify the score calculation in the task description.
# "score is max over |z|=1 of |P(z)| divided by sqrt(24)"
# My `_sup_norm_fast` returns `max |P(z)|`.
# The evaluator will divide by `sqrt(24)`.
# So I should minimize `max |P(z)|`.
# My search minimizes `norm` which is `max |P(z)|`.
# This is consistent.

# The optimal score is 1.2583758.
# So optimal `max |P|` is `1.2583758 * sqrt(24)`.

# The code looks solid.
