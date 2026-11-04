import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 100
TARGET_SUM = N / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    def get_correlations(h):
        """
        Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
        Returns array of C_k for k != 0.
        """
        g = 1.0 - h
        # np.correlate(h, g, mode='full') computes sum h[n] * g[n-k]
        # We want overlap of h with shifted 1-h.
        # Let's stick to the definition used in the seed: max overlap of h with shifted 1-h.
        # The seed used: c = np.correlate(h, g, mode='full')
        # c[N-1] is zero shift.
        # We exclude zero shift.
        c = np.correlate(h, g, mode='full')
        return np.concatenate([c[:N-1], c[N:]])

    def max_overlap(h):
        """Return the maximum overlap value."""
        c = get_correlations(h)
        if len(c) == 0:
            return 0.0
        return np.max(c)

    def project(h):
        """
        Project h onto feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
        Uses iterative clipping and adjustment.
        """
        h = np.clip(h, 0.0, 1.0)
        for _ in range(100):
            diff = TARGET_SUM - np.sum(h)
            if abs(diff) < 1e-12:
                break
            # Adjust only those not at bounds
            mask = (h > 0.0) & (h < 1.0)
            if not np.any(mask):
                # If all are at bounds, we can't adjust while maintaining bounds?
                # But sum must be N/2. If all are 0 or 1, sum is integer.
                # N=100, sum=50. This is feasible (e.g., 50 ones, 50 zeros).
                # If we are stuck, it means we hit a corner.
                # For now, just break. The optimization should avoid this.
                break
            # Distribute diff among free variables
            adj = diff / np.sum(mask)
            h[mask] += adj
            h = np.clip(h, 0.0, 1.0)
        return h

    def objective_lse(x, beta, lam_sum):
        """
        Objective function using LogSumExp approximation of the max overlap.
        x is the vector of heights.
        """
        h = np.clip(x, 0.0, 1.0)
        overlaps = get_correlations(h)
        
        if len(overlaps) == 0:
            max_ov = 0.0
            lse = 0.0
        else:
            max_ov = np.max(overlaps)
            # Shift for numerical stability
            shifted = overlaps - max_ov
            # Use logsumexp for stability
            lse = max_ov + logsumexp(beta * shifted) / beta
        
        # Penalty for sum constraint
        sum_err = np.sum(h) - TARGET_SUM
        sum_penalty = lam_sum * (sum_err ** 2)
        
        return lse + sum_penalty

    def run_optimization(x0, beta, lam_sum, maxiter=300, maxfun=10000):
        """
        Run L-BFGS-B optimization from x0 with given parameters.
        """
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        def obj_func(x):
            return objective_lse(x, beta, lam_sum)
        
        try:
            result = minimize(
                obj_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': maxfun}
            )
            return result.x
        except Exception:
            return x0

    def local_search_swap(h, max_iters=5000):
        """
        Perform swap-based local search. Try swapping pairs of values to reduce max overlap.
        Maintains sum constraint by only swapping pairs.
        This helps escape shallow local minima in the continuous landscape.
        """
        current_h = h.copy()
        current_score = max_overlap(current_h)
        indices = np.arange(N)
        rng_local = np.random.default_rng(int(current_score * 1e6) % 1000000)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            # Try random pair swaps
            for _ in range(20):
                i, j = rng_local.choice(indices, size=2, replace=False)
                if i == j:
                    continue
                    
                # Swap values at i and j
                new_h = current_h.copy()
                new_h[i], new_h[j] = current_h[j], current_h[i]
                
                # Check bounds (swap preserves bounds if original did)
                # Project to ensure sum is exact (swap preserves sum, so projection is cheap/no-op mostly)
                new_h = project(new_h)
                new_score = max_overlap(new_h)
                
                if new_score < current_score - 1e-12:
                    current_h = new_h
                    current_score = new_score
                    improved = True
                    break
            
            if not improved:
                # Try small perturbations if swaps don't work
                perturbed = False
                for _ in range(10):
                    idx = rng_local.choice(indices)
                    step = rng_local.uniform(-0.05, 0.05)
                    new_h = current_h.copy()
                    new_h[idx] += step
                    new_h = project(new_h)
                    new_score = max_overlap(new_h)
                    if new_score < current_score - 1e-12:
                        current_h = new_h
                        current_score = new_score
                        perturbed = True
                        break
                if not perturbed:
                    break

        return current_h

    # --- Initialization ---
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Alternating binary-like (smoothed)
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # 3. Sturmian / Quasi-periodic
    def generate_sturmian(alpha, N):
        h = np.zeros(N)
        for i in range(N):
            if (i * alpha) % 1.0 < 0.5:
                h[i] = 1.0
            else:
                h[i] = 0.0
        return h

    alphas = [
        (1.0 - np.sqrt(5.0)) / 2.0 + 1.0,  # Golden ratio conjugate ~ 0.618
        np.sqrt(2.0) - 1.0,
        np.sqrt(3.0) - 1.5,
        0.5 + 0.1,
        0.5 - 0.1,
        np.pi / 10.0,
        0.381966
    ]
    
    for a in alphas:
        h_init = generate_sturmian(a, N)
        h_init = project(h_init)
        candidates.append(h_init)
        
    # 4. Sinusoidal patterns
    t = np.arange(N)
    for freq in range(1, 10):
        for amp in [0.1, 0.2, 0.3]:
            for phase in [0, np.pi/2]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
                
    # 5. Random
    for _ in range(20):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
        
    # 6. Random Binary
    for _ in range(5):
        h_rand_bin = rng.choice([0.0, 1.0], size=N)
        candidates.append(h_rand_bin)

    # Evaluate and select best initial candidates
    scored_candidates = []
    for h_cand in candidates:
        h_proj = project(h_cand)
        score = max_overlap(h_proj)
        scored_candidates.append((score, h_proj))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Keep top 3 for multi-start
    best_starts = [h for _, h in scored_candidates[:3]]
    best_score = scored_candidates[0][0]
    best_h = best_starts[0]

    # --- Optimization Phases ---
    # We increase beta to sharpen the max approximation.
    # We also increase the sum penalty lambda to enforce constraint more strictly as we converge.
    
    beta_schedule = [5, 10, 20, 50, 100, 200, 500]
    
    current_h = best_h.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
            
        # Increase lambda as beta increases to keep sum constraint tight
        lam_sum = 10.0 * (beta / 5.0)
        
        # Run optimization from current best
        new_h = run_optimization(current_h, beta, lam_sum, maxiter=300)
        
        # Project to ensure strict feasibility for evaluation
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj
            current_h = new_h_proj
        else:
            # If it didn't improve, try a local search swap from the best known
            if time.time() < deadline - 10.0:
                h_swapped = local_search_swap(best_h, max_iters=1000)
                h_swapped_proj = project(h_swapped)
                score_swap = max_overlap(h_swapped_proj)
                if score_swap < best_score:
                    best_score = score_swap
                    best_h = h_swapped_proj
                    current_h = h_swapped_proj

    # --- Final Polish ---
    # Use very high beta and high lambda for final precision
    if time.time() < deadline - 10.0:
        beta_final = 1000
        lam_final = 1000.0
        new_h = run_optimization(best_h, beta_final, lam_final, maxiter=500, maxfun=20000)
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj

    # Final local search polish
    if time.time() < deadline - 2.0:
        h_final = local_search_swap(best_h, max_iters=2000)
        h_final_proj = project(h_final)
        score_final = max_overlap(h_final_proj)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final_proj

    # Final Validation
    best_h = project(best_h)
    
    # Ensure strict constraints
    assert np.all(best_h >= 0) and np.all(best_h <= 1), "h out of bounds"
    
    # Sum constraint might be slightly off due to floating point in projection loop
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-9:
        # Simple adjustment
        mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
        if np.any(mask):
            adj = diff / np.sum(mask)
            best_h[mask] += adj
            best_h = np.clip(best_h, 0.0, 1.0)
            # Re-project if clipping broke sum
            best_h = project(best_h)
    
    return best_h
