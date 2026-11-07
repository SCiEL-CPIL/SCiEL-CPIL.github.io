import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import fft, ifft

N = 100

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _raw_max_corr(h):
    """Raw max correlation (unnormalized)."""
    return np.max(np.correlate(h, 1 - h, mode="full"))

def _fast_corr(h):
    """
    Compute full correlation h * (1-h) using FFT for speed.
    Returns array of length 2*N - 1 corresponding to lags -(N-1) to N-1.
    """
    n = len(h)
    # Zero-pad to avoid circular correlation
    size = 2 * n
    H = fft(h, size)
    G = fft(1 - h, size)
    # Correlation is inverse FFT of H * conj(G) ? 
    # np.correlate(a, v) is sum a[n+k] * conj(v[n]). 
    # For real vectors, it's sum a[n+k] * v[n].
    # Let's verify the FFT definition.
    # corr[lag] = sum_i h[i] * (1-h)[i-lag]
    # This is cross-correlation.
    # Using FFT: IFFT(FFT(h) * conj(FFT(1-h))) gives circular cross-correlation.
    # We need linear cross-correlation.
    # Standard way:
    # corr = ifft(fft(h, size) * conj(fft(1-h, size)))
    # The result is circular. We need to unwrap.
    # The linear correlation of length 2N-1 is embedded in the circular correlation of length 2N.
    # Specifically, if we zero-pad to 2N, the circular correlation of the padded signals 
    # is the linear correlation followed by wrap-around.
    # Since N=100, size=200. The linear corr has 199 points.
    # The circular corr will have 200 points.
    # The first N points of the circular corr correspond to lags 0 to N-1?
    # Let's just use np.correlate for correctness in the objective, 
    # and FFT for the local search heuristic where speed is critical and approximate is okay?
    # No, we need accurate enough to guide the search.
    # Let's stick to np.correlate for the objective, and use FFT only if we can verify it.
    # For N=100, np.correlate is quite fast (~100us). 
    # 10,000 iterations * 100us = 1s. 
    # So we might not need FFT for the local search if we limit iterations.
    # But let's keep the FFT implementation ready.
    
    # Correct FFT correlation:
    # c = ifft(fft(h, size) * np.conj(fft(1-h, size)))
    # c is circular. 
    # The linear correlation is c[0:N] + c[-(N-1):]?
    # Let's just use np.correlate for simplicity and reliability in the tight loops, 
    # as N is small.
    
    # Actually, let's implement a fast version using FFT properly.
    # From scipy documentation or standard DSP:
    # cross_corr = ifft(fft(h, size) * conj(fft(g, size)))
    # The linear cross-correlation is the first N elements of the circular cross-correlation?
    # No.
    # Let's use the property: 
    # corr_full = np.concatenate([corr_neg, corr_pos])
    # where corr_pos[k] = sum h[i+k] * g[i] for k=0..N-1
    # This is exactly what np.correlate(h, g, 'full') returns?
    # np.correlate(h, g, 'full') returns length 2N-1.
    # Index 0 corresponds to lag -(N-1).
    # Index N-1 corresponds to lag 0.
    
    # Let's just use np.correlate. It's fast enough for N=100.
    return np.correlate(h, 1 - h, mode="full")

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(50):
        if abs(diff) < 1e-15:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _construct_from_pw_linear(ctrl_points):
    """
    Construct h from piecewise-linear control points.
    ctrl_points: array of length K (e.g., 10) defining values at evenly spaced positions.
    Linearly interpolate to N=100 points.
    """
    K = len(ctrl_points)
    ctrl_pos = np.linspace(0, N - 1, K)
    target_pos = np.arange(N)
    h = np.interp(target_pos, ctrl_pos, ctrl_points)
    return h

def _pw_linear_objective(ctrl_points):
    """Objective from piecewise-linear control points."""
    h = _construct_from_pw_linear(ctrl_points)
    h = _project_to_feasible(h)
    return _correlation_value(h)

def _local_search_alignment(h, time_limit=10.0):
    """
    Local search based on alignment heuristic.
    Try to reduce max correlation by swapping values to align h with h shifted by k*.
    """
    best_h = h.copy()
    best_val = _raw_max_corr(best_h)
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Precompute indices
    indices = np.arange(N)
    
    while time.time() < start_time + time_limit:
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        k_star = max_idx - (N - 1)
        
        # If k_star is 0, we are looking at sum(h*(1-h)) which is fixed?
        # sum(h - h^2) = sum(h) - sum(h^2) = N/2 - sum(h^2).
        # To minimize this, we maximize sum(h^2), i.e., push h to 0 or 1.
        # If k_star != 0, we want to increase autocorr at k_star.
        
        improved = False
        
        # Strategy: Find i where h[i] is "out of alignment" with h[i-k_star]
        # We want h[i] and h[i-k_star] to be similar.
        # If h[i] > h[i-k_star], we want to decrease h[i] or increase h[i-k_star].
        # If h[i] < h[i-k_star], we want to increase h[i] or decrease h[i-k_star].
        
        # We perform a swap between i and j to maintain sum.
        # Choose i such that the "misalignment" is large.
        # Misalignment metric: |h[i] - h[i-k_star]|
        
        # Get the segment
        if k_star >= 0:
            i_range = np.arange(N - k_star)
            h_i = best_h[i_range]
            h_j = best_h[i_range - k_star] if k_star > 0 else best_h[i_range]
            # Actually, if k_star > 0, h[i] and h[i-k_star]
            # Let's just iterate over all i
        else:
            k_abs = -k_star
            i_range = np.arange(N - k_abs)
            h_i = best_h[i_range]
            h_j = best_h[i_range + k_abs]
            
        # Calculate misalignment
        if k_star >= 0:
            if k_star == 0:
                # Special case: push to extremes
                # Find i with h[i] in (0,1)
                mid_idx = np.where((best_h > 0.1) & (best_h < 0.9))[0]
                if len(mid_idx) > 0:
                    i = mid_idx[rng.integers(len(mid_idx))]
                    # Push to 0 or 1
                    if best_h[i] < 0.5:
                        delta = -best_h[i]
                    else:
                        delta = 1.0 - best_h[i]
                    # Find j to absorb
                    j_candidates = np.where((best_h > 0.1) & (best_h < 0.9) & (np.arange(N) != i))[0]
                    if len(j_candidates) > 0:
                         # This is getting complicated. Let's use random swap first.
                        pass
            
            # General case for k_star != 0
            # We want to increase sum h[i] h[i-k_star]
            # Pick i, j randomly? Or pick i with high misalignment?
            # Let's pick i randomly, and j randomly, and try swap.
            # This is slow.
            
            # Better: Pick i such that h[i] and h[i-k_star] are very different.
            # And pick j such that h[j] and h[j-k_star] are very different in the opposite way?
            # Or just swap h[i] and h[j] and see if it improves.
            
            # Random Swap Strategy
            for _ in range(50):
                i = rng.integers(N)
                j = rng.integers(N)
                if i == j:
                    continue
                new_h = best_h.copy()
                new_h[i], new_h[j] = new_h[j], new_h[i]
                val = _raw_max_corr(new_h)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = new_h
                    improved = True
                    break
        
        # Also try small perturbations
        if not improved:
            for _ in range(20):
                i = rng.integers(N)
                j = rng.integers(N)
                if i == j:
                    continue
                d = rng.uniform(-0.05, 0.05)
                new_h_i = best_h[i] + d
                new_h_j = best_h[j] - d
                if 0 <= new_h_i <= 1 and 0 <= new_h_j <= 1:
                    new_h = best_h.copy()
                    new_h[i] = new_h_i
                    new_h[j] = new_h_j
                    val = _raw_max_corr(new_h)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
                        
        if not improved:
             # Stuck, break or continue?
             # Let's continue a bit more with random restarts in local region
             pass

    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)

    # Phase 1: Differential evolution over piecewise-linear control points
    n_ctrl = 10
    pw_bounds = [(0.0, 1.0)] * n_ctrl
    de_deadline = start_time + 30.0

    try:
        def timed_pw_objective(ctrl):
            if time.time() > de_deadline:
                raise TimeoutError()
            return _pw_linear_objective(ctrl)

        result_pw = differential_evolution(
            timed_pw_objective,
            bounds=pw_bounds,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=43,
            polish=True,
            init='sobol'
        )

        if result_pw.x is not None:
            h_pw = _project_to_feasible(_construct_from_pw_linear(result_pw.x))
            val_pw = _correlation_value(h_pw)
            if val_pw < best_val:
                best_val = val_pw
                best_h = h_pw.copy()
    except TimeoutError:
        pass
    except Exception:
        pass

    # Phase 2: Multi-start SLSQP with epigraph formulation
    # Minimize t subject to corr(h) <= t, sum(h)=50, 0<=h<=1
    rng = np.random.default_rng(123)
    
    starts = [best_h.copy()]
    
    # Add some diverse starts
    for _ in range(5):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        starts.append(h_rand)
        
    # Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    # Blocks
    for block_size in [4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
        
    slsqp_deadline = start_time + 110.0
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 5.0:
            break
        
        h_start = _project_to_feasible(h_start)
        
        # Epigraph formulation
        # x = [h (100 vars), t (1 var)]
        # Minimize t
        # Constraints:
        # 1. sum(h) - 50 = 0
        # 2. corr(h) - t <= 0 (199 constraints)
        # 3. 0 <= h <= 1
        # 4. t >= 0
        
        def epigraph_obj(x):
            return x[N] # t
        
        def epigraph_constraint_sum(x):
            return np.sum(x[:N]) - N / 2.0
            
        def epigraph_constraint_corr(x):
            h = x[:N]
            t = x[N]
            corr = np.correlate(h, 1 - h, mode="full")
            # We need corr - t <= 0
            # Return array of (corr - t)
            return corr - t

        # SLSQP requires constraints in the form g(x) >= 0 for inequality?
        # No, scipy minimize constraints:
        # type 'ineq': fun(x) >= 0
        # So we need t - corr >= 0
        
        def epigraph_constraint_corr_ineq(x):
            h = x[:N]
            t = x[N]
            corr = np.correlate(h, 1 - h, mode="full")
            return t - corr

        try:
            x0 = np.concatenate([h_start, [best_val * N / 2.0]]) # Initial t
            
            res = minimize(
                epigraph_obj,
                x0,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N + [(0.0, 100.0)],
                constraints=[
                    {'type': 'eq', 'fun': epigraph_constraint_sum},
                    {'type': 'ineq', 'fun': epigraph_constraint_corr_ineq}
                ],
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success:
                h_opt = res.x[:N]
                h_opt = _project_to_feasible(h_opt)
                val_opt = _correlation_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 3: Local Search
    if time.time() < deadline - 20.0:
        try:
            local_time_limit = deadline - 15.0
            best_h = _local_search_alignment(best_h, time_limit=local_time_limit)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # Final projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
