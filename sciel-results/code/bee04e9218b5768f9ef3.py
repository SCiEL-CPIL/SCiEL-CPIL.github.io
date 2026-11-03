"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Model-and-optimize approach: parameterize h via a compact set of basis functions
and directly optimize the parameters against the true objective (max absolute
autoconvolution / (sum h)^2) using derivative-free optimization.

The key insight is that the objective is scale-invariant (scaling h by c scales
both numerator and denominator by c^2), so we can fix sum(h)=1 without loss of
generality. This reduces the problem to optimizing N-1 free parameters.

We use a low-dimensional parameterization: h = sum_i a_i * phi_i where phi_i are
smooth basis functions (Gaussian bumps, sines, cosines, etc.). This keeps the
search space small while allowing sufficient flexibility.

Additionally, we exploit the spectral structure: autoconvolution in frequency
domain is |H(f)|^2, so max|ac| relates to the peak of the power spectrum.
We use a spectral surrogate that directly optimizes the ratio of spectral peak
to (sum h)^2, which is smoother and more amenable to gradient-based optimization.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.signal import fftconvolve

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _autoconv_full(h):
    """Compute full autoconvolution and its max absolute value."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return ac, np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _score_and_ac(h):
    """Compute score and full autoconvolution for further analysis."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, None
    ac, max_abs = _autoconv_full(h)
    return max_abs / (s * s), ac

def _soft_max_abs(ac, beta=50.0):
    """
    Soft approximation of max|ac| using log-sum-exp for differentiability.
    Returns a smooth surrogate that approximates max|ac|.
    """
    abs_ac = np.abs(ac)
    # Use log-sum-exp for numerical stability
    max_val = np.max(abs_ac)
    if max_val < 1e-12:
        return 0.0
    shifted = abs_ac - max_val
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return max_val + log_sum

def _score_smooth(h, beta=50.0):
    """
    Smooth surrogate for the objective: soft_max|autoconv| / (sum h)^2.
    This is differentiable and can be used with gradient-based optimizers.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    
    soft_max = _soft_max_abs(ac, beta)
    return soft_max / (s * s)

def _objective_from_params(params, basis, t0, deadline):
    """Objective function for optimizer: returns score or inf if deadline passed."""
    if time.time() > deadline:
        return 1e10
    h = np.dot(params, basis)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _smooth_objective_from_params(params, basis, t0, deadline, beta=50.0):
    """Smooth objective function for gradient-based optimizer."""
    if time.time() > deadline:
        return 1e10
    h = np.dot(params, basis)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    
    soft_max = _soft_max_abs(ac, beta)
    return soft_max / (s * s)

def _make_basis_functions(n, n_basis=30):
    """
    Create a set of basis functions for parameterizing h.
    Returns a 2D array of shape (n_basis, N).
    """
    x = np.arange(n, dtype=float)
    basis = []
    
    # Constant function (needed for sum normalization)
    basis.append(np.ones(n))
    
    # Low-frequency cosines
    for k in range(1, 10):
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Low-frequency sines
    for k in range(1, 10):
        basis.append(np.sin(2 * np.pi * k * x / n))
    
    # Gaussian bumps at different positions with varied widths
    for center_frac in [0.15, 0.3, 0.5, 0.7, 0.85]:
        mu = center_frac * n
        sigma = n / 10
        basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    
    # Quadratic polynomial
    basis.append((x - n/2)**2 / (n**2))
    
    # Linear ramp
    basis.append(x / n - 0.5)
    
    # Higher frequency cosines
    for k in [12, 16, 20, 25]:
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Cubic polynomial for more shape flexibility
    basis.append(((x - n/2) / (n/2))**3)
    
    # Additional basis: alternating sign pattern (helps with cancellation)
    basis.append(np.where(x % 2 == 0, 1.0, -1.0))
    
    # Block patterns for structured cancellation
    block_size = n // 8
    for b in range(4):
        block = np.zeros(n)
        block[b*block_size:(b+1)*block_size] = 1.0
        block[(b+4)*block_size:(b+5)*block_size] = -1.0
        basis.append(block)
    
    return np.array(basis[:n_basis])

def _local_search_perturb(h, max_passes=5, deadline=None, step_size=0.05):
    """
    Local optimization: try small perturbations to each element, keep if improves score.
    Uses coordinate descent style updates with adaptive step size.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(777)
    
    current_step = step_size
    no_improve_count = 0
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Random order for better exploration
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            # Try several perturbation sizes with random direction
            best_local = old_val
            best_local_val = best_val
            for delta in [current_step, -current_step, 0.5*current_step, -0.5*current_step]:
                h[i] = old_val + delta
                new_val = _score(h)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
            if best_local != old_val:
                h[i] = best_local
                best_val = best_local_val
                improved = True
                no_improve_count = 0
            else:
                h[i] = old_val  # revert if no improvement
                no_improve_count += 1
        # Adapt step size
        if not improved:
            break
        if no_improve_count > N // 2:
            current_step *= 0.7
        elif improved:
            current_step = min(current_step * 1.1, step_size)
    return h, best_val

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses randomized order and early termination with adaptive pass count.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(555)
    passes_done = 0
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        passes_done += 1
        if not improved:
            break
    return h, best_val

def _gaussian_perturbation(h, scale=0.1, deadline=None, max_iters=200):
    """
    Simulated annealing style search with Gaussian perturbations.
    Uses adaptive temperature and occasional restarts to escape local minima.
    """
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    
    T_start = 0.5
    T_end = 0.005
    n_iter = max_iters
    no_improve_count = 0
    restart_interval = 50
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, n_iter - 1))
        
        # Generate perturbation with occasional larger jumps
        if rng.random() < 0.08:
            perturb = rng.standard_normal(N) * scale * 3
        else:
            perturb = rng.standard_normal(N) * scale
        h_trial = h_curr + perturb
        
        trial_val = _score(h_trial)
        
        # Metropolis acceptance
        if trial_val < curr_val or rng.random() < np.exp(-(trial_val - curr_val) / max(T, 1e-10)):
            h_curr = h_trial
            curr_val = trial_val
            if curr_val < best_val:
                best_val = curr_val
                h_best = h_curr.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Periodic restart from best if stuck
        if no_improve_count > restart_interval and it < n_iter - 10:
            h_curr = h_best.copy()
            curr_val = best_val
            no_improve_count = 0
        
        # Gradually reduce perturbation scale
        scale *= 0.997
    
    return h_best, best_val

def _peak_suppression_search(h, deadline=None, max_iters=100):
    """
    Targeted search that tries to reduce the maximum autoconvolution peak.
    Identifies the location of the max peak and applies counter-perturbations.
    Uses both single-element and paired perturbations for better convergence.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(888)
    base_step = 0.03
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac, max_abs = _autoconv_full(h)
        if ac is None:
            break
        
        # Find the top 3 peak indices for multi-peak suppression
        abs_ac = np.abs(ac)
        top_k = 3
        peak_indices = np.argsort(abs_ac)[-top_k:][::-1]
        
        for k in peak_indices:
            if deadline is not None and time.time() > deadline:
                break
            
            # Elements that contribute to autoconv at position k
            valid_j = [j for j in range(N) if 0 <= k - j < N]
            if not valid_j:
                continue
            
            # Try perturbing a random subset of contributing elements
            n_perturb = min(3, len(valid_j))
            perturbed_idx = rng.choice(valid_j, size=n_perturb, replace=False)
            
            h_trial = h.copy()
            for j in perturbed_idx:
                # Try to reduce the contribution at the peak
                sign = np.sign(ac[k])
                delta = rng.uniform(0.01, base_step) * (-sign)
                h_trial[j] += delta
                # Also try the complementary element
                kj = k - j
                if 0 <= kj < N and kj != j:
                    h_trial[kj] += delta * 0.5
            
            trial_val = _score(h_trial)
            if trial_val < best_val:
                best_val = trial_val
                h = h_trial.copy()
                base_step = max(0.01, base_step * 0.95)
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence
    
    # Evaluate baseline
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Create basis functions for parameterization
    n_basis = 30
    basis = _make_basis_functions(N, n_basis)
    
    # Phase 1: Global optimization with differential evolution on compact parameterization
    phase1_deadline = deadline - 45
    if time.time() < phase1_deadline:
        bounds = [(-2.0, 2.0)] * n_basis
        
        def de_objective(params):
            if time.time() > phase1_deadline:
                return 1e10
            h = np.dot(params, basis)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _autoconv_max_abs(h) / (s * s)
        
        try:
            res_de = differential_evolution(
                de_objective, bounds, 
                seed=42, maxiter=35, tol=1e-5,
                popsize=12, mutation=(0.3, 1.2), recombination=0.7,
                polish=False
            )
            if res_de.fun < best_score:
                h_de = np.dot(res_de.x, basis)
                s_de = np.sum(h_de)
                if abs(s_de) > 1e-6:
                    h_de = h_de / s_de
                val_de = _score(h_de)
                if val_de < best_score:
                    best_score = val_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: Smooth surrogate optimization with adaptive beta
    phase2_deadline = deadline - 25
    if time.time() < phase2_deadline:
        # Generate diverse initializations
        inits = []
        
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr = h_curr / s_curr
        
        params_init, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
        inits.append(params_init)
        
        rng = np.random.default_rng(123)
        for i in range(8):
            params_rand = rng.standard_normal(n_basis) * 0.4
            inits.append(params_rand)
        
        for amp in [0.2, 0.5, 0.8]:
            params_struct = np.zeros(n_basis)
            params_struct[0] = 1.0
            params_struct[1] = amp * 0.1
            inits.append(params_struct)
        
        # Smooth surrogate optimization with multiple beta values
        for params_init in inits[:8]:
            if time.time() > phase2_deadline - 5:
                break
            
            try:
                # Use adaptive beta: start smooth, get sharper
                for beta in [50.0, 100.0, 200.0]:
                    if time.time() > phase2_deadline - 3:
                        break
                    
                    def smooth_obj(params, b=beta):
                        if time.time() > phase2_deadline - 2:
                            return 1e10
                        h = np.dot(params, basis)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, beta=b)
                    
                    res_smooth = minimize(
                        smooth_obj, params_init, method='BFGS',
                        options={'maxiter': 200, 'disp': False, 'gtol': 1e-7}
                    )
                    
                    h_smooth = np.dot(res_smooth.x, basis)
                    s_smooth = np.sum(h_smooth)
                    if abs(s_smooth) > 1e-6:
                        h_smooth = h_smooth / s_smooth
                        params_init = np.linalg.lstsq(basis.T, h_smooth, rcond=None)[0]
                
                # Final refinement with true objective
                def lbfgs_obj(params):
                    if time.time() > phase2_deadline - 1:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, params_init, method='L-BFGS-B',
                    options={'maxiter': 200, 'disp': False, 'gtol': 1e-8}
                )
                h_opt = np.dot(res_lbfgs.x, basis)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val_opt = _score(h_opt)
                    if val_opt < best_score:
                        best_score = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Basin hopping
    phase3_deadline = deadline - 12
    if time.time() < phase3_deadline:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def basin_obj(params):
                if time.time() > phase3_deadline:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_bh = basinhopping(
                basin_obj, params_curr, 
                niter=20, T=0.1, stepsize=0.3,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'gtol': 1e-7}},
                seed=42
            )
            if res_bh.fun < best_score:
                h_bh = np.dot(res_bh.x, basis)
                s_bh = np.sum(h_bh)
                if abs(s_bh) > 1e-6:
                    h_bh = h_bh / s_bh
                    val_bh = _score(h_bh)
                    if val_bh < best_score:
                        best_score = val_bh
                        best_h = h_bh.copy()
        except Exception:
            pass
    
    # Phase 4: Peak suppression search
    phase4_deadline = deadline - 10
    if time.time() < phase4_deadline:
        h_peak, val_peak = _peak_suppression_search(best_h, deadline=phase4_deadline, max_iters=100)
        if val_peak < best_score:
            best_score = val_peak
            best_h = h_peak.copy()
    
    # Phase 4b: Spectral surrogate optimization
    if time.time() < deadline - 8:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def spectral_obj(params):
                if time.time() > deadline - 8:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                n = len(h)
                size = 2 * n - 1
                fsize = 1
                while fsize < size:
                    fsize *= 2
                H = np.fft.rfft(h, fsize)
                power_spec = np.abs(H) ** 2
                # Soft max of power spectrum approximates peak of autoconv
                beta = 50.0
                max_ps = np.max(power_spec)
                if max_ps < 1e-12:
                    return 0.0
                shifted = power_spec - max_ps
                log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
                soft_peak = max_ps + log_sum
                return soft_peak / (s * s)
            
            res_spec = minimize(
                spectral_obj, params_curr, method='L-BFGS-B',
                options={'maxiter': 150, 'disp': False, 'gtol': 1e-8}
            )
            h_spec = np.dot(res_spec.x, basis)
            s_spec = np.sum(h_spec)
            if abs(s_spec) > 1e-6:
                h_spec = h_spec / s_spec
                val_spec = _score(h_spec)
                if val_spec < best_score:
                    best_score = val_spec
                    best_h = h_spec.copy()
        except Exception:
            pass
    
    # Phase 5: Local search refinement
    phase5_deadline = deadline - 3
    if time.time() < phase5_deadline:
        h_flip, val_flip = _local_search_flip(best_h, max_passes=5, deadline=phase5_deadline - 2)
        if val_flip < best_score:
            best_score = val_flip
            best_h = h_flip.copy()
        
        if time.time() < phase5_deadline - 1:
            h_pert, val_pert = _local_search_perturb(best_h, max_passes=5, deadline=phase5_deadline - 0.5, step_size=0.03)
            if val_pert < best_score:
                best_score = val_pert
                best_h = h_pert.copy()
    
    # Phase 6: Final simulated annealing polish
    if time.time() < deadline - 1:
        h_sa, val_sa = _gaussian_perturbation(best_h, scale=0.03, deadline=deadline - 0.5, max_iters=400)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 0.5:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def final_obj(params):
                if time.time() > deadline - 0.2:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_final = minimize(
                final_obj, params_curr, method='L-BFGS-B',
                options={'maxiter': 500, 'disp': False, 'gtol': 1e-12}
            )
            h_final = np.dot(res_final.x, basis)
            s_final = np.sum(h_final)
            if abs(s_final) > 1e-6:
                h_final = h_final / s_final
                val_final = _score(h_final)
                if val_final < best_score:
                    best_score = val_final
                    best_h = h_final.copy()
        except Exception:
            pass
    
    # Final sign-flip polish after continuous optimization
    if time.time() < deadline - 0.3:
        try:
            h_final_flip, val_final_flip = _local_search_flip(best_h, max_passes=2, deadline=deadline - 0.1)
            if val_final_flip < best_score:
                best_score = val_final_flip
                best_h = h_final_flip.copy()
        except Exception:
            pass
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
