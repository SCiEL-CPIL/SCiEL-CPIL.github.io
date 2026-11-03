"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Combined approach:
- Compact basis parameterization + global optimization (DE, L-BFGS-B, basin hopping)
- O(N) fast incremental autoconvolution update for local search (from inspiration)
- Smooth log-sum-exp surrogate with adaptive beta for gradient-based refinement
- Peak-suppression phase targeting the argmax of the autoconvolution
- Sign-flip local search with efficient incremental updates
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

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _autoconv(h):
    """Compute full autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _soft_max_abs(ac, beta=50.0):
    """
    Soft approximation of max|ac| using log-sum-exp for differentiability.
    Returns a smooth surrogate that approximates max|ac|.
    """
    abs_ac = np.abs(ac)
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

def _make_basis_functions(n, n_basis=25):
    """
    Create a set of basis functions for parameterizing h.
    Returns a 2D array of shape (n_basis, N).
    """
    x = np.arange(n, dtype=float)
    basis = []
    
    # Constant function (needed for sum normalization)
    basis.append(np.ones(n))
    
    # Low-frequency cosines
    for k in range(1, 8):
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Low-frequency sines
    for k in range(1, 8):
        basis.append(np.sin(2 * np.pi * k * x / n))
    
    # Gaussian bumps at different positions
    for center_frac in [0.2, 0.4, 0.6, 0.8]:
        mu = center_frac * n
        sigma = n / 8
        basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    
    # Quadratic polynomial
    basis.append((x - n/2)**2 / (n**2))
    
    # Linear ramp
    basis.append(x / n - 0.5)
    
    # Higher frequency cosine
    for k in [10, 15, 20]:
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    return np.array(basis[:n_basis])

def _try_flip(h, idx, ac, cur_sum):
    """
    Efficiently compute the new autoconvolution and score after flipping h[idx].
    Returns (new_ac, new_sum, new_score).
    """
    val = h[idx]
    delta = -2.0 * val
    new_ac = ac.copy()
    start = idx
    end = idx + N
    new_ac[start:end] += -4.0 * val * h
    new_ac[idx] += 4.0 * val * val
    new_sum = cur_sum + delta
    nsq = new_sum ** 2
    if nsq < 1e-10:
        return new_ac, new_sum, np.inf
    return new_ac, new_sum, float(np.max(np.abs(new_ac)) / nsq)

def _local_search_flip(h, max_passes=100, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses O(N) incremental update for efficiency.
    """
    h = h.copy()
    ac = _autoconv(h)
    cur_sum = float(np.sum(h))
    best_val = float(np.max(np.abs(ac)) / (cur_sum * cur_sum))
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        best_j = -1
        best_js = best_val
        best_new_ac = None
        best_new_sum = cur_sum
        
        # Evaluate all possible single flips
        for j in range(N):
            if deadline is not None and time.time() > deadline:
                break
            new_ac, new_sum, new_score = _try_flip(h, j, ac, cur_sum)
            if new_score < best_js:
                best_js = new_score
                best_j = j
                best_new_ac = new_ac
                best_new_sum = new_sum
        
        if best_j == -1 or best_js >= best_val - 1e-12:
            break
            
        h[best_j] = -h[best_j]
        ac = best_new_ac
        cur_sum = best_new_sum
        best_val = best_js
        improved = True
        
        if not improved:
            break
    return h, best_val

def _local_search_perturb(h, max_passes=5, deadline=None, step_size=0.05):
    """
    Local optimization: try small perturbations to each element, keep if improves score.
    Uses coordinate descent style updates.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(777)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            best_local = old_val
            best_local_val = best_val
            for delta in [step_size, -step_size, 2*step_size, -2*step_size, 
                         0.5*step_size, -0.5*step_size]:
                h[i] = old_val + delta
                new_val = _score(h)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
            if best_local != old_val:
                h[i] = best_local
                best_val = best_local_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _gaussian_perturbation(h, scale=0.1, deadline=None, max_iters=200):
    """
    Simulated annealing style search with Gaussian perturbations.
    """
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    
    T_start = 0.5
    T_end = 0.01
    n_iter = max_iters
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, n_iter - 1))
        
        perturb = rng.standard_normal(N) * scale
        h_trial = h_curr + perturb
        trial_val = _score(h_trial)
        
        if trial_val < curr_val or rng.random() < np.exp(-(trial_val - curr_val) / max(T, 1e-10)):
            h_curr = h_trial
            curr_val = trial_val
            if curr_val < best_val:
                best_val = curr_val
                h_best = h_curr.copy()
        
        scale *= 0.995
    
    return h_best, best_val

def _peak_suppression(h, basis, max_iters=50, deadline=None):
    """
    Peak suppression phase: directly target the argmax of the autoconvolution
    by adjusting parameters to reduce the peak value.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, _score(h)
    h = h / s
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autoconv(h)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        # Try to reduce the peak by adjusting parameters
        # Compute gradient of peak value w.r.t. parameters (approximate)
        # For simplicity, try small perturbations in parameter space
        params, _, _, _ = np.linalg.lstsq(basis.T, h, rcond=None)
        
        best_params = params.copy()
        best_val = _score(h)
        rng = np.random.default_rng(12345)
        
        for _ in range(20):
            if deadline is not None and time.time() > deadline:
                break
            # Perturb parameters
            pert = rng.standard_normal(len(params)) * 0.1
            new_params = params + pert
            new_h = np.dot(new_params, basis)
            new_s = np.sum(new_h)
            if abs(new_s) < 1e-6:
                continue
            new_h = new_h / new_s
            new_val = _score(new_h)
            if new_val < best_val:
                best_val = new_val
                best_params = new_params
        
        h = np.dot(best_params, basis)
        s = np.sum(h)
        if abs(s) < 1e-6:
            break
        h = h / s
    
    return h, _score(h)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    n_basis = 25
    basis = _make_basis_functions(N, n_basis)
    
    # Phase 1: Global optimization with differential evolution
    if time.time() < deadline - 50:
        bounds = [(-2.0, 2.0)] * n_basis
        
        def de_objective(params):
            if time.time() > deadline - 45:
                return 1e10
            h = np.dot(params, basis)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _autoconv_max_abs(h) / (s * s)
        
        try:
            res_de = differential_evolution(
                de_objective, bounds, 
                seed=42, maxiter=40, tol=1e-5,
                popsize=15, mutation=(0.3, 1.2), recombination=0.7,
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
    
    # Phase 2: L-BFGS-B refinement with smooth surrogate and adaptive beta
    if time.time() < deadline - 30:
        inits = []
        
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr = h_curr / s_curr
        
        if len(basis.shape) == 2:
            params_init, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            inits.append(params_init)
        
        rng = np.random.default_rng(123)
        for i in range(10):
            params_rand = rng.standard_normal(n_basis) * 0.5
            inits.append(params_rand)
        
        for amp in [0.1, 0.3, 0.5, 0.7, 1.0]:
            params_struct = np.zeros(n_basis)
            params_struct[0] = 1.0
            params_struct[1] = amp * 0.1
            inits.append(params_struct)
        
        # Adaptive beta schedule: 60 -> 120 -> 250
        beta_schedule = [60.0, 120.0, 250.0]
        
        for params_init in inits[:8]:
            if time.time() > deadline - 25:
                break
            
            try:
                current_params = params_init.copy()
                
                for beta in beta_schedule:
                    if time.time() > deadline - 22:
                        break
                    
                    def smooth_obj(params, b=beta):
                        if time.time() > deadline - 20:
                            return 1e10
                        h = np.dot(params, basis)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, b)
                    
                    res_smooth = minimize(
                        smooth_obj, current_params, method='BFGS',
                        options={'maxiter': 200, 'disp': False, 'gtol': 1e-7}
                    )
                    current_params = res_smooth.x
                
                # Final refinement with true objective
                def lbfgs_obj(params):
                    if time.time() > deadline - 18:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, current_params, method='L-BFGS-B',
                    options={'maxiter': 300, 'disp': False, 'gtol': 1e-8}
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
    
    # Phase 3: Peak suppression
    if time.time() < deadline - 20:
        try:
            h_ps, val_ps = _peak_suppression(best_h, basis, max_iters=30, deadline=deadline-18)
            if val_ps < best_score:
                best_score = val_ps
                best_h = h_ps.copy()
        except Exception:
            pass
    
    # Phase 4: Basin hopping
    if time.time() < deadline - 15:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def basin_obj(params):
                if time.time() > deadline - 10:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_bh = basinhopping(
                basin_obj, params_curr, 
                niter=25, T=0.1, stepsize=0.3,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 150, 'gtol': 1e-7}},
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
    
    # Phase 5: Local search refinement on the full N-dimensional space
    if time.time() < deadline - 10:
        # Sign flip search with efficient incremental updates
        h_flip, val_flip = _local_search_flip(best_h, max_passes=100, deadline=deadline-7)
        if val_flip < best_score:
            best_score = val_flip
            best_h = h_flip.copy()
        
        # Perturbation search
        if time.time() < deadline - 5:
            h_pert, val_pert = _local_search_perturb(best_h, max_passes=6, deadline=deadline-3, step_size=0.03)
            if val_pert < best_score:
                best_score = val_pert
                best_h = h_pert.copy()
    
    # Phase 6: Final Gaussian perturbation / simulated annealing
    if time.time() < deadline - 3:
        h_sa, val_sa = _gaussian_perturbation(best_h, scale=0.05, deadline=deadline-1, max_iters=400)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def final_obj(params):
                if time.time() > deadline - 1:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_final = minimize(
                final_obj, params_curr, method='L-BFGS-B',
                options={'maxiter': 500, 'disp': False, 'gtol': 1e-10}
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
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
