"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Combined approach:
- Compact basis parameterization + global optimization (DE, L-BFGS-B, basin hopping)
- O(N) fast incremental autoconvolution update for local search (from inspiration)
- Greedy single-flip and block-flip local search with exact score evaluation
- Simulated annealing with fast evaluation
- Chirp, Barker-like, and golden-ratio structured initializations (from inspiration)
- Direct h-space smooth optimization with multi-beta annealing (from inspiration)
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
    """Compute full autoconvolution array."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return ac

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _score_from_ac(ac, h):
    """Compute score from precomputed autoconvolution."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """Soft approximation of max|ac| using log-sum-exp."""
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    if max_val < 1e-12:
        return 0.0
    shifted = abs_ac - max_val
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return max_val + log_sum

def _score_smooth(h, beta=50.0):
    """Smooth surrogate for the objective."""
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
    """Create a set of basis functions for parameterizing h."""
    x = np.arange(n, dtype=float)
    basis = []
    basis.append(np.ones(n))
    for k in range(1, 8):
        basis.append(np.cos(2 * np.pi * k * x / n))
    for k in range(1, 8):
        basis.append(np.sin(2 * np.pi * k * x / n))
    for center_frac in [0.2, 0.4, 0.6, 0.8]:
        mu = center_frac * n
        sigma = n / 8
        basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
    basis.append(np.cos(np.pi * x / n))
    basis.append((x - n/2)**2 / (n**2))
    basis.append(x / n - 0.5)
    for k in [10, 15, 20]:
        basis.append(np.cos(2 * np.pi * k * x / n))
    return np.array(basis[:n_basis])

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    """Generate a sequence with good autocorrelation properties."""
    h = np.zeros(n)
    for i in range(n):
        if i % 4 == 0:
            h[i] = 1.0
        elif i % 4 == 1:
            h[i] = -1.0
        elif i % 4 == 2:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _gen_golden_ratio(n):
    """Generate a sequence based on golden ratio properties."""
    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    frac = (i * phi) % 1.0
    return 2 * frac - 1

def _flip_ac_update(ac, h, idx):
    """
    O(N) incremental update of autoconvolution after flipping h[idx].
    """
    new_ac = ac.copy()
    delta = -2.0 * h[idx]
    k_start = idx
    k_end = idx + N
    k_start = max(0, k_start)
    k_end = min(2 * N - 1, k_end)
    for k in range(k_start, k_end):
        j = k - idx
        if 0 <= j < N:
            new_ac[k] += 2.0 * delta * h[j]
    k_self = 2 * idx
    if 0 <= k_self < 2 * N - 1:
        new_ac[k_self] += delta * delta
    return new_ac, delta

def _local_search_flip_fast(h, ac, max_passes=10, deadline=None):
    """
    Fast local search using O(N) flip evaluation.
    """
    h = h.copy()
    ac = ac.copy()
    best_val = _score_from_ac(ac, h)
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
            h[i] = -old_val
            new_ac, _ = _flip_ac_update(ac, h, i)
            s = np.sum(h)
            if abs(s) < 1e-9:
                h[i] = old_val
                continue
            new_val = np.max(np.abs(new_ac)) / (s * s)
            if new_val < best_val:
                best_val = new_val
                ac = new_ac
                improved = True
            else:
                h[i] = old_val
                ac, _ = _flip_ac_update(ac, h, i)
        if not improved:
            break
    return h, ac, best_val

def _local_search_perturb_fast(h, ac, max_passes=5, deadline=None, step_size=0.05):
    """
    Fast local search with value perturbations using O(N) incremental update.
    """
    h = h.copy()
    ac = ac.copy()
    best_val = _score_from_ac(ac, h)
    rng = np.random.default_rng(888)
    
    def _perturb_ac(ac, h, i, delta):
        new_ac = ac.copy()
        k_start = max(0, i)
        k_end = min(2 * N - 1, i + N)
        for k in range(k_start, k_end):
            j = k - i
            if 0 <= j < N:
                new_ac[k] += delta * h[j] + delta * h[i]
        k_self = 2 * i
        if 0 <= k_self < 2 * N - 1:
            new_ac[k_self] += delta * delta
        return new_ac
    
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
            best_local_ac = None
            for delta in [step_size, -step_size, 2*step_size, -2*step_size,
                         0.5*step_size, -0.5*step_size]:
                h[i] = old_val + delta
                new_ac = _perturb_ac(ac, h, i, delta)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    continue
                new_val = np.max(np.abs(new_ac)) / (s * s)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
                    best_local_ac = new_ac
            if best_local != old_val:
                h[i] = best_local
                ac = best_local_ac
                best_val = best_local_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, ac, best_val

def _block_flip_search(h, ac, max_passes=5, deadline=None, block_sizes=(1, 3, 5, 10)):
    """
    Try flipping blocks of consecutive indices.
    """
    h = h.copy()
    ac = ac.copy()
    best_val = _score_from_ac(ac, h)
    rng = np.random.default_rng(555)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for _trial in range(50):
            if deadline is not None and time.time() > deadline:
                break
            bs = rng.choice(block_sizes)
            start = rng.integers(0, N - bs + 1)
            block = list(range(start, start + bs))
            
            for i in block:
                h[i] = -h[i]
            new_ac = _autoconv_full(h)
            s = np.sum(h)
            if abs(s) < 1e-9:
                for i in block:
                    h[i] = -h[i]
                continue
            new_val = np.max(np.abs(new_ac)) / (s * s)
            if new_val < best_val:
                best_val = new_val
                ac = new_ac
                improved = True
            else:
                for i in block:
                    h[i] = -h[i]
        if not improved:
            break
    return h, ac, best_val

def _gaussian_perturbation(h, ac, scale=0.1, deadline=None, max_iters=200):
    """
    Simulated annealing with Gaussian perturbations, using FFT for full recompute.
    """
    h_curr = h.copy()
    ac_curr = ac.copy()
    h_best = h.copy()
    best_val = _score_from_ac(ac_curr, h_curr)
    curr_val = best_val
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
        
        ac_trial = _autoconv_full(h_trial)
        s = np.sum(h_trial)
        if abs(s) < 1e-9:
            scale *= 0.995
            continue
        trial_val = np.max(np.abs(ac_trial)) / (s * s)
        
        if trial_val < curr_val or rng.random() < np.exp(-(trial_val - curr_val) / max(T, 1e-10)):
            h_curr = h_trial
            ac_curr = ac_trial
            curr_val = trial_val
            if curr_val < best_val:
                best_val = curr_val
                h_best = h_curr.copy()
        
        scale *= 0.995
    
    return h_best, ac_curr, best_val

def _direct_smooth_optimize(h_init, deadline, beta_schedule=None):
    """
    Direct h-space smooth optimization with multi-beta annealing.
    From inspiration: optimize h directly using log-sum-exp surrogate.
    """
    if beta_schedule is None:
        beta_schedule = [20, 50, 100]
    
    h_curr = h_init.copy()
    s0 = np.sum(h_curr)
    if abs(s0) < 1e-6:
        return h_curr, _score(h_curr)
    h_curr = h_curr / s0
    
    best_h = h_curr.copy()
    best_val = _score(h_curr)
    
    for beta in beta_schedule:
        if deadline is not None and time.time() > deadline:
            break
        
        def obj(x):
            return _score_smooth(x, beta=beta)
        
        try:
            res = minimize(obj, h_curr, method='L-BFGS-B',
                           options={'maxiter': 100, 'disp': False, 'gtol': 1e-5})
            h_opt = res.x
            s_opt = np.sum(h_opt)
            if abs(s_opt) > 1e-6:
                h_opt = h_opt / s_opt
                val = _score(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                    h_curr = h_opt.copy()
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_score = np.inf
    best_h = np.ones(N) / N
    best_ac = _autoconv_full(best_h)
    base_score = _score(best_h)
    best_score = base_score
    
    # Create basis functions
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
                        best_ac = _autoconv_full(best_h)
        except Exception:
            pass
    
    # Phase 1b: Structured initializations with direct smooth optimization
    # This is the key structural addition from inspiration: chirp, Barker, golden-ratio
    # sequences optimized directly in h-space with multi-beta annealing
    if time.time() < deadline - 35:
        x = np.arange(N)
        structured_inits = [
            _gen_chirp(N),
            _gen_barker_like(N),
            _gen_golden_ratio(N),
            np.cos(np.pi * x / N),
            np.cos(np.pi * x / (2*N))**2,
            np.exp(-(x - N/2)**2 / (2*(N/4)**2)),
            np.maximum(0, 1 - np.abs(x - N/2)/(N/2)),
        ]
        
        rng = np.random.default_rng(42)
        for i in range(3):
            noise = rng.standard_normal(N)
            kernel = np.ones(10)/10
            smooth = np.convolve(noise, kernel, mode='same')
            structured_inits.append(smooth)
        
        for h_init in structured_inits:
            if time.time() > deadline - 25:
                break
            s0 = np.sum(h_init)
            if abs(s0) < 1e-6:
                continue
            h_opt, val_opt = _direct_smooth_optimize(h_init, deadline - 20)
            if val_opt < best_score:
                best_score = val_opt
                best_h = h_opt.copy()
                best_ac = _autoconv_full(best_h)
    
    # Phase 2: L-BFGS-B refinement with smooth surrogate in basis space
    if time.time() < deadline - 25:
        inits = []
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr = h_curr / s_curr
        params_init, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
        inits.append(params_init)
        
        rng = np.random.default_rng(123)
        for i in range(8):
            params_rand = rng.standard_normal(n_basis) * 0.5
            inits.append(params_rand)
        
        for amp in [0.1, 0.3, 0.5, 0.7, 1.0]:
            params_struct = np.zeros(n_basis)
            params_struct[0] = 1.0
            params_struct[1] = amp * 0.1
            inits.append(params_struct)
        
        for params_init in inits[:8]:
            if time.time() > deadline - 20:
                break
            try:
                def smooth_obj(params):
                    if time.time() > deadline - 18:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _score_smooth(h, beta=80.0)
                
                res_smooth = minimize(
                    smooth_obj, params_init, method='BFGS',
                    options={'maxiter': 300, 'disp': False, 'gtol': 1e-7}
                )
                
                h_smooth = np.dot(res_smooth.x, basis)
                s_smooth = np.sum(h_smooth)
                if abs(s_smooth) > 1e-6:
                    h_smooth = h_smooth / s_smooth
                    params_refined, _, _, _ = np.linalg.lstsq(basis.T, h_smooth, rcond=None)
                
                def lbfgs_obj(params):
                    if time.time() > deadline - 16:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, params_refined, method='L-BFGS-B',
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
                        best_ac = _autoconv_full(best_h)
            except Exception:
                pass
        
        for params_init in inits[8:]:
            if time.time() > deadline - 12:
                break
            try:
                def lbfgs_obj(params):
                    if time.time() > deadline - 10:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, params_init, method='L-BFGS-B',
                    options={'maxiter': 200, 'disp': False, 'gtol': 1e-6}
                )
                h_opt = np.dot(res_lbfgs.x, basis)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val_opt = _score(h_opt)
                    if val_opt < best_score:
                        best_score = val_opt
                        best_h = h_opt.copy()
                        best_ac = _autoconv_full(best_h)
            except Exception:
                pass
    
    # Phase 3: Basin hopping
    if time.time() < deadline - 12:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def basin_obj(params):
                if time.time() > deadline - 8:
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
                        best_ac = _autoconv_full(best_h)
        except Exception:
            pass
    
    # Phase 4: Fast local search using O(N) flip evaluation
    if time.time() < deadline - 12:
        h_flip, ac_flip, val_flip = _local_search_flip_fast(
            best_h, best_ac, max_passes=20, deadline=deadline - 8
        )
        if val_flip < best_score:
            best_score = val_flip
            best_h = h_flip
            best_ac = ac_flip
        
        if time.time() < deadline - 8:
            h_pert, ac_pert, val_pert = _local_search_perturb_fast(
                best_h, best_ac, max_passes=8, deadline=deadline - 5, step_size=0.03
            )
            if val_pert < best_score:
                best_score = val_pert
                best_h = h_pert
                best_ac = ac_pert
    
    # Phase 5: Block flip search
    if time.time() < deadline - 8:
        h_block, ac_block, val_block = _block_flip_search(
            best_h, best_ac, max_passes=5, deadline=deadline - 5
        )
        if val_block < best_score:
            best_score = val_block
            best_h = h_block
            best_ac = ac_block
    
    # Phase 6: Final fast flip polish
    if time.time() < deadline - 6:
        h_final_flip, ac_final_flip, val_final_flip = _local_search_flip_fast(
            best_h, best_ac, max_passes=15, deadline=deadline - 3
        )
        if val_final_flip < best_score:
            best_score = val_final_flip
            best_h = h_final_flip
            best_ac = ac_final_flip
    
    # Phase 7: Gaussian perturbation / simulated annealing
    if time.time() < deadline - 4:
        h_sa, ac_sa, val_sa = _gaussian_perturbation(
            best_h, best_ac, scale=0.05, deadline=deadline - 2, max_iters=300
        )
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa
            best_ac = ac_sa
    
    # Phase 8: Final high-precision polish in basis space
    if time.time() < deadline - 3:
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
                    best_ac = _autoconv_full(best_h)
        except Exception:
            pass
    
    # Phase 9: Final fast flip polish after basis-space refinement
    if time.time() < deadline - 2:
        h_pf, ac_pf, val_pf = _local_search_flip_fast(
            best_h, best_ac, max_passes=10, deadline=deadline - 1
        )
        if val_pf < best_score:
            best_score = val_pf
            best_h = h_pf
            best_ac = ac_pf
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
