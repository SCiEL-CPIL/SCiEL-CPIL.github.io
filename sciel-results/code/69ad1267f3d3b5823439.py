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
    Efficiently compute the new max|autoconv|/sum^2 after flipping h[idx] in O(N) time.
    The autoconvolution change for flipping h[idx] from v to -v is:
      delta_ac[k] = -4*v*h[k-idx] for k in [idx, idx+N-1]
      delta_ac[idx] += 4*v^2  (correction for the double-counted term)
    We track the current max and update it incrementally without copying the full array.
    """
    v = h[idx]
    new_sum = cur_sum - 2.0 * v
    nsq = new_sum * new_sum
    if nsq < 1e-10:
        return None, new_sum, np.inf
    
    # Compute the perturbation to autoconvolution
    # delta_ac[k] = -4*v*h[k-idx] for k in [idx, idx+N-1], plus 4*v^2 at k=idx
    # The new ac at position k is: ac[k] - 4*v*h[k-idx] (for k in [idx, idx+N-1])
    # and ac[k] unchanged elsewhere
    # But we also need to add 4*v^2 at position idx (since h[idx]*h[idx] changes)
    
    # For positions in [idx, idx+N-1]:
    # new_ac[k] = ac[k] - 4*v*h[k-idx]
    # At k=idx: new_ac[idx] = ac[idx] - 4*v*h[0] + 4*v^2  (wait, let me re-derive)
    
    # Actually: ac[k] = sum_j h[j]*h[k-j]
    # After flipping h[idx]: h'[idx] = -h[idx]
    # ac'[k] = sum_j h'[j]*h'[k-j]
    # = ac[k] - 2*v*h[k-idx] - 2*v*h[idx-(k)] + 4*v^2 * (1 if k-idx==idx else 0)
    # Wait, more carefully:
    # ac'[k] = ac[k] - v*h[k-idx] - h[idx]*v*h[k-idx] ... no
    # Let me think again. ac[k] = sum_j h[j] h[k-j]
    # ac'[k] = sum_j h'[j] h'[k-j] where h'[idx]=-v, h'[j]=h[j] for j!=idx
    # = sum_{j!=idx} h[j]h[k-j] + h'[idx]h'[k-idx] + h'[k-idx]h'[idx] - (terms where j=idx or k-j=idx already in sum)
    # Actually: ac'[k] = ac[k] - v*h[k-idx] - h[idx]*h[k-idx] + (-v)*(-v) if idx==k-idx
    # Hmm, let me be more careful:
    # ac[k] = sum_j h[j]*h[k-j]
    # The terms involving h[idx] are: j=idx gives h[idx]*h[k-idx], and k-j=idx i.e. j=k-idx gives h[k-idx]*h[idx]
    # If idx != k-idx (i.e., k != 2*idx), these are two distinct terms, each = v*h[k-idx]
    # If idx == k-idx (i.e., k = 2*idx), it's one term = v*v
    # After flip:
    # j=idx term becomes (-v)*h[k-idx] = -v*h[k-idx]
    # j=k-idx term becomes h[k-idx]*(-v) = -v*h[k-idx]  (if k-idx != idx)
    # So change at k (when k != 2*idx): -2v*h[k-idx] - 2v*h[k-idx] = -4v*h[k-idx]
    # When k = 2*idx: original term was v^2, new term is v^2, change = 0? No:
    #   j=idx=k-idx, term was v*v=v^2, new is (-v)*(-v)=v^2, change=0
    # Wait that can't be right. Let me reconsider.
    
    # For k in [0, 2N-2]:
    # ac'[k] - ac[k] = sum_j (h'[j]-h[j])h'[k-j] + sum_j h[j](h'[k-j]-h[k-j])
    # Only h'[idx]-h[idx] = -2v is nonzero
    # = (-2v)*h'[k-idx] + h[idx]*(-2v)  [when k-idx != idx]
    # = (-2v)*h'[k-idx] + h[idx]*(-2v)
    # h'[k-idx] = h[k-idx] if k-idx != idx, else -v
    # If k-idx != idx: change = -2v*h[k-idx] - 2v*h[idx] = -2v*(h[k-idx]+h[idx])... 
    # Hmm wait, that's not symmetric. Let me redo:
    
    # ac'[k] = sum_j h'[j] h'[k-j]
    # ac[k] = sum_j h[j] h[k-j]
    # ac'[k] - ac[k] = sum_j (h'[j]h'[k-j] - h[j]h[k-j])
    # For j where both j!=idx and k-j!=idx: h'[j]h'[k-j] = h[j]h[k-j], contribution 0
    # For j=idx (and k-idx != idx): h'[idx]h'[k-idx] - h[idx]h[k-idx] = (-v)(h[k-idx]) - v*h[k-idx] = -2v*h[k-idx]
    # For j=k-idx (and idx != k-idx, i.e., j!=idx): h'[k-idx]h'[idx] - h[k-idx]h[idx] = h[k-idx]*(-v) - h[k-idx]*v = -2v*h[k-idx]
    # For j=idx=k-idx (i.e., k=2idx): h'[idx]h'[idx] - h[idx]h[idx] = v^2 - v^2 = 0
    # Total: if k != 2*idx and idx <= k <= idx+N-1 (so k-idx is valid):
    #   change = -4v*h[k-idx]
    # if k = 2*idx: change = 0
    # if k-idx out of [0,N-1]: change = 0
    
    # So: new_ac[k] = ac[k] - 4*v*h[k-idx] for k in [idx, idx+N-1] where k != 2*idx
    #             new_ac[k] = ac[k] for k = 2*idx (if in range)
    #             new_ac[k] = ac[k] for k outside [idx, idx+N-1]
    
    # To find max|new_ac| efficiently:
    # Positions outside [idx, idx+N-1] are unchanged: max is max(|ac[k]| for k outside)
    # Positions inside: new_ac[k] = ac[k] - 4*v*h[k-idx]
    # At k=2*idx: unchanged
    
    # We can compute the max over the changed region and compare with the unchanged region max.
    # But computing the unchanged region max requires knowing which positions are unchanged.
    # Simpler: just compute max over all positions using the formula.
    
    # For speed, compute the affected slice and take max there, then compare with precomputed max outside.
    # But we don't have a precomputed max outside. Let's just compute it directly but efficiently.
    
    # Actually for N=400, the affected region has at most N elements. Let's just compute it.
    lo = idx
    hi = idx + N  # exclusive upper bound for k-idx in [0, N-1]
    if hi > 2 * N - 1:
        hi = 2 * N - 1
    
    # new_ac in [lo, hi) = ac[lo:hi] - 4*v*h[0:hi-lo]
    # except at k=2*idx if 2*idx is in [lo, hi): it should be ac[2*idx] unchanged
    affected = ac[lo:hi] - 4.0 * v * h[:hi - lo]
    two_idx = 2 * idx
    if lo <= two_idx < hi:
        affected[two_idx - lo] = ac[two_idx]  # unchanged
    
    max_affected = float(np.max(np.abs(affected)))
    
    # For positions outside [lo, hi): they are unchanged
    # We need max(|ac[k]|) for k in [0, lo) union [hi, 2N-2]
    # Quick approach: compute max of full ac, if the global max is in the affected region and 
    # was the only large value, we might have lost it. But for correctness, let's just check.
    # Since N=400, 2N-2=798, and the affected region is at most 400 elements,
    # the unaffected region is at most 398 elements. Computing max over it is cheap.
    
    # Build mask or just use slicing
    if lo > 0:
        max_before = float(np.max(np.abs(ac[:lo])))
    else:
        max_before = 0.0
    if hi < 2 * N - 1:
        max_after = float(np.max(np.abs(ac[hi:])))
    else:
        max_after = 0.0
    max_unaffected = max(max_before, max_after)
    
    new_max = max(max_affected, max_unaffected)
    new_score = new_max / nsq
    
    return None, new_sum, new_score

def _local_search_flip(h, max_passes=200, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses O(N) incremental update for efficiency. In-place flip application.
    """
    h = h.copy()
    ac = _autoconv(h)
    cur_sum = float(np.sum(h))
    best_val = float(np.max(np.abs(ac)) / (cur_sum * cur_sum))
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        best_j = -1
        best_js = best_val
        best_new_sum = cur_sum
        
        # Evaluate all possible single flips
        for j in range(N):
            if deadline is not None and time.time() > deadline:
                break
            _, new_sum, new_score = _try_flip(h, j, ac, cur_sum)
            if new_score < best_js:
                best_js = new_score
                best_j = j
                best_new_sum = new_sum
        
        if best_j == -1 or best_js >= best_val - 1e-12:
            break
            
        # Apply flip in-place and update ac
        v = h[best_j]
        h[best_j] = -v
        # Update ac: ac[k] -= 4*v*h[k-best_j] for k in [best_j, best_j+N-1], k != 2*best_j
        lo = best_j
        hi = min(best_j + N, 2 * N - 1)
        ac[lo:hi] -= 4.0 * v * h[:hi - lo]
        two_idx = 2 * best_j
        if lo <= two_idx < hi:
            # This position should NOT have been changed (net effect is 0)
            # But we subtracted 4*v*h[0] = 4*v*v at that position
            # The correct change is 0, so we need to add back 4*v*v
            ac[two_idx] += 4.0 * v * v
        cur_sum = best_new_sum
        best_val = best_js
    
    return h, best_val

def _local_search_perturb(h, max_passes=8, deadline=None, step_size=0.03):
    """
    Local optimization: try small perturbations to each element, keep if improves score.
    Uses coordinate descent style updates with FFT-based scoring.
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
    if time.time() < deadline - 45:
        bounds = [(-2.0, 2.0)] * n_basis
        
        def de_objective(params):
            if time.time() > deadline - 40:
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
    
    # Phase 2: L-BFGS-B refinement with smooth surrogate and adaptive beta
    if time.time() < deadline - 28:
        inits = []
        
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr = h_curr / s_curr
        
        if len(basis.shape) == 2:
            params_init, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            inits.append(params_init)
        
        rng = np.random.default_rng(123)
        for i in range(12):
            params_rand = rng.standard_normal(n_basis) * 0.5
            inits.append(params_rand)
        
        for amp in [0.1, 0.3, 0.5, 0.7, 1.0]:
            params_struct = np.zeros(n_basis)
            params_struct[0] = 1.0
            params_struct[1] = amp * 0.1
            inits.append(params_struct)
        
        # Adaptive beta schedule: 60 -> 120 -> 250
        beta_schedule = [60.0, 120.0, 250.0]
        
        for params_init in inits[:10]:
            if time.time() > deadline - 22:
                break
            
            try:
                current_params = params_init.copy()
                
                for beta in beta_schedule:
                    if time.time() > deadline - 20:
                        break
                    
                    def smooth_obj(params, b=beta):
                        if time.time() > deadline - 18:
                            return 1e10
                        h = np.dot(params, basis)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, b)
                    
                    res_smooth = minimize(
                        smooth_obj, current_params, method='BFGS',
                        options={'maxiter': 150, 'disp': False, 'gtol': 1e-7}
                    )
                    current_params = res_smooth.x
                
                # Final refinement with true objective
                def lbfgs_obj(params):
                    if time.time() > deadline - 16:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, current_params, method='L-BFGS-B',
                    options={'maxiter': 250, 'disp': False, 'gtol': 1e-8}
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
    if time.time() < deadline - 18:
        try:
            h_ps, val_ps = _peak_suppression(best_h, basis, max_iters=25, deadline=deadline-16)
            if val_ps < best_score:
                best_score = val_ps
                best_h = h_ps.copy()
        except Exception:
            pass
    
    # Phase 4: Basin hopping
    if time.time() < deadline - 14:
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
                niter=20, T=0.1, stepsize=0.3,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 120, 'gtol': 1e-7}},
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
        # Sign flip search with efficient incremental updates (now faster)
        h_flip, val_flip = _local_search_flip(best_h, max_passes=1200, deadline=deadline-8)
        if val_flip < best_score:
            best_score = val_flip
            best_h = h_flip.copy()
        
        # Perturbation search
        if time.time() < deadline - 6:
            h_pert, val_pert = _local_search_perturb(best_h, max_passes=8, deadline=deadline-4, step_size=0.03)
            if val_pert < best_score:
                best_score = val_pert
                best_h = h_pert.copy()
    
    # Phase 6: Final Gaussian perturbation / simulated annealing (more iterations)
    if time.time() < deadline - 4:
        h_sa, val_sa = _gaussian_perturbation(best_h, scale=0.06, deadline=deadline-2, max_iters=15000)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Final high-precision polish (tighter tolerance, more iterations)
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
                options={'maxiter': 800, 'disp': False, 'gtol': 1e-12}
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
    
    # Second polish pass with even tighter tolerance
    if time.time() < deadline - 1:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def final_obj2(params):
                if time.time() > deadline - 0.5:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_final2 = minimize(
                final_obj2, params_curr, method='L-BFGS-B',
                options={'maxiter': 300, 'disp': False, 'gtol': 1e-14}
            )
            h_final2 = np.dot(res_final2.x, basis)
            s_final2 = np.sum(h_final2)
            if abs(s_final2) > 1e-6:
                h_final2 = h_final2 / s_final2
                val_final2 = _score(h_final2)
                if val_final2 < best_score:
                    best_score = val_final2
                    best_h = h_final2.copy()
        except Exception:
            pass
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
