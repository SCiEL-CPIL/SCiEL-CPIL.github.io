"""Analytic uncertainty bound with exact gradient + multi-stage optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    # Precompute I_n = integral x^{2n} e^{-x^2} dx using recurrence
    # I_0 = sqrt(pi), I_n = (2n-1)/2 * I_{n-1}
    I = np.zeros(14)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 14):
        I[n] = (2*n - 1) / 2.0 * I[n-1]
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)  # m0, m2, m4
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            if pp == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]
    
    # For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            if qq == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # Precompute dp/dac matrix: dp[7] for each c in 0,1,2
    # dp_k/da_c = base_k[c] + (dp_k/da12) * da12[c]
    # base_k[c]: coefficient of a_c in p_k before a12 contribution
    # p0: a0+3a4+105a8+945a12 -> base=[1,3,105], da12_coeff=945
    # p1: -6a4-420a8-7560a12 -> base=[0,-6,-420], da12_coeff=-7560
    # p2: a4+210a8+13860a12 -> base=[0,1,210], da12_coeff=13860
    # p3: -28a8-6930a12 -> base=[0,0,-28], da12_coeff=-6930
    # p4: a8+1155a12 -> base=[0,0,1], da12_coeff=1155
    # p5: -66a12 -> base=[0,0,0], da12_coeff=-66
    # p6: a12 -> base=[0,0,0], da12_coeff=1
    
    base = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0]
    ])
    da12_coeff = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = base[:, c] + da12_coeff * da12[c]
        
        # dm_k/da_c = 2*sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * I[idx]
                dm[k] += 2.0 * dpi * s
        
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += q[j] * I[idx]
                dn[k] += 2.0 * 2.0*np.pi * dqi * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    best_val = np.inf
    best_params = None
    
    def try_optimize(start, method='L-BFGS-B', maxiter=3000, ftol=1e-14, gtol=1e-13):
        nonlocal best_val, best_params
        if time.time() > _DEADLINE - 25:
            return
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            else:
                res = minimize(_objective, start, method=method,
                              options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 1: Coarse grid search to find good basin
    # The seed is around [-1.13, 0.04, 0.0003]
    # Explore a range around it
    grid_a0 = np.linspace(-2.5, 0.5, 13)
    grid_a4 = np.linspace(-0.1, 0.2, 13)
    grid_a8 = np.linspace(-0.002, 0.003, 11)
    
    # Evaluate on a subset of the grid (coarse)
    grid_results = []
    for a0 in grid_a0:
        for a4 in grid_a4:
            for a8 in grid_a8:
                if time.time() > _DEADLINE - 60:
                    break
                try:
                    val, _ = _bound_and_grad(np.array([a0, a4, a8]))
                    if np.isfinite(val) and val < 1e6:
                        grid_results.append((val, np.array([a0, a4, a8])))
                except Exception:
                    pass
            if time.time() > _DEADLINE - 60:
                break
        if time.time() > _DEADLINE - 60:
            break
    
    # Sort by value and take top candidates
    grid_results.sort(key=lambda x: x[0])
    
    # Stage 2: L-BFGS-B from top grid points + seed + random starts
    starts = []
    
    # Always include the seed
    starts.append(np.array([-1.13, 0.04, 0.000308642]))
    
    # Top grid points
    for val, params in grid_results[:20]:
        starts.append(params.copy())
    
    # Additional structured starts
    extra_starts = [
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-1.5, 0.02, 0.0001]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-2.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-1.15, 0.035, 0.00025]),
    ]
    for s in extra_starts:
        starts.append(s)
    
    # Random starts
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal([-1.1, 0.04, 0.0003], [0.5, 0.05, 0.001], 3))
    
    # Run L-BFGS-B from each start
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try_optimize(start, 'L-BFGS-B', maxiter=2000, ftol=1e-13, gtol=1e-12)
    
    # Stage 3: High-precision polish from best
    if best_params is not None and time.time() < _DEADLINE - 20:
        # Nelder-Mead for local refinement
        try_optimize(best_params, 'Nelder-Mead', maxiter=50000, ftol=1e-16)
        
        # Final L-BFGS-B with very tight tolerances
        if time.time() < _DEADLINE - 10:
            try_optimize(best_params, 'L-BFGS-B', maxiter=5000, ftol=1e-15, gtol=1e-14)
    
    # Stage 4: Adaptive line-search polish around best point
    if best_params is not None and time.time() < _DEADLINE - 5:
        # Try multiple small perturbations in random directions with decreasing step sizes
        rng2 = np.random.default_rng(777)
        for step_scale in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]:
            if time.time() > _DEADLINE - 5:
                break
            for i in range(8):
                if time.time() > _DEADLINE - 5:
                    break
                direction = rng2.standard_normal(3)
                direction /= np.linalg.norm(direction)
                perturbed = best_params + step_scale * direction
                try_optimize(perturbed, 'L-BFGS-B', maxiter=2000, ftol=1e-15, gtol=1e-14)
                # Also try the opposite direction
                if time.time() > _DEADLINE - 5:
                    break
                perturbed2 = best_params - step_scale * direction
                try_optimize(perturbed2, 'L-BFGS-B', maxiter=2000, ftol=1e-15, gtol=1e-14)
    
    # Stage 5: Very tight final polish
    if best_params is not None and time.time() < _DEADLINE - 2:
        try_optimize(best_params, 'L-BFGS-B', maxiter=10000, ftol=1e-16, gtol=1e-15)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
