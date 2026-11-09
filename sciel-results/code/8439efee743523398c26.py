# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution
from itertools import product

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1

# Pre-compute quadrature and basis functions
def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        # Compute (10-am)! / (10+am)!
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        
        # Condon-Shortley phase is included in lpmv
        P = lpmv(am, _K, x)
        
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
            
    return Y, W

_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)

def _objective(c):
    """R^4 = <f^4> / <f^2>^2"""
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)

def _amplitudes_to_c(amps):
    """Convert 11-dim amplitude vector to 21-dim coefficient vector.
    amps[0] = c_0, amps[1..10] = |z_m| for m=1..10
    We set phases to zero (c_{-m} = 0, c_m = |z_m|) which is WLOG due to phase invariance."""
    c = np.zeros(_NC, dtype=np.float64)
    c[_K] = amps[0]  # c_0
    for m in range(1, _K + 1):
        c[_K + m] = amps[m]  # c_m = |z_m|, c_{-m} = 0
    return c

def _c_to_amplitudes(c):
    """Extract 11-dim amplitude vector from 21-dim coefficient vector."""
    amps = np.zeros(_K + 1, dtype=np.float64)
    amps[0] = abs(c[_K])  # |c_0|
    for m in range(1, _K + 1):
        amps[m] = np.sqrt(c[_K + m]**2 + c[_K - m]**2)  # |z_m|
    return amps

def _gradient(c):
    """Analytic gradient of R^4"""
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    
    # d/dc ( <f^4> ) = 4 Y^T (W f^3)
    # d/dc ( <f^2>^2 ) = 4 L2 Y^T (W f)
    # dJ/dc = 4/B^2 Y^T (W f^3) - 4A/B^3 Y^T (W f)
    
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad

def _project_sphere(c):
    """Project c onto the unit sphere."""
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    return c / nrm

def _polish(c0, deadline):
    """Refine c0 using SLSQP with sphere constraint, then projected gradient descent"""
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    
    # Normalize initial guess
    c = c / nrm
    
    def fun(x):
        return _objective(x)
        
    def jac(x):
        return _gradient(x)
        
    best_c = c.copy()
    best_val = fun(c)
    
    # 1. SLSQP with sphere constraint
    try:
        constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
        res = minimize(fun, c, jac=jac, method='SLSQP',
                       constraints=[constraint],
                       options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c = _project_sphere(best_c)
    except Exception:
        pass
        
    # 2. Projected gradient descent with backtracking line search
    # This helps escape shallow local minima that SLSQP might get stuck in
    c_cur = best_c.copy()
    val_cur = best_val
    
    for step_iter in range(100):
        if time.time() >= deadline - 5.0:
            break
            
        g = _gradient(c_cur)
        # Project gradient onto tangent space of sphere
        g_tan = g - np.dot(g, c_cur) * c_cur
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-12:
            break
        
        # Backtracking line search
        alpha = 1.0
        improved = False
        for _ls in range(30):
            c_trial = c_cur - alpha * g_tan
            c_trial = _project_sphere(c_trial)
            if c_trial is None:
                break
            val_trial = _objective(c_trial)
            if val_trial < val_cur:
                c_cur = c_trial
                val_cur = val_trial
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
        
        if val_cur < best_val:
            best_val = val_cur
            best_c = c_cur.copy()
    
    # 3. Final SLSQP polish with tighter tolerance
    if time.time() < deadline - 3.0:
        try:
            constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
            res = minimize(fun, best_c, jac=jac, method='SLSQP',
                           constraints=[constraint],
                           options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_c = _project_sphere(res.x.copy())
        except Exception:
            pass
    
    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    # Return normalized c
    return best_c / nrm, float(best_val)

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 625.0  # 5s safety margin
    
    best_c = None
    best_val = np.inf
    
    # Strategy: Grid search over the 11-dim amplitude space, then SLSQP polish
    # The objective is invariant under phase rotations of each m-pair, so we can
    # restrict to c_{-m} = 0, c_m >= 0 for m > 0, and c_0 real.
    # This gives us an 11-dim search space (c_0, |z_1|, ..., |z_10|).
    
    # Phase 1: Coarse grid search over amplitudes
    # We'll use a structured grid with varying resolutions per dimension
    # based on importance (higher m modes likely have smaller optimal amplitudes)
    
    # For the grid, we'll search over relative amplitudes on the unit sphere in 11D
    # Use a multi-resolution approach:
    # - First pass: coarse grid with ~5 points per dim for most dims
    # - Second pass: refine around best region
    
    # Define grid points for each amplitude dimension
    # amps[0] = c_0, amps[1..10] = |z_m|
    # We'll normalize so that sum(amps^2) = 1
    
    # Coarse grid: use 4-6 points per dimension depending on m
    # Higher m modes tend to have smaller amplitudes in optimal solutions
    grid_res = []
    for m in range(_K + 1):
        if m == 0:
            grid_res.append(6)  # c_0: 6 points
        elif m <= 3:
            grid_res.append(5)  # low m: 5 points
        elif m <= 6:
            grid_res.append(4)  # mid m: 4 points
        else:
            grid_res.append(3)  # high m: 3 points
    
    # Generate grid points on the unit sphere in 11D
    # We'll use a recursive construction or just enumerate all combinations
    # and normalize
    
    # For efficiency, we'll use a stratified approach
    # Generate all combinations of grid values, normalize, and evaluate
    
    # Grid values for each dimension (before normalization)
    grid_vals = []
    for m in range(_K + 1):
        npts = grid_res[m]
        if m == 0:
            # c_0 can be negative, so use symmetric range
            vals = np.linspace(-1.0, 1.0, npts)
        else:
            # |z_m| >= 0
            vals = np.linspace(0.0, 1.0, npts)
        grid_vals.append(vals)
    
    # Enumerate all combinations and evaluate
    # Total points: product of grid_res
    total_grid_points = 1
    for r in grid_res:
        total_grid_points *= r
    
    # This could be up to 6*5^3*4^3*3^4 = 6*125*64*81 = 3,888,000 which is too many
    # Let's reduce the grid resolution
    grid_res = []
    for m in range(_K + 1):
        if m == 0:
            grid_res.append(5)
        elif m <= 2:
            grid_res.append(4)
        elif m <= 5:
            grid_res.append(3)
        else:
            grid_res.append(2)
    
    total_grid_points = 1
    for r in grid_res:
        total_grid_points *= r
    
    # 5 * 4^2 * 3^3 * 2^5 = 5 * 16 * 27 * 32 = 69,120 - manageable
    
    grid_vals = []
    for m in range(_K + 1):
        npts = grid_res[m]
        if m == 0:
            vals = np.linspace(-1.0, 1.0, npts)
        else:
            vals = np.linspace(0.0, 1.0, npts)
        grid_vals.append(vals)
    
    # Enumerate grid and evaluate
    best_grid_val = np.inf
    best_grid_amps = None
    
    # Use itertools.product for enumeration
    for combo in product(*grid_vals):
        if time.time() >= deadline - 300.0:
            break
        
        amps = np.array(combo, dtype=np.float64)
        nrm = np.linalg.norm(amps)
        if nrm < 1e-14:
            continue
        amps /= nrm
        
        c = _amplitudes_to_c(amps)
        val = _objective(c)
        
        if val < best_grid_val:
            best_grid_val = val
            best_grid_amps = amps.copy()
    
    # Phase 2: Refine with finer grid around the best point
    if best_grid_amps is not None and time.time() < deadline - 200.0:
        # Local refinement: perturb each dimension slightly
        refine_res = 5
        local_best_val = best_grid_val
        local_best_amps = best_grid_amps.copy()
        
        for dim in range(_K + 1):
            # Create local grid around best_grid_amps[dim]
            delta = 0.2  # relative step
            local_vals = best_grid_amps[dim] * (1.0 + np.linspace(-delta, delta, refine_res))
            
            for new_val in local_vals:
                if time.time() >= deadline - 200.0:
                    break
                
                test_amps = best_grid_amps.copy()
                test_amps[dim] = new_val
                nrm = np.linalg.norm(test_amps)
                if nrm < 1e-14:
                    continue
                test_amps /= nrm
                
                c = _amplitudes_to_c(test_amps)
                val = _objective(c)
                
                if val < local_best_val:
                    local_best_val = val
                    local_best_amps = test_amps.copy()
        
        best_grid_val = local_best_val
        best_grid_amps = local_best_amps
    
    # Phase 3: Convert best grid point to 21D and polish
    if best_grid_amps is not None:
        best_c = _amplitudes_to_c(best_grid_amps)
        best_val = best_grid_val
        
        # Polish with SLSQP
        polished = _polish(best_c, deadline)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c
    
    # Phase 4: Additional random restarts in amplitude space
    if time.time() < deadline - 100.0:
        rng = np.random.default_rng(456)
        
        for _ in range(50):
            if time.time() >= deadline - 100.0:
                break
            
            # Random point on unit sphere in 11D amplitude space
            amps = rng.standard_normal(_K + 1)
            nrm = np.linalg.norm(amps)
            if nrm < 1e-14:
                continue
            amps /= nrm
            
            c = _amplitudes_to_c(amps)
            val = _objective(c)
            
            if val < best_val:
                best_val = val
                best_c = c.copy()
                
                # Polish immediately if promising
                if time.time() < deadline - 50.0:
                    polished = _polish(best_c, deadline)
                    if polished:
                        c_pol, val_pol = polished
                        if val_pol < best_val:
                            best_val = val_pol
                            best_c = c_pol
    
    # Phase 5: Also try some structured starts in full 21D space
    if time.time() < deadline - 50.0:
        rng2 = np.random.default_rng(789)
        
        # Pure modes
        for m in range(-_K, _K + 1):
            if time.time() >= deadline - 50.0:
                break
            c0 = np.zeros(_NC)
            c0[m + _K] = 1.0
            val = _objective(c0)
            if val < best_val:
                best_val = val
                best_c = c0.copy()
        
        # Random 21D starts
        for _ in range(20):
            if time.time() >= deadline - 50.0:
                break
            c0 = rng2.standard_normal(_NC)
            c0 /= np.linalg.norm(c0)
            val = _objective(c0)
            if val < best_val:
                best_val = val
                best_c = c0.copy()
                
                if time.time() < deadline - 30.0:
                    polished = _polish(best_c, deadline)
                    if polished:
                        c_pol, val_pol = polished
                        if val_pol < best_val:
                            best_val = val_pol
                            best_c = c_pol
    
    # Phase 6: Final high-precision polish
    if best_c is not None and time.time() < deadline - 10.0:
        for _attempt in range(3):
            if time.time() >= deadline - 10.0:
                break
            final_polished = _polish(best_c, deadline)
            if final_polished:
                c, val = final_polished
                if val < best_val:
                    best_val = val
                    best_c = c
                else:
                    break
    
    # Final safety
    if best_c is None:
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
        
    best_c = np.asarray(best_c, dtype=np.float64)
    
    # Validation checks
    if not np.all(np.isfinite(best_c)):
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
        
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
        
    return best_c

# EVOLVE-BLOCK-END
