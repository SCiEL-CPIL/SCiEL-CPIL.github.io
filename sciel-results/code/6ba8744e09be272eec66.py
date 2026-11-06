"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from scipy.integrate import solve_ivp

def construct():
    """
    Fit six coefficients for the growth-index formula.
    Strategy:
    1. Generate a dense reference grid of (Om0, w0, wa) and a-values.
    2. Solve the GR growth ODE for each to get reference f and D.
    3. Compute the true gamma = log(f)/log(Om) for each point.
    4. Define the ansatz: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
       where q = 1-Om, u = 1+w0+wa*(1-a).
    5. Optimize c to minimize the maximum relative error of f and D.
       Use a two-stage approach:
       - Stage 1: Linear least squares fit to gamma to get a good start.
       - Stage 2: Nelder-Mead or Powell minimization on the min-max objective.
    """
    start_time = time.time()
    deadline = start_time + 78  # Safety margin before 90s kill

    # --- 1. Generate Reference Data ---
    # Grid for cosmology parameters
    Om0_vals = np.array([0.20, 0.25, 0.27, 0.30, 0.35])
    w0_vals = np.linspace(-1.5, -0.5, 15)
    wa_vals = np.linspace(-1.0, 1.0, 15)
    a_vals = np.linspace(0.1, 1.0, 100)  # Start at 0.1 to avoid extreme log issues, end at 1.0

    # Store reference data
    # We need to store the basis vectors X and the target gamma, plus f_ref, D_ref for error calc
    # To save memory and time, we'll process in chunks or just build the full array if it fits.
    # 5 * 15 * 15 * 100 = 112,500 points. This is small enough for NumPy.
    
    X_list = []
    gamma_true_list = []
    f_ref_list = []
    D_ref_list = []
    a_list = []
    # We also need the specific variables q, u, wa, a to reconstruct X if needed, 
    # but since the basis is fixed by the ansatz, we can just store X directly.
    # X columns: [1, q, q^2, u, wa*a, u*q]

    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                try:
                    f_ref, D_ref = _solve_growth_ode_fast(Om0, w0, wa, a_vals)
                    if f_ref is None:
                        continue
                    
                    # Compute Om_a for each a
                    Om_a = _compute_Om_a(Om0, w0, wa, a_vals)
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a_vals)
                    
                    # Filter valid points
                    # Om_a should be in (0, 1) for log to be valid and physical
                    valid = (Om_a > 0.01) & (Om_a < 0.99) & (f_ref > 0) & (D_ref > 0) & np.isfinite(f_ref) & np.isfinite(D_ref)
                    
                    if not np.any(valid):
                        continue
                    
                    # Compute true gamma
                    # f = Om^gamma => gamma = log(f) / log(Om)
                    # Note: log(Om) is negative since Om < 1. log(f) is negative since f < 1.
                    # So gamma is positive.
                    gamma_true = np.zeros_like(f_ref)
                    gamma_true[valid] = np.log(f_ref[valid]) / np.log(Om_a[valid])
                    
                    # Build X for valid points
                    X_valid = np.column_stack([
                        np.ones(np.sum(valid)),
                        q[valid],
                        q[valid]**2,
                        u[valid],
                        wa * a_vals[valid],
                        u[valid] * q[valid]
                    ])
                    
                    X_list.append(X_valid)
                    gamma_true_list.append(gamma_true[valid])
                    f_ref_list.append(f_ref[valid])
                    D_ref_list.append(D_ref[valid])
                    a_list.append(a_vals[valid])
                    
                except Exception:
                    continue

    if len(X_list) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Concatenate all data
    X_all = np.vstack(X_list)
    gamma_all = np.concatenate(gamma_true_list)
    f_ref_all = np.concatenate(f_ref_list)
    D_ref_all = np.concatenate(D_ref_list)
    a_all = np.concatenate(a_list)

    # --- 2. Stage 1: Initial Linear Fit ---
    # Fit X @ c = gamma
    try:
        c_init, residuals, rank, s = np.linalg.lstsq(X_all, gamma_all, rcond=None)
        c_init = np.clip(c_init, -10, 10)
    except:
        c_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    best_params = c_init.copy()
    
    # Precompute things needed for error calculation
    # The error calculation requires integrating f to get D.
    # This is expensive if done naively for every evaluation.
    # However, the grid a_all is not contiguous across the different cosmology models.
    # We must group by model to integrate correctly.
    
    # Let's reorganize data into a list of models for efficient error calculation
    # model_data = [ (X_model, gamma_model, f_ref_model, D_ref_model, a_model), ... ]
    # But we already concatenated. Let's keep the list structure for error calc.
    
    # Actually, the least_squares / minimize function needs to evaluate the error for the whole set.
    # We can just use the concatenated arrays, but we need to know the boundaries for integration.
    # It's safer to keep the list of models.
    
    model_data = list(zip(X_list, gamma_true_list, f_ref_list, D_ref_list, a_list))
    
    # --- 3. Stage 2: Min-Max Optimization ---
    
    def compute_max_error(params):
        """
        Compute the maximum relative error for f and D across all models.
        """
        c0, c1, c2, c3, c4, c5 = params
        max_err = 0.0
        
        for X_model, gamma_model, f_ref_model, D_ref_model, a_model in model_data:
            # Compute gamma
            gamma = X_model @ params
            
            # Check bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                # Return a large penalty
                return 10.0 + np.sum(np.maximum(0.1 - gamma, 0)) + np.sum(np.maximum(gamma - 1.2, 0))
            
            # Compute f_model
            # We need Om_a to compute f = Om^gamma.
            # We didn't store Om_a in model_data. Let's reconstruct it or store it.
            # It's better to store Om_a in the initial loop.
            # Let's modify the initial loop to store Om_a.
            # Wait, I can't modify the loop now without re-running.
            # I can reconstruct Om_a from X? No, X has q=1-Om, so Om = 1-q.
            # X[:, 1] is q. So Om = 1 - X[:, 1].
            
            Om_a_model = 1 - X_model[:, 1]
            
            f_model = Om_a_model ** gamma
            
            # Relative error for f
            rel_err_f = np.abs(f_model - f_ref_model) / np.maximum(f_ref_model, 1e-10)
            max_err = max(max_err, np.max(rel_err_f))
            
            # Compute D_model from f_model
            # D(a)/D(1) = exp(-integral from ln(a) to 0 f(exp(x)) dx)
            # a_model is sorted ascending.
            x_model = np.log(a_model)
            dx = np.diff(x_model)
            f_avg = 0.5 * (f_model[:-1] + f_model[1:])
            integrands = f_avg * dx
            rev_int = integrands[::-1]
            cumsum_rev = np.cumsum(rev_int)
            
            N = len(a_model)
            D_model = np.ones(N)
            if N > 1:
                idx_map = np.arange(N-2, -1, -1)
                integrals = cumsum_rev[idx_map]
                D_model[:N-1] = np.exp(-integrals)
            
            rel_err_D = np.abs(D_model - D_ref_model) / np.maximum(D_ref_model, 1e-10)
            max_err = max(max_err, np.max(rel_err_D))
            
        return max_err

    # Initial score
    best_score = compute_max_error(best_params)
    
    # Optimization
    # Use Nelder-Mead or Powell
    # Nelder-Mead is derivative-free and robust for low dimensions.
    # Powell is also good.
    # Let's try Nelder-Mead first.
    
    try:
        # Define objective function
        def obj_func(params):
            return compute_max_error(params)
        
        # Run Nelder-Mead
        res = minimize(
            obj_func,
            best_params,
            method='Nelder-Mead',
            bounds=[(-10, 10) for _ in range(6)],
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Check time
    if time.time() > deadline - 10:
        best_params = np.clip(best_params, -10, 10)
        return list(best_params)
    
    # --- 4. Stage 3: Local Refinement / Second Pass ---
    # Try a second optimization pass with a tighter tolerance or different method
    try:
        res2 = minimize(
            obj_func,
            best_params,
            method='Powell',
            bounds=[(-10, 10) for _ in range(6)],
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res2.fun < best_score:
            best_score = res2.fun
            best_params = res2.x
            
    except Exception:
        pass
    
    best_params = np.clip(best_params, -10, 10)
    return list(best_params)

def _compute_Om_a(Om0, w0, wa, a):
    """Compute matter density at scale factor a."""
    a = np.asarray(a)
    Om_m = Om0 * a**3
    w_eff = w0 + wa * (1 - a)
    Om_de = (1 - Om0) * a**(-3 * w_eff)
    return Om_m / (Om_m + Om_de)

def _solve_growth_ode_fast(Om0, w0, wa, a_values):
    """Solve the growth ODE efficiently."""
    a_min = min(a_values)
    a_max = max(a_values)
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode_system(x, y):
        D, Dp = y
        a = np.exp(x)
        Om_m = _compute_Om_a(Om0, w0, wa, a)
        coeff = (3/2) * (1 + Om_m / (1 - Om_m))
        return [Dp, -coeff * Dp + (3/2) * Om_m * D]

    x_eval = np.linspace(x_max, x_min, len(a_values))
    sol = solve_ivp(ode_system, [x_max, x_min], [1.0, 0.0], t_eval=x_eval,
                    rtol=1e-6, atol=1e-8, method='RK45')

    if not sol.success:
        return None, None

    D_vals = sol.y[0]
    Dp_vals = sol.y[1]

    # Check for invalid values
    if np.any(D_vals <= 0) or np.any(~np.isfinite(D_vals)):
        return None, None

    f_vals = Dp_vals / D_vals
    D_vals = D_vals / D_vals[0]  # Normalize D(1) = 1

    if np.any(~np.isfinite(f_vals)) or np.any(~np.isfinite(D_vals)):
        return None, None

    return f_vals, D_vals

# EVOLVE-BLOCK-END
