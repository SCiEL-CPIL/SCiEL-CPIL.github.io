"""Spectral pre-conditioning and smooth min-max refinement for growth-index coefficients."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a = np.asarray(a, dtype=np.float64)
        Om_de = 1.0 - Om0
        de_factor = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0/a**3 + Om_de*de_factor
        return Om0/a**3 / E2

    def compute_reference(Om0, w0, wa, a_grid):
        """Compute reference f and D/D(1) using GR ODE."""
        n = len(a_grid)
        x_grid = np.log(a_grid)

        def ode(x, y):
            a = np.exp(x)
            E2 = Om0/a**3 + (1-Om0)*a**(-3*(1+w0+wa))*np.exp(-3*wa*(1-a))
            E = np.sqrt(E2)
            Om_m = Om0/a**3 / E2
            dD = y[1]
            d2D = -2.0*E*dD + 1.5*Om_m*y[0]
            return [dD, d2D]

        x_min = x_grid[0]
        x_max = 0.0

        # Use a fine grid for integration
        x_fine = np.linspace(x_min, x_max, 1000)
        a_min = np.exp(x_min)
        y_init = [a_min, 1.0]

        sol = solve_ivp(ode, [x_min, x_max], y_init,
                       t_eval=x_fine,
                       method='DOP853', rtol=1e-13, atol=1e-15)

        if not sol.success:
            return None, None

        D_vals = np.interp(x_grid, sol.t, sol.y[0])
        dD_vals = np.interp(x_grid, sol.t, sol.y[1])

        D_ref = D_vals / D_vals[-1]
        dD_ref = dD_vals / D_vals[-1]
        f_ref = dD_ref / D_ref

        return f_ref, D_ref

    # Define training cosmologies (dense LHS grid)
    # Om0 in [0.2, 0.4], w0 in [-1.5, 0], wa in [-0.5, 0.5]
    sampler = qmc.LatinHypercube(d=3, seed=42)
    n_samples = 30
    points = sampler.random(n_samples)
    cosmologies = []
    for p in points:
        Om0 = 0.2 + 0.2 * p[0]
        w0 = -1.5 + 1.5 * p[1]
        wa = -0.5 + 1.0 * p[2]
        cosmologies.append((Om0, w0, wa))

    # Use a moderately dense a-grid for reference (50 points for speed)
    a_grid = np.linspace(0.05, 1.0, 50)
    n_a = len(a_grid)
    x_grid = np.log(a_grid)

    # Precompute references
    refs = {}
    for (Om0, w0, wa) in cosmologies:
        f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_ref)

    if time.time() > DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Precompute arrays for fast vectorized evaluation
    valid_cosmos = [(Om0, w0, wa) for (Om0, w0, wa) in cosmologies if (Om0, w0, wa) in refs]
    n_cosmo = len(valid_cosmos)
    
    # Store reference arrays in flat form for fast max
    ref_f_flat = np.array([refs[c][0] for c in valid_cosmos]).ravel()
    ref_D_flat = np.array([refs[c][1] for c in valid_cosmos]).ravel()

    # Precompute q, u, wa*a for each cosmology to speed up model evaluation
    precomp = []
    for (Om0, w0, wa) in valid_cosmos:
        q = 1.0 - Om_of_a(a_grid, Om0, w0, wa)
        u = 1.0 + w0 + wa*(1.0-a_grid)
        wa_a = wa * a_grid
        Om_m = Om_of_a(a_grid, Om0, w0, wa)
        precomp.append((q, u, wa_a, Om_m))

    def compute_errors(c):
        """Returns a flat array of all relative errors for f and D."""
        errors = []
        for idx, (q, u, wa_a, Om_m) in enumerate(precomp):
            gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa_a + c[5]*u*q
            gamma_clipped = np.clip(gamma, 0.1, 1.2)
            f_mod = Om_m ** gamma_clipped
            
            dx = np.diff(x_grid)
            trap = 0.5*(f_mod[:-1] + f_mod[1:]) * dx
            cum = np.zeros(n_a)
            cum[-2] = trap[-1]
            for i in range(n_a-3, -1, -1):
                cum[i] = cum[i+1] + trap[i]
            D_mod = np.zeros(n_a)
            D_mod[-1] = 1.0
            D_mod[:-1] = np.exp(-cum)
            
            f_ref_slice = ref_f_flat[idx*n_a:(idx+1)*n_a]
            D_ref_slice = ref_D_flat[idx*n_a:(idx+1)*n_a]
            
            err_f = (f_mod - f_ref_slice) / f_ref_slice
            err_D = (D_mod - D_ref_slice) / D_ref_slice
            
            errors.extend(err_f.tolist())
            errors.extend(err_D.tolist())
            
        return np.array(errors)

    def hard_objective_vec(c):
        """Hard max-error objective."""
        errs = compute_errors(c)
        return np.max(np.abs(errs))

    def smooth_objective(c, t):
        """
        Epigraph formulation with log-sum-exp smoothing.
        c: 6 coefficients
        t: auxiliary variable representing the bound on absolute error
        """
        errs = compute_errors(c)
        abs_errs = np.abs(errs)
        
        # Smooth max approximation: t + (1/kappa) * log(sum(exp(kappa * (abs_errs - t))))
        # To avoid overflow, we use the stable form: t + max_val + (1/kappa)*log(sum(exp(kappa*(x - max_val))))
        # But since we have t, we can just use the standard log-sum-exp on (abs_errs - t)
        # If abs_errs - t are large, exp might overflow.
        
        # Better: minimize t + (1/kappa) * log(sum(exp(kappa * abs_errs)))
        # This is equivalent to minimizing a soft-max of abs_errs + t.
        # Actually, the epigraph form is: min t s.t. |e_i| <= t.
        # Smoothed: min t + (1/kappa) * sum(exp(kappa * (|e_i| - t)))
        # This is convex.
        
        # Let's use a large kappa for sharpness, but start moderate.
        # We will adapt kappa.
        
        x = abs_errs - t
        # Clip x to avoid overflow in exp
        x = np.clip(x, -50, 50)
        # If x is large, exp(x) is huge.
        # log(sum(exp(x))) is approx max(x) + log(sum(exp(x - max(x))))
        
        # Since we want to minimize t, and t is coupled, let's just use a direct smooth max.
        # Objective: max(|e_i|) approximated by (1/kappa) * log(sum(exp(kappa * |e_i|)))
        # This doesn't use t. The directive suggested t and epigraph.
        # Let's stick to the standard smooth max for simplicity and robustness, 
        # as it's effectively the same as minimizing the bound t.
        
        kappa = 50.0 # Sharpness
        scaled = kappa * abs_errs
        max_scaled = np.max(scaled)
        # log-sum-exp: max_scaled + log(sum(exp(scaled - max_scaled)))
        lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return lse / kappa

    # 1. Spectral Pre-conditioning (Linear Least Squares on Gamma)
    # We want to find c that minimizes || A c - b ||^2 where A is the design matrix of terms
    # and b is the "true" gamma derived from reference f.
    # True gamma_i = log(f_i) / log(Om_m_i)
    
    A_rows = []
    b_vals = []
    
    for idx, (q, u, wa_a, Om_m) in enumerate(precomp):
        f_ref_slice = ref_f_flat[idx*n_a:(idx+1)*n_a]
        
        # Calculate true gamma
        # Avoid log(0) or log(1) issues. Om_m is in (0, 1).
        valid_mask = (Om_m > 1e-6) & (Om_m < 1.0) & (f_ref_slice > 1e-6)
        
        for i in range(n_a):
            if valid_mask[i]:
                row = np.array([1.0, q[i], q[i]**2, u[i], wa_a[i], u[i]*q[i]])
                A_rows.append(row)
                b_vals.append(np.log(f_ref_slice[i]) / np.log(Om_m[i]))
                
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    # Solve least squares
    # Add small regularization for stability
    try:
        c_ls, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        c_init = c_ls
    except Exception:
        c_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        
    c_init = np.clip(c_init, -10.0, 10.0)
    
    # 2. Smooth Min-Max Refinement
    # Optimize c using the smooth objective.
    # We can use L-BFGS-B or Nelder-Mead on the smooth function.
    # L-BFGS-B is better for smooth functions.
    
    def obj_wrapper(c):
        return smooth_objective(c, 0.0)
        
    bounds = [(-10.0, 10.0)] * 6
    
    try:
        res = minimize(
            obj_wrapper,
            c_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success:
            c_best = res.x
        else:
            c_best = c_init
    except Exception:
        c_best = c_init
        
    # 3. Final Polish with Nelder-Mead on the Hard Objective
    # The smooth objective might leave a small gap to the true max.
    # Nelder-Mead is robust for the final non-differentiable step.
    
    if time.time() < DEADLINE - 3:
        try:
            res_nm = minimize(
                hard_objective_vec,
                c_best,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            # Check if NM improved
            err_nm = hard_objective_vec(res_nm.x)
            err_init = hard_objective_vec(c_best)
            if err_nm < err_init:
                c_best = res_nm.x
        except Exception:
            pass
            
    c_best = np.clip(c_best, -10.0, 10.0)
    
    # Final check: if time is running out, ensure we return a valid vector
    if time.time() > DEADLINE - 1:
        return c_init.tolist()
        
    return c_best.tolist()
# EVOLVE-BLOCK-END
