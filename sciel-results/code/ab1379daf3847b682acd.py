"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s margin
    
    # Reference model parameters (Lambda CDM)
    Om = 0.3
    w0 = -1.0
    wa = 0.0
    
    # Grid of scale factors for fitting
    # We need to cover the range where the approximation is evaluated.
    # Usually a in [0.1, 1.0] or similar. Let's use a logarithmic grid.
    a_grid = np.logspace(np.log10(0.1), 0.0, 100)
    
    def compute_omega_m(a):
        # Flat universe: Omega_m(a) = Om * a^-3 / (Om * a^-3 + (1-Om) * a^(-3(1+w0+wa*(1-a))) )
        # For CPL: Omega_de(a) = (1-Om) * a^(-3(1+w0+wa*(1-a)))
        # Note: The exponent is -3(1+w(a)) = -3(1+w0+wa(1-a))
        # Let's compute the denominator terms
        term_m = Om * a**(-3)
        w_a = w0 + wa * (1 - a)
        term_de = (1 - Om) * a**(-3 * (1 + w_a))
        return term_m / (term_m + term_de)
    
    def compute_q(a):
        return 1 - compute_omega_m(a)
        
    def compute_u(a):
        return 1 + w0 + wa * (1 - a)
        
    def true_growth_f(a):
        # Numerical integration of the growth ODE to get f(a)
        # f(a) = d ln D / d ln a
        # ODE: D'' + (3 + H'/H) D' - (1/2) Omega_m D = 0 in terms of ln a
        # Let y = [D, D'] where D' = dD/dln(a)
        # H'/H in terms of a:
        # H^2 = H0^2 [Om a^-3 + (1-Om) a^-3(1+w0+wa(1-a))]
        # Let's just integrate numerically for the reference.
        # We will pre-compute the reference values on the grid using a simple ODE solver.
        pass
        
    # Pre-compute reference values for the specific model
    # We need f_ref and D_ref on a_grid
    # Let's use a simple explicit integration for the reference
    # f = d ln D / d ln a
    # f' = -f^2 - 1.5 f (1 + q) + 0.5 Omega_m
    # q = -0.5 Omega_m - 1.5 Omega_de (1+w)
    # For flat universe, q = -0.5 Omega_m - 1.5 (1-Omega_m)(1+w)
    
    def ref_omega_m(a):
        term_m = Om * a**(-3)
        w_a = w0 + wa * (1 - a)
        term_de = (1 - Om) * a**(-3 * (1 + w_a))
        return term_m / (term_m + term_de)
        
    def ref_q(a):
        Om_a = ref_omega_m(a)
        w_a = w0 + wa * (1 - a)
        return -0.5 * Om_a - 1.5 * (1 - Om_a) * (1 + w_a)
        
    # Integrate f from a=1 (where f(1) ~ 1.0, D(1)=1) backwards to a=0.1
    # Or integrate D from a=0.1 to 1.
    # Let's integrate D.
    # D'' + (3 + H'/H) D' - 0.5 Omega_m D = 0
    # H'/H = -1.5 (1 + q)
    # So coeff of D' is 3 - 1.5(1+q) = 1.5 - 1.5 q
    
    n_steps = 1000
    a_int = np.linspace(0.1, 1.0, n_steps)
    D_vals = np.zeros(n_steps)
    dD_vals = np.zeros(n_steps) # dD/dln(a)
    
    # Initial conditions at a=1
    D_vals[-1] = 1.0
    # f(1) for LCDM is approx 0.9-1.0. Let's use 1.0 as a good start for normalization.
    dD_vals[-1] = 1.0 # f(1) = 1.0
    
    # Backward integration
    for i in range(n_steps - 1, 0, -1):
        a = a_int[i]
        Da = D_vals[i]
        dDa = dD_vals[i]
        
        Om_a = ref_omega_m(a)
        q_a = ref_q(a)
        
        # D'' = - (1.5 - 1.5 q_a) dDa + 0.5 Om_a Da
        d2Da = -(1.5 - 1.5 * q_a) * dDa + 0.5 * Om_a * Da
        
        # Euler step (backward)
        da = a_int[i] - a_int[i-1]
        dlna = np.log(a) - np.log(a_int[i-1])
        
        # Update D and dD
        # dD/dlna is the derivative.
        # dD_new = dD_old + d2D * dlna
        # D_new = D_old + dD_old * dlna
        dD_vals[i-1] = dDa + d2Da * dlna
        D_vals[i-1] = Da + dDa * dlna
        
    # Compute reference f and D on a_grid
    # Interpolate D and dD to a_grid
    ln_a_int = np.log(a_int)
    ln_a_grid = np.log(a_grid)
    
    D_ref_grid = np.interp(ln_a_grid, ln_a_int, D_vals)
    dD_ref_grid = np.interp(ln_a_grid, ln_a_int, dD_vals)
    f_ref_grid = dD_ref_grid / D_ref_grid
    
    # Normalize D such that D(1)=1 is already satisfied by IC.
    # The evaluator computes D(a)/D(1) = exp(-integral... no, exp(integral f dx))
    # Actually, D(a)/D(1) = exp( \int_1^a f(a') d ln a' ) = exp( - \int_a^1 f(a') d ln a' )
    # The task says: D(a)/D(1)=exp(-integral_ln(a)^0 f(exp(x)) dx)
    # Wait, the integral is from ln(a) to 0?
    # \int_{\ln a}^0 f(e^x) dx. If a<1, ln a < 0.
    # Let's check the sign.
    # d ln D = f d ln a.
    # ln D(a) - ln D(1) = \int_1^a f d ln a' = - \int_a^1 f d ln a'
    # Let x = ln a'. dx = d ln a'.
    # \int_{\ln a}^0 f(e^x) dx = \ln D(1) - \ln D(a) = - (\ln D(a) - \ln D(1))
    # So D(a)/D(1) = exp( - \int_{\ln a}^0 f(e^x) dx ). This matches the task description.
    
    D_ref_norm = D_ref_grid / D_ref_grid[-1] # D(1) is at a=1 (last index)
    
    # Model functions
    def model_gamma(c, a):
        Om_a = compute_omega_m(a)
        q = 1 - Om_a
        u = compute_u(a)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        
    def model_f(c, a):
        Om_a = compute_omega_m(a)
        g = model_gamma(c, a)
        # Clip gamma to valid range to avoid errors, though we enforce constraints
        g = np.clip(g, 0.1, 1.2)
        return Om_a ** g
        
    def model_D(c, a):
        # Compute D(a)/D(1) by integrating f
        # We can compute this numerically on the grid for the objective function
        # Since we are optimizing, we can just compute the error on the grid points
        # using the cumulative sum of f * dlna
        f_vals = model_f(c, a)
        # We need D(a)/D(1).
        # D(a)/D(1) = exp( - \int_{\ln a}^0 f dx )
        # Let's compute the integral I(a) = \int_{\ln a}^0 f(e^x) dx
        # This is equivalent to \int_a^1 f(a') d ln a'
        # We can compute this by summing f * dlna from a to 1.
        # Since a_grid is sorted ascending, we can compute cumulative sum from the end.
        ln_a = np.log(a)
        dlna = np.diff(ln_a)
        # f_vals corresponds to a_grid
        # Integral from a[i] to 1 is sum of f[j] * (ln a[j+1] - ln a[j]) for j from i to end-1
        # Approximate with trapezoidal or just rectangle. Rectangle is fine for optimization.
        # Let's use simple cumulative sum from the end.
        # f_vals * dlna
        # We need to be careful with the last point.
        
        # Alternative: Just compute the relative error of f and the relative error of D.
        # For D, we can compute the model D by integrating the model f.
        
        # To make it fast, let's just return the errors for f and D.
        # D_model calculation:
        # We assume D(1)=1.
        # D(a) = exp( - \int_a^1 f_model d ln a' )
        # Let's compute the integral array.
        # dlna array has length N-1.
        # integral_from_a[i] = sum_{j=i}^{N-2} f[j] * dlna[j] ? No, this is from a[i] to a[i+1]...
        # Let's do it properly.
        
        # Create a vector of integrals from a[i] to 1.
        # I[i] = \int_{a[i]}^1 f d ln a
        # I[N-1] = 0
        # I[i] = I[i+1] + f[i+1] * (ln a[i+1] - ln a[i]) (using right Riemann sum or average)
        # Let's use trapezoidal: 0.5 * (f[i] + f[i+1]) * dlna[i]
        
        I = np.zeros_like(a)
        I[-1] = 0.0
        for i in range(len(a)-2, -1, -1):
            # Average f
            f_avg = 0.5 * (f_vals[i] + f_vals[i+1])
            I[i] = I[i+1] + f_avg * dlna[i]
            
        D_model_norm = np.exp(-I)
        
        return f_vals, D_model_norm, g
        
    def objective(c):
        if time.time() > deadline:
            return 1e10
            
        f_vals, D_model_norm, g = model_D(c, a_grid)
        
        # Check constraints
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1e10 + np.max(np.abs(g - np.clip(g, 0.1, 1.2)))
            
        # Relative errors
        # f error
        rel_err_f = np.abs(f_vals - f_ref_grid) / np.abs(f_ref_grid)
        
        # D error
        rel_err_D = np.abs(D_model_norm - D_ref_norm) / np.abs(D_ref_norm)
        
        # Max relative error
        max_err = np.max(np.concatenate([rel_err_f, rel_err_D]))
        
        return max_err

    # Initial guess
    c0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize
    # Use Nelder-Mead for robustness without gradients
    res = minimize(objective, c0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if time.time() > deadline:
         return c0
         
    best_c = res.x
    best_val = res.fun
    
    # If optimization failed or didn't improve, try a few random restarts?
    # Nelder-Mead is usually good enough for this smooth problem.
    # Let's verify the solution.
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10, 10)
    
    # Final check on gamma range
    g_final = model_gamma(best_c, a_grid)
    if np.any(g_final < 0.1) or np.any(g_final > 1.2):
        # Fallback to seed if constraints violated
        return c0
        
    return list(best_c)

# EVOLVE-BLOCK-END
