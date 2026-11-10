"""Growth-index fitting for CPL dark-energy cosmologies."""
import time
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import differential_evolution

# Deadline guard: 85s hard limit (5s margin before 90s kill)
_T0 = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _T0

def _time_left():
    return _DEADLINE - _elapsed()

def _gamma_of(c, a, Om, w0, wa):
    """Compute gamma from the 6-coefficient formula."""
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    g = (c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q)
    return g

def _f_of(c, a, Om, w0, wa):
    """Growth rate f = Om^gamma."""
    g = _gamma_of(c, a, Om, w0, wa)
    # Clamp gamma to valid range to avoid issues
    g = np.clip(g, 0.1, 1.2)
    return Om ** g

def _Om_of_x(x, Om0, w0, wa):
    """Compute Om(a) where a = exp(x)."""
    a = np.exp(x)
    integ = (1.0 + w0) * x + wa * (x - a + 1.0)
    Om_m = Om0 * np.exp(-3.0 * x)
    Om_de = (1.0 - Om0) * np.exp(-3.0 * integ)
    return Om_m / (Om_m + Om_de)

def _numerical_f_and_D(a_arr, Om0, w0, wa):
    """Solve the GR growth ODE numerically to get f and D(a)/D(1)."""
    x_arr = np.log(a_arr)
    
    def Om_and_dOm(x):
        a = np.exp(x)
        w = w0 + wa * (1.0 - a)
        integ = (1.0 + w0) * x + wa * (x - a + 1.0)
        Om_m = Om0 * np.exp(-3.0 * x)
        Om_de = (1.0 - Om0) * np.exp(-3.0 * integ)
        total = Om_m + Om_de
        Om = Om_m / total
        Om_m_prime = -3.0 * Om_m
        Om_de_prime = -3.0 * (1.0 + w) * Om_de
        dOm = (Om_m_prime * Om_de - Om_m * Om_de_prime) / (total * total)
        return Om, dOm
    
    x_start = np.log(0.01)  # a=0.01
    x_end = 0.0             # a=1
    
    def ode_f(x, y):
        f = y[0]
        Om, dOm = Om_and_dOm(x)
        return [-1.5*(1.0+Om)*f*f - 0.5*dOm*f + 1.5*Om]
    
    Om_init, _ = Om_and_dOm(x_start)
    f_init = Om_init ** 0.55
    
    sol = solve_ivp(ode_f, [x_start, x_end], [f_init], 
                    t_eval=x_arr, method='RK45', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        # Fallback: use constant gamma
        f_num = np.array([_Om_of_x(np.log(a), Om0, w0, wa)**0.55 for a in a_arr])
        D_num = np.ones_like(a_arr)
        return f_num, D_num
    
    f_num = sol.y[0]
    
    # Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(x) dx)
    cum_int = cumulative_trapezoid(f_num, x_arr, initial=0.0)
    total_int = cum_int[-1]
    int_to_0 = total_int - cum_int
    D_num = np.exp(-int_to_0)
    
    return f_num, D_num

def _objective(c, test_cosmos, a_test, num_solutions):
    """Compute max relative error over all test cosmologies and a values."""
    max_err_f = 0.0
    max_err_D = 0.0
    
    for idx, (Om0, w0, wa) in enumerate(test_cosmos):
        f_num, D_num = num_solutions[idx]
        
        a_arr = a_test
        Om_arr = np.zeros_like(a_arr)
        for i, a in enumerate(a_arr):
            Om_arr[i] = _Om_of_x(np.log(a), Om0, w0, wa)
        
        f_model = _f_of(c, a_arr, Om_arr, w0, wa)
        
        valid = np.abs(f_num) > 1e-10
        if np.any(valid):
            err_f = np.abs(f_model[valid] - f_num[valid]) / np.abs(f_num[valid])
            max_err_f = max(max_err_f, np.max(err_f))
        
        valid_D = np.abs(D_num) > 1e-10
        if np.any(valid_D):
            x_arr = np.log(a_arr)
            cum_int_m = cumulative_trapezoid(f_model, x_arr, initial=0.0)
            total_int_m = cum_int_m[-1]
            int_to_0_m = total_int_m - cum_int_m
            D_model = np.exp(-int_to_0_m)
            
            err_D = np.abs(D_model[valid_D] - D_num[valid_D]) / np.abs(D_num[valid_D])
            max_err_D = max(max_err_D, np.max(err_D))
    
    return max(max_err_f, max_err_D)

def construct():
    """Return 6 coefficients for the growth-index formula."""
    # Representative CPL test cosmologies (disjoint from validation)
    test_cosmos = [
        (0.30, -1.0, 0.0),    # LCDM
        (0.30, -0.5, 0.5),    # CPL positive wa
        (0.30, -1.5, 0.5),    # CPL negative w0
        (0.30, -0.8, -0.3),   # CPL negative wa
        (0.25, -1.0, 0.3),    # Low Om, positive wa
        (0.35, -1.2, 0.2),    # High Om
        (0.30, -1.0, 0.0),    # LCDM again (weight)
        (0.30, -0.9, 0.4),    # Moderate CPL
    ]
    
    # Test a values
    a_test = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    
    # Precompute numerical solutions (expensive, do once)
    print(f"Precomputing numerical solutions... elapsed={_elapsed():.1f}s")
    num_solutions = []
    for (Om0, w0, wa) in test_cosmos:
        f_num, D_num = _numerical_f_and_D(a_test, Om0, w0, wa)
        num_solutions.append((f_num, D_num))
    print(f"Precomputation done. elapsed={_elapsed():.1f}s")
    
    # Objective function for optimizer
    def obj(c):
        if _time_left() < 1.0:
            return 100.0  # Return large value if running out of time
        return _objective(c, test_cosmos, a_test, num_solutions)
    
    # Bounds for each coefficient
    bounds = [(-10, 10)] * 6
    
    # Initial guess: constant gamma ~ 0.55
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print(f"Starting optimization. elapsed={_elapsed():.1f}s")
    
    # Use differential_evolution for global optimization
    # This is more robust than Nelder-Mead for multi-modal landscapes
    try:
        res = differential_evolution(obj, bounds, x0=x0, seed=42,
                                     maxiter=100, tol=1e-8, 
                                     mutation=(0.5, 1.5), recombination=0.7,
                                     popsize=15, polish=True,
                                     callback=lambda x, convergence: None)
        best_c = res.x
        best_obj = res.fun
        print(f"Differential evolution result: {best_obj:.6f} at c={best_c}")
    except Exception as e:
        print(f"Differential evolution failed: {e}")
        best_c = x0
        best_obj = obj(x0)
    
    # Clamp to [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    print(f"Final coefficients: {best_c}")
    print(f"Final objective: {best_obj:.6f}")
    print(f"Total elapsed: {_elapsed():.1f}s")
    
    return list(best_c)
