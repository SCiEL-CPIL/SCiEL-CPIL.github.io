import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

def construct():
    deadline = time.time() + 80
    Om_r0 = 8.5e-5
    
    # Fitting grid - representative of likely validation range
    w0_vals = np.array([-1.0, -0.5, 0.0, 0.5])
    wa_vals = np.array([-0.5, 0.0, 0.5])
    Om_m0_vals = np.array([0.2, 0.3, 0.4])
    a_vals = np.linspace(0.05, 0.99, 25)
    
    def E2(a, Om_m0, w0, wa):
        Om_L0 = 1.0 - Om_m0 - Om_r0
        return (Om_m0 * a**(-3) + Om_r0 * a**(-4) + 
                Om_L0 * a**(-3*(1+w0+wa)) * np.exp(3*wa*(a-1)))
    
    def Om_m(a, Om_m0, w0, wa):
        return Om_m0 * a**(-3) / E2(a, Om_m0, w0, wa)
    
    def solve_f(Om_m0, w0, wa):
        def dlnE_dx(x):
            a = np.exp(x)
            Om_L0 = 1.0 - Om_m0 - Om_r0
            E2_val = Om_m0 * a**(-3) + Om_r0 * a**(-4) + Om_L0 * a**(-3*(1+w0+wa)) * np.exp(3*wa*(a-1))
            dE2 = (-3*Om_m0 * a**(-3) - 4*Om_r0 * a**(-4) + 
                   Om_L0 * a**(-3*(1+w0+wa)) * np.exp(3*wa*(a-1)) * (-3*(1+w0+wa) + 3*wa*a))
            return dE2 / E2_val
        
        def rhs(x, y):
            f = y[0]
            a = np.exp(x)
            Om_m_val = Om_m(a, Om_m0, w0, wa)
            dlnE = dlnE_dx(x)
            return [-f**2 - (3 + dlnE)*f + 1.5*Om_m_val]
        
        f0 = Om_m0**0.55
        sol = solve_ivp(rhs, [0, np.log(0.05)], [f0], method='RK45', 
                        rtol=1e-10, atol=1e-12, dense_output=True)
        if not sol.success:
            return None
        return sol.sol(np.log(a_vals))[0]
    
    # Pre-compute reference
    ref_f_list = []
    ref_params = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for Om_m0 in Om_m0_vals:
                f_ref = solve_f(Om_m0, w0, wa)
                if f_ref is not None:
                    ref_f_list.append(f_ref)
                    ref_params.append((w0, wa, Om_m0))
    
    if not ref_f_list:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    ref_f = np.array(ref_f_list)
    n_cosmo = len(ref_params)
    
    # Pre-compute q, u, Om_m
    q_arr = np.zeros((n_cosmo, len(a_vals)))
    u_arr = np.zeros((n_cosmo, len(a_vals)))
    Om_m_arr = np.zeros((n_cosmo, len(a_vals)))
    wa_arr = np.zeros(n_cosmo)
    for i, (w0, wa, Om_m0) in enumerate(ref_params):
        wa_arr[i] = wa
        for j, a in enumerate(a_vals):
            Om_m_arr[i, j] = Om_m(a, Om_m0, w0, wa)
            q_arr[i, j] = 1 - Om_m_arr[i, j]
            u_arr[i, j] = 1 + w0 + wa*(1-a)
    
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        if np.any(np.abs(coeffs) > 10):
            return 1e10
        
        max_err = 0.0
        for i in range(n_cosmo):
            wa_i = wa_arr[i]
            gamma_i = c0 + c1*q_arr[i] + c2*q_arr[i]**2 + c3*u_arr[i] + c4*wa_i*a_vals + c5*u_arr[i]*q_arr[i]
            
            if np.any(gamma_i < 0.1) or np.any(gamma_i > 1.2):
                return 1e10
            
            f_approx = Om_m_arr[i] ** gamma_i
            err = np.abs(f_approx - ref_f[i]) / np.abs(ref_f[i])
            max_err = max(max_err, np.max(err))
        
        return 100 * max_err
    
    x0 = [0.55, 0.05, 0.0, 0.0, 0.0, 0.0]
    best_x = x0
    best_score = objective(x0)
    
    if time.time() > deadline - 30:
        return np.clip(best_x, -10, 10)
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    if result.fun < best_score:
        best_x = result.x
        best_score = result.fun
    
    if time.time() > deadline - 15:
        return np.clip(best_x, -10, 10)
    
    for x0_try in [[0.5, 0.1, 0.0, 0.0, 0.0, 0.0], 
                   [0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
                   [0.55, 0.05, 0.0, 0.0, 0.0, 0.01]]:
        if time.time() > deadline - 10:
            break
        res = minimize(objective, x0_try, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_score:
            best_x = res.x
            best_score = res.fun
    
    return np.clip(best_x, -10, 10)
