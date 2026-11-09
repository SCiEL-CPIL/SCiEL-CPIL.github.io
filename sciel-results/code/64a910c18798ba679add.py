import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s kill

    # 1. Define fitting grid
    om0_vals = np.linspace(0.2, 0.35, 4)
    w0_vals = np.linspace(-1.2, 0.0, 4)
    wa_vals = np.linspace(-0.5, 0.5, 4)
    a_vals = np.linspace(0.1, 1.0, 15)
    
    refs = []
    params_list = []
    
    def compute_reference(om0, w0, wa, a_eval):
        def h2(a):
            om_m = om0 * a**(-3)
            om_l = (1.0 - om0) * a**(-3.0 * (1.0 + w0)) * np.exp(3.0 * wa * (1.0 - a))
            return om_m + om_l

        def om_m_of_a(a):
            om_m = om0 * a**(-3)
            return om_m / h2(a)

        def growth_rhs(x, y):
            a = np.exp(x)
            om_m_a = om_m_of_a(a)
            
            om_m_val = om0 * a**(-3)
            om_l_val = (1.0 - om0) * a**(-3.0 * (1.0 + w0)) * np.exp(3.0 * wa * (1.0 - a))
            dh2_da = -3.0 * om_m_val / a + om_l_val * (-3.0 * (1.0 + w0) / a - 3.0 * wa)
            a_hp_h = 0.5 * a * dh2_da / h2(a)
            
            f = y[1]
            df_dx = 1.5 * om_m_a - f**2 - 0.5 * f - a_hp_h * f
            return [y[1] * y[0], df_dx]

        x_start = 0.0
        x_end = np.log(0.1)
        om_m_1 = om_m_of_a(1.0)
        f_init = om_m_1**0.55
        D_init = 1.0
        
        sol = solve_ivp(growth_rhs, [x_start, x_end], [D_init, f_init], 
                        t_eval=np.log(a_eval), method='RK45', rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            return None, None
            
        D_sol = sol.y[0]
        f_sol = sol.y[1]
        return f_sol, D_sol

    for om0 in om0_vals:
        if time.time() - start_time > time_limit:
            break
        for w0 in w0_vals:
            if time.time() - start_time > time_limit:
                break
            for wa in wa_vals:
                f_ref, D_ref = compute_reference(om0, w0, wa, a_vals)
                if f_ref is not None:
                    refs.append((f_ref, D_ref))
                    params_list.append((om0, w0, wa))

    if not refs:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    def objective(c):
        c0, c1, c2, c3, c4, c5 = c
        max_err = 0.0
        penalty = 0.0
        
        for i, (f_ref, D_ref) in enumerate(refs):
            om0, w0, wa = params_list[i]
            
            om_m_arr = np.zeros_like(a_vals)
            u_arr = np.zeros_like(a_vals)
            
            for j, a in enumerate(a_vals):
                om_m = om0 * a**(-3)
                om_l = (1.0 - om0) * a**(-3.0 * (1.0 + w0)) * np.exp(3.0 * wa * (1.0 - a))
                h2 = om_m + om_l
                om_m_arr[j] = om_m / h2
                u_arr[j] = 1.0 + w0 + wa * (1.0 - a)
            
            q = 1.0 - om_m_arr
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u_arr + c4 * wa * a_vals + c5 * u_arr * q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                penalty += 1e6 * (np.max(np.abs(np.clip(gamma, 0.1, 1.2) - gamma)) + 1.0)
            
            f_model = om_m_arr**gamma
            
            x_vals = np.log(a_vals)
            integrand = f_model
            cum_int = cumulative_trapezoid(integrand, x_vals, initial=0.0)
            total_int = cum_int[-1]
            integral_x_to_0 = total_int - cum_int
            D_model = np.exp(-integral_x_to_0)
            
            rel_err_f = np.abs(f_model - f_ref) / np.abs(f_ref)
            rel_err_D = np.abs(D_model - D_ref) / np.abs(D_ref)
            
            max_err = max(max_err, np.max(rel_err_f), np.max(rel_err_D))
            
        return max_err + penalty

    # Analytic initial guess based on Linder (2005)
    # gamma ~ 0.55 - 0.05(1+w0) - 0.05 wa
    # Our ansatz: c0 + c1 q + c2 q^2 + c3 u + c4 wa a + c5 u q
    # In matter limit (a->0, q->1, u->1+w0+wa):
    # gamma ~ c0 + c1 + c2 + c3(1+w0+wa) + c4 wa + c5(1+w0+wa)
    # We want this to match 0.55 - 0.05(1+w0+wa)
    # Let's set c1=0, c2=0, c4=0, c5=0 for the initial guess structure
    # c0 + c3(1+w0+wa) = 0.55 - 0.05(1+w0+wa)
    # c3 = -0.05
    # c0 + c3 = 0.55 - 0.05 => c0 + (-0.05) = 0.5 => c0 = 0.55?
    # If c3 = -0.05, c0 = 0.55, then at u=1 (w=0), gamma = 0.5.
    # Linder says for w=0, gamma=0.55.
    # So c0 + c3(1) = 0.55.
    # And for general u, c0 + c3 u = 0.55 - 0.05(u-1)?
    # 0.55 - 0.05 u + 0.05 = 0.60 - 0.05 u.
    # So c0 = 0.60, c3 = -0.05.
    
    x0 = np.array([0.60, 0.0, 0.0, -0.05, 0.0, 0.0])
    
    best_x = x0
    best_score = objective(x0)
    
    if time.time() - start_time < time_limit - 10:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_score:
            best_x = res.x
            best_score = res.fun

    best_x = np.clip(best_x, -10.0, 10.0)
    return best_x
