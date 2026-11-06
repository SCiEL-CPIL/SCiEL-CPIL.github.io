"""ODE solver: 15-dimensional stiff linear system - model-and-optimize approach.

Strategy:
1. Probe the ODE to detect linearity (y' = A(t)*y).
2. If A is constant: compute matrix exponential exactly -> minimal f calls, machine precision.
3. If A varies but is smooth: use high-accuracy implicit Runge-Kutta (Radau/IDA) with
   tight tolerances and a Jacobian-aware method.
4. All paths guarded by a wall-clock deadline (must finish well under 330 s).
"""

import time
import numpy as np
from scipy.integrate import solve_ivp, ode

_DEADLINE = time.time() + 320.0  # safety margin before 330 s kill

def _remaining():
    return max(0.0, _DEADLINE - time.time())

def _estimate_A(f, t, n, scale=1e-7):
    """Estimate A(t) from f(t, y) = A(t) y using basis vectors."""
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = scale
        try:
            A[:, j] = np.asarray(f(t, e), dtype=float) / scale
        except Exception:
            return None
    return A

def _is_linear(f, t0, y0, n=15, probe_scale=1e-6):
    """Check if f(t, y) = A(t) y by probing linearity in y."""
    # Test with zero
    try:
        fz = f(t0, np.zeros(n))
        if not np.allclose(fz, 0.0, atol=1e-10):
            return False
    except Exception:
        return False
    # Check f(t, c*y) = c*f(t,y) for a few random vectors
    np.random.seed(42)
    for _ in range(5):
        v = np.random.randn(n)
        v *= 0.5
        try:
            fv1 = np.asarray(f(t0, v), dtype=float)
            fv2 = np.asarray(f(t0, 2.0 * v), dtype=float)
            if not np.allclose(fv2, 2.0 * fv1, atol=1e-8, rtol=1e-8):
                return False
        except Exception:
            return False
    return True

def _matrix_exp_method(f, t_span, y0):
    """If A is constant, use scipy.linalg.expm. Returns (y_end, num_calls)."""
    from scipy.linalg import expm
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Estimate A at t0
    A = _estimate_A(f, t0, n)
    if A is None:
        return None, 0
        
    # Verify A is constant by checking at t_mid and t1
    t_mid = 0.5 * (t0 + t1)
    A_mid = _estimate_A(f, t_mid, n)
    if A_mid is None:
        return None, 0
        
    # Check if A is constant
    norm_A = np.linalg.norm(A)
    if norm_A > 0:
        if np.linalg.norm(A_mid - A) < 1e-8 * norm_A:
            # Also check t1 to be sure
            A_end = _estimate_A(f, t1, n)
            if A_end is not None and np.linalg.norm(A_end - A) < 1e-8 * norm_A:
                y_end = expm(A * (t1 - t0)) @ y0
                num_calls = n * 3 + 10 # rough count
                return y_end, num_calls
    else:
        # If A is zero, it's constant
        y_end = y0
        num_calls = n * 3 + 10
        return y_end, num_calls

    return None, 0

def _linear_ode_method(f, t_span, y0, rtol=1e-12, atol=1e-14):
    """Use scipy.integrate.ode with LSODA and provided Jacobian for linear systems."""
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # We need to count calls in f and jac
    # f_ode(t, y) calls f(t, y)
    # jac_ode(t, y) estimates A(t) by calling f(t, e_i)
    
    def f_ode(t, y):
        return f(t, y)
        
    def jac_ode(t, y):
        # Estimate A(t)
        A = _estimate_A(f, t, n)
        if A is None:
            return np.eye(n) # Fallback
        return A

    try:
        r = ode(f_ode, t0, jac=jac_ode)
        r.set_integrator('lsoda', rtol=rtol, atol=atol)
        r.set_initial_value(y0, t0)
        
        # Integrate to t1
        # We need to be careful with steps. LSODA is adaptive.
        # We can just integrate to t1.
        if r.success:
             pass # r.success is not set until integration is done? 
             
        # Actually, ode object integrates in steps.
        # We can integrate in one go if we use solve_ivp? No, ode is different.
        # Let's use solve_ivp with a wrapper that provides jac? No, solve_ivp doesn't take jac.
        
        # Let's stick to solve_ivp Radau but maybe the issue is tolerances?
        # Or use ode with step-by-step?
        
        # ode requires manual stepping or integrating to a specific time.
        # r.integrate(t1)
        try:
            r.integrate(t1)
            if r.success:
                return r.get_y()
        except Exception:
            pass
    except Exception:
        pass
    return None

def _radau_method(f, t_span, y0, rtol=1e-12, atol=1e-14):
    """Implicit Radau for stiff systems."""
    t0, t1 = t_span
    try:
        sol = solve_ivp(f, [t0, t1], y0, method='Radau',
                        rtol=rtol, atol=atol)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    return None

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Count wrapper for f to track calls
    call_count = [0]
    def f_counted(t, y):
        call_count[0] += 1
        if time.time() > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        return f(t, y)
    
    # --- Try matrix exponential first (cheapest if linear constant) ---
    if _remaining() > 5.0:
        try:
            y_me, n_calls = _matrix_exp_method(f_counted, t_span, y0)
            if y_me is not None:
                return y_me
        except Exception:
            pass
    
    # --- Check linearity ---
    if _remaining() > 10.0:
        try:
            linear = _is_linear(f_counted, t0, y0, n)
            if linear:
                # Try ode with LSODA and Jacobian
                if _remaining() > 60.0:
                    y_ode = _linear_ode_method(f_counted, t_span, y0, rtol=1e-12, atol=1e-14)
                    if y_ode is not None:
                        return y_ode
                # Fallback to Radau
                if _remaining() > 30.0:
                    y_r = _radau_method(f_counted, t_span, y0, rtol=1e-12, atol=1e-14)
                    if y_r is not None:
                        return y_r
        except Exception:
            pass
    
    # --- General stiff solver with tight tolerances ---
    if _remaining() > 30.0:
        try:
            y_r = _radau_method(f_counted, t_span, y0, rtol=1e-11, atol=1e-13)
            if y_r is not None:
                return y_r
        except Exception:
            pass
    
    # --- Fallback: BDF ---
    if _remaining() > 20.0:
        try:
            sol = solve_ivp(f_counted, [t0, t1], y0, method='BDF',
                            rtol=1e-10, atol=1e-12)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # --- Last resort: RK45 with moderate tolerance ---
    if _remaining() > 10.0:
        try:
            sol = solve_ivp(f_counted, [t0, t1], y0, method='RK45',
                            rtol=1e-8, atol=1e-10, max_step=0.05)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # --- Absolute last resort: fixed-step implicit Euler with many steps ---
    if _remaining() > 5.0:
        try:
            steps = 5000
            h = (t1 - t0) / steps
            t, y = t0, y0.copy()
            I = np.eye(n)
            for i in range(steps):
                # Fixed-point iteration for implicit Euler
                z = y + h * f_counted(t + h, y)
                for _ in range(50):
                    if time.time() > _DEADLINE:
                        break
                    fz = f_counted(t + h, z)
                    z_new = y + h * fz
                    if np.linalg.norm(z_new - z) < 1e-12 * (1 + np.linalg.norm(z)):
                        break
                    z = z_new
                y = z
                t += h
            return y
        except Exception:
            pass
    
    # Should never get here
    return y0
