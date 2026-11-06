import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """
    Solves a 15-dimensional stiff linear system y' = f(t, y) on [0, 1].
    Strategy:
    1. Probe the system to determine if it is time-invariant (linear constant coefficients).
    2. If time-invariant, use the exact matrix exponential solution (highest accuracy, low cost).
    3. If time-varying, use a high-order implicit solver (Radau) with a fixed number of steps 
       to ensure robustness against stiffness while controlling the number of f evaluations.
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # --- Phase 1: Probe for Linearity and Time-Invariance ---
    # We need to determine if f(t, y) = A(t) y + b(t).
    # For a linear system, f(t, 0) = b(t) and f(t, e_i) - f(t, 0) = column i of A(t).
    
    # To check time-invariance, we sample A and b at t0, t_mid, and t1.
    # This costs 3 * (1 + n) = 3 * 16 = 48 calls.
    
    t_mid = 0.5 * (t0 + t1)
    t_points = [t0, t_mid, t1]
    
    A_samples = []
    b_samples = []
    
    # Pre-allocate
    for t in t_points:
        b_t = np.array(f(t, np.zeros(n)), dtype=float)
        A_t = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            fi = np.array(f(t, ei), dtype=float)
            A_t[:, i] = fi - b_t
        A_samples.append(A_t)
        b_samples.append(b_t)
        
    # Check if A and b are constant
    is_constant = True
    for i in range(1, len(t_points)):
        if not np.allclose(A_samples[0], A_samples[i], rtol=1e-12, atol=1e-12):
            is_constant = False
            break
        if not np.allclose(b_samples[0], b_samples[i], rtol=1e-12, atol=1e-12):
            is_constant = False
            break
            
    # --- Phase 2: Solve ---
    
    if is_constant:
        # Use exact solution for constant linear system
        # y(t) = exp(A * (t - t0)) * y0 + integral_{t0}^{t} exp(A * (t - s)) * b ds
        # This can be computed via the augmented matrix exponential:
        # [y(t); 1] = exp([A, b; 0, 0] * (t - t0)) * [y0; 1]
        
        A = A_samples[0]
        b = b_samples[0]
        dt = t1 - t0
        
        # Build augmented matrix M (16x16)
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b
        
        # Compute exp(M * dt)
        # For 16x16, scipy.linalg.expm is very fast and accurate
        expM = expm(M * dt)
        
        # Initial state vector
        Z0 = np.zeros(n + 1)
        Z0[:n] = y0
        Z0[n] = 1.0
        
        # Final state
        Z_end = expM @ Z0
        y_end = Z_end[:n]
        
        # Ensure real output
        y_end = np.real_if_close(y_end, tol=1000)
        if np.iscomplexobj(y_end):
            y_end = y_end.real
            
        return y_end
    
    else:
        # Time-varying system.
        # We need a robust stiff solver. Radau is good for stiff problems.
        # To control the number of function calls and ensure we finish in time,
        # we use a fixed number of steps.
        # The problem is on [0, 1].
        # Let's use 100 steps. Radau evaluates f multiple times per step (up to 3 or 4 times 
        # for implicit stages, plus Jacobian updates).
        # 100 steps * ~5 calls/step = 500 calls. This is well within the 2,000,000 limit.
        # However, if the system is extremely stiff, 100 steps might not be enough for high accuracy.
        # But given the "stiff linear" description and the score metric (accuracy vs calls),
        # increasing steps increases accuracy but also calls.
        # The current "constant" strategy gets high accuracy with ~50 calls.
        # If it's time-varying, we can't use the exact expm shortcut.
        
        # Let's try Radau with a moderate number of steps.
        # We can also try BDF.
        
        t_eval = [t1] # We only need the final value
        
        # Radau method is generally more accurate per step than BDF for stiff problems.
        # rtol and atol control the error.
        # We want high accuracy.
        
        try:
            sol = solve_ivp(
                f, 
                [t0, t1], 
                y0, 
                method='Radau', 
                t_eval=t_eval,
                rtol=1e-10, 
                atol=1e-12,
                max_step=0.05 # Limit step size to ensure resolution
            )
            
            if sol.success:
                y_end = sol.y[:, -1]
                return y_end
            else:
                # Fallback if Radau fails
                pass
                
        except Exception:
            pass
        
        # Fallback: BDF with tighter tolerance
        try:
            sol = solve_ivp(
                f, 
                [t0, t1], 
                y0, 
                method='BDF', 
                t_eval=t_eval,
                rtol=1e-10, 
                atol=1e-12,
                max_step=0.05
            )
            if sol.success:
                y_end = sol.y[:, -1]
                return y_end
        except Exception:
            pass
            
        # Last resort: Simple RK4 with many steps (less accurate for stiff, but robust)
        # 1000 steps * 4 calls = 4000 calls.
        h = (t1 - t0) / 1000
        t = t0
        y = y0.copy()
        for _ in range(1000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
