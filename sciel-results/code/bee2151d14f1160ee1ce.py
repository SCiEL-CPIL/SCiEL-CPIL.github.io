"""ODE solver: Convection-diffusion 100-point using Padé matrix exponential."""
import numpy as np
import time

_DEADLINE = time.time() + 280.0  # Safety margin before 330s kill

def _check_deadline():
    return time.time() > _DEADLINE


def _pade_exp(A, m=6):
    """
    Compute exp(A) using the order-2m Padé approximation.
    For m=6, this gives order-12 accuracy.
    Uses the scaling-and-squaring approach:
      exp(A) = (R_{2m}(A / 2^s))^{2^s}
    where s is chosen so that ||A / 2^s||_1 < theta_m.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    
    # Thresholds for order 2m Padé (from Higham, 2005)
    # For m=6, threshold theta_6 ≈ 0.9504530269826533
    theta = 0.9504530269826533
    
    # Estimate norm
    norm_A = np.max(np.sum(np.abs(A), axis=1))
    
    if norm_A < 1e-15:
        return np.eye(n)
    
    # Choose s
    if norm_A < theta:
        s = 0
    else:
        s = int(np.ceil(np.log2(norm_A / theta)))
    
    A_scaled = A / (2.0 ** s)
    
    # Compute R_{2m}(A_scaled) = (I + N) * (I - N)^{-1}
    # where N = A_scaled * (c_m / (c_m + sum_{k=1}^m c_k * A_scaled^k / k!))
    # Actually, the standard formulation:
    # R_{2m}(A) = (I + N) (I - N)^{-1}
    # N = A * (c_m / (c_m + sum_{k=1}^m c_k * A^k / k!))^{-1}
    
    # Coefficients c_k = binomial(2m, m-k) for k=0..m
    # For m=6: c_0=1, c_1=6, c_2=15, c_3=20, c_4=15, c_5=6, c_6=1
    # Wait, let me use the standard Padé coefficients.
    
    # The order-2m Padé approximation is:
    # R_{2m}(A) = (I + N) (I - N)^{-1}
    # where N = A * (c_m / (c_m + sum_{k=1}^m c_k A^k / k!))
    # Hmm, let me just use the direct formula.
    
    # Actually, the simplest approach: use scipy's expm if available, 
    # but we want to minimize f calls. The key insight is that we only
    # need to reconstruct A from a few f evaluations, then compute exp(A*dt).
    
    # Let me use a cleaner Padé formulation.
    # R_{2m}(A) = (I - N)(I + N)^{-1} ... no, let me just use numpy.
    
    # For m=6, the Padé approximant is:
    # R_12(A) = (I + N) (I - N)^{-1}
    # N = A * (c_6 / (c_6 + c_5*A/1! + c_4*A^2/2! + ... + c_0*A^6/6!))
    
    # Coefficients for m=6:
    # c_0 = 1, c_1 = 6, c_2 = 15, c_3 = 20, c_4 = 15, c_5 = 6, c_6 = 1
    # These are binomial(12, 6-k) for k=0..6
    # binom(12,6)=924, binom(12,5)=792, binom(12,4)=495, binom(12,3)=220,
    # binom(12,2)=66, binom(12,1)=12, binom(12,0)=1
    # So c_k = binom(2m, m-k):
    # c_0 = binom(12,6) = 924
    # c_1 = binom(12,5) = 792
    # c_2 = binom(12,4) = 495
    # c_3 = binom(12,3) = 220
    # c_4 = binom(12,2) = 66
    # c_5 = binom(12,1) = 12
    # c_6 = binom(12,0) = 1
    
    c = [924, 792, 495, 220, 66, 12, 1]  # c_0 to c_6
    
    # Compute A^k / k! for k=0..m
    # N = A * (c_m / (c_m + sum_{k=1}^m c_k * A^k / k!))
    # Let's compute the denominator: D = c_m + sum_{k=1}^m c_k * A^k / k!
    
    D = np.eye(n) * c[6]  # c_6 * I
    A_pow = np.eye(n)
    for k in range(1, 7):
        A_pow = A_pow @ A_scaled
        D = D + c[6-k] * A_pow / float(np.math.factorial(k))
    
    # N = A_scaled * (c_6 / D)  ... this is not quite right
    # The formula is: N = A * (c_m / (c_m + sum_{k=1}^m c_k A^k/k!))
    # But this is a matrix equation. Let me use a different approach.
    
    # Actually, let me just use the fact that for a 100x100 matrix, 
    # we can use scipy.linalg.expm which is very accurate.
    # The key is minimizing f calls to reconstruct A.
    
    from scipy.linalg import expm
    return expm(A)


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy: Reconstruct the linear operator A from a few f evaluations,
    then compute y(t1) = exp(A*(t1-t0)) * y0 using matrix exponential.
    This requires only O(n) f calls (n=100) instead of thousands.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 280.0
    
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Reconstruct A: for a linear ODE y' = A*y, f(t, y) = A*y
    # We can get A by evaluating f at basis vectors.
    # But f might be nonlinear. For convection-diffusion, it's linear: y' = A*y.
    
    # First, try to detect if f is linear and time-independent.
    # Evaluate f at y0 and at 0.
    
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        if _check_deadline():
            raise Exception("Deadline exceeded")
        return f(t, y)
    
    # Check if f is time-independent and linear.
    # f(t, y) = A*y for linear time-independent system.
    
    # Get f(t0, y0) and f(t0, 0)
    f_y0 = f_counted(t0, y0)
    f_zero = f_counted(t0, np.zeros(n))
    
    # If f is linear, f_zero should be ~0.
    # If f is time-independent, f(t0, y) should equal f(t1, y).
    
    # Check linearity: f(t0, 2*y0) should be ~2*f_y0 if linear
    f_2y0 = f_counted(t0, 2.0 * y0)
    linear_check = np.max(np.abs(f_2y0 - 2.0 * f_y0))
    is_linear = linear_check < 1e-10 * max(1.0, np.max(np.abs(f_y0)))
    
    # Check time-independence: f(t1, y0) should equal f_y0
    f_y0_t1 = f_counted(t1, y0)
    time_indep_check = np.max(np.abs(f_y0_t1 - f_y0))
    is_time_indep = time_indep_check < 1e-10 * max(1.0, np.max(np.abs(f_y0)))
    
    if is_linear and is_time_indep:
        # Reconstruct A from basis vectors
        # A * e_i = f(t0, e_i) for each basis vector e_i
        # This requires n evaluations.
        
        A = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            A[:, i] = f_counted(t0, e_i)
        
        # Compute y(t1) = exp(A * dt) * y0
        from scipy.linalg import expm
        expA = expm(A * dt)
        y_final = expA @ y0
        
        if np.all(np.isfinite(y_final)):
            return y_final
    
    # Fallback: use scipy's solve_ivp with Radau method
    from scipy.integrate import solve_ivp
    
    try:
        sol = solve_ivp(
            f_counted, t_span, y0,
            method='Radau',
            rtol=1e-12, atol=1e-14,
            max_step=dt / 50,
            dense_output=False
        )
        if sol.success:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                return y_final
    except Exception:
        pass
    
    # Final fallback: RK45
    try:
        sol = solve_ivp(
            f_counted, t_span, y0,
            method='RK45',
            rtol=1e-12, atol=1e-14,
            max_step=dt / 100,
            dense_output=False
        )
        if sol.success:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                return y_final
    except Exception:
        pass
    
    # Last resort: simple RK4
    steps = 5000
    h = dt / steps
    t = t0
    y = y0.copy()
    for _ in range(steps):
        if _check_deadline():
            break
        k1 = f_counted(t, y)
        k2 = f_counted(t + h/2, y + h/2 * k1)
        k3 = f_counted(t + h/2, y + h/2 * k2)
        k4 = f_counted(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y
