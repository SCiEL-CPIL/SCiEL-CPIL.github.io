"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Dormand-Prince 5(4) coefficients
_DP5_C2 = 1/5
_DP5_C3 = 3/10
_DP5_C4 = 4/5
_DP5_C5 = 8/9
_DP5_C6 = 1.0

_DP5_A21 = 1/5
_DP5_A31 = 3/40
_DP5_A32 = 9/40
_DP5_A41 = 44/45
_DP5_A42 = -56/15
_DP5_A43 = 32/9
_DP5_A51 = 19372/6561
_DP5_A52 = -25360/2187
_DP5_A53 = 64448/6561
_DP5_A54 = -212/729
_DP5_A61 = 9017/3168
_DP5_A62 = -355/33
_DP5_A63 = 46732/5247
_DP5_A64 = 49/176
_DP5_A65 = -5103/18656

# 5th order weights
_DP5_B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
# 4th order weights
_DP5_Bstar = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

_DP5_E = _DP5_B - _DP5_Bstar

# Error control coefficients for adaptive step
_DP5_E1 = 35/384 - 5179/57600
_DP5_E2 = 0 - 0
_DP5_E3 = 500/1113 - 7571/16695
_DP5_E4 = 125/192 - 393/640
_DP5_E5 = -2187/6784 - (-92097/339200)
_DP5_E6 = 11/84 - 187/2100
_DP5_E7 = 0 - 1/40

def _dp5_step(f, t, y, h, k1=None):
    """Perform one DP5(4) step, returning y_new, err. k1 can be precomputed."""
    if k1 is None:
        k1 = f(t, y)
    k2 = f(t + h*_DP5_C2, y + h*(_DP5_A21*k1))
    k3 = f(t + h*_DP5_C3, y + h*(_DP5_A31*k1 + _DP5_A32*k2))
    k4 = f(t + h*_DP5_C4, y + h*(_DP5_A41*k1 + _DP5_A42*k2 + _DP5_A43*k3))
    k5 = f(t + h*_DP5_C5, y + h*(_DP5_A51*k1 + _DP5_A52*k2 + _DP5_A53*k3 + _DP5_A54*k4))
    k6 = f(t + h*_DP5_C6, y + h*(_DP5_A61*k1 + _DP5_A62*k2 + _DP5_A63*k3 + _DP5_A64*k4 + _DP5_A65*k5))
    # 5th order solution
    y_new = y + h * (_DP5_B[0]*k1 + _DP5_B[1]*k2 + _DP5_B[2]*k3 + _DP5_B[3]*k4 + _DP5_B[4]*k5 + _DP5_B[5]*k6 + _DP5_B[6]*k6)
    # Error estimate
    err = h * (_DP5_E1*k1 + _DP5_E3*k3 + _DP5_E4*k4 + _DP5_E5*k5 + _DP5_E6*k6 + _DP5_E7*k6)
    return y_new, err, k1

def _norm_err(y, err, rtol, atol):
    """Scaled error norm"""
    return np.linalg.norm(err / (atol + rtol * np.abs(y)))

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = float(t0)
    n = len(y)
    
    deadline = time.time() + 280.0
    call_count = 0
    total_span = t1 - t0
    
    # Fixed step size for DP5 - chosen to give high accuracy
    h = 0.001
    n_steps = int(np.ceil(total_span / h))
    h = total_span / n_steps  # Adjust to land exactly on t1
    
    # First step
    y_new, err, k1 = _dp5_step(f, t, y, h, k1=None)
    call_count += 6
    
    for i in range(1, n_steps):
        if time.time() > deadline:
            break
        if call_count > 1800000:
            break
        
        t = t0 + i * h
        # Compute k1 for current state
        k1 = f(t, y_new)
        call_count += 1
        
        y_new, err, k1 = _dp5_step(f, t, y_new, h, k1=k1)
        call_count += 5
    
    y = y_new
    return y
# EVOLVE-BLOCK-END
