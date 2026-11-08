"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Dormand-Prince coefficients (DP54)
# This method uses 6 stages but the 6th stage reuses the first stage of the next step (FSAL).
# However, for simplicity and robustness in a fixed budget, we will implement a standard 6-stage DP5
# and use the 5th order result for the step, using the 4th order estimate for error control.
# Coefficients:
c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
A = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
])
b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

# But-cher tableau for FSAL is complex to implement perfectly in a short time.
# We will use a standard 6-evaluation step.
# To save calls, we can use the fact that for Van der Pol, the dynamics are smooth.
# We will implement an adaptive step size controller targeting local error <= 1e-13.

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Initial step size estimate
    h = (t1 - t0) / 100.0
    # Safety margin for time limit
    start_time = time.time()
    time_limit = 250.0  # 50s buffer before 300s kill
    
    # Error tolerances
    rtol = 1e-11
    atol = 1e-13
    
    # DP54 Coefficients (Standard, not FSAL optimized for simplicity, but we can optimize if needed)
    # Let's use the standard 6 stages.
    # k1 = f(t, y)
    # k2 = f(t + c2*h, y + h*A21*k1)
    # ...
    
    # Precompute A and b for speed
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ])
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]) # Note: b4 has 7 elements in some tables, here we align with 6 stages? 
    # Wait, standard DP54 has 6 stages. b4 usually corresponds to the 4th order estimate.
    # Let's verify coefficients.
    # Standard DP54:
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1]
    # A rows:
    # 1: [1/5, 0, 0, 0, 0]
    # 2: [3/40, 9/40, 0, 0, 0]
    # 3: [44/45, -56/15, 32/9, 0, 0]
    # 4: [19372/6561, -25360/2187, 64448/6561, -212/729, 0]
    # 5: [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    # b5 (5th order): [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    # b4 (4th order): [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40] -> This has 7 elements? 
    # Ah, the last element 1/40 is for the 6th stage which is often reused. 
    # In a non-FSAL implementation, we compute 6 k's.
    # Let's stick to a robust 4th order RK4 with adaptive step? No, DP5 is better.
    # Let's use a simpler, very robust approach: 4th order Runge-Kutta with adaptive step size.
    # It uses 4 calls per step. DP5 uses 6. 
    # However, DP5 has higher order error constant, so it needs fewer steps.
    # Let's try to implement DP5 correctly.
    
    # Correct DP54 coefficients for 6 stages:
    c = [0, 1/5, 3/10, 4/5, 8/9, 1]
    A = [
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ]
    b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    # Wait, b4 has 7 elements in many references because of the FSAL property where k7=k1_next.
    # If we do NOT use FSAL, we calculate k1..k6.
    # The 4th order weights for the first 6 stages are:
    b4_6 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100]
    # The 5th order weights for the first 6 stages are:
    b5_6 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    
    # Let's verify the sum of b4_6 and b5_6.
    # This is getting risky with manual coefficient verification.
    # Alternative: Use a high-precision fixed-step RK4? 
    # The seed used 2000 steps. 
    # If we use 10000 steps of RK4, calls = 40000.
    # Score penalty: -0.5 * log10(40000) = -0.5 * 4.6 = -2.3.
    # Seed score 1.0 implies base score around 1.0 + 2.3 + ...
    # If we use DP5 adaptive, we might get away with ~500-1000 steps (6 calls each) = 3000-6000 calls.
    # log10(6000) = 3.8. Penalty -1.9.
    # Gain in accuracy allows smaller error term.
    
    # Let's implement a robust Adaptive RK4 (Bogacki-Pratt or similar) or just Adaptive RK4.
    # Adaptive RK4 uses 4 calls per accepted step.
    
    def rk4_step(t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Error estimation: Take two half-steps vs one full step.
    # Calls: 2 * 4 = 8 calls per step.
    # This is expensive.
    
    # Better: Use a 5th order method with 6 calls (DP5) if I can trust the coefficients.
    # Let's use the coefficients from a reliable source in my memory.
    # Or, simpler: Just use a very small fixed step size for RK4?
    # The Van der Pol oscillator with mu=5 is not extremely stiff.
    # Period is approx 2*pi + mu*2 = 16.6.
    # Frequency is moderate.
    # Stability limit for RK4 is h < 2.78 / lambda_max.
    # Eigenvalues for VDP are around O(1) to O(mu).
    # Let's try a fixed step of 1e-3? 
    # Steps = 10 / 1e-3 = 10000. Calls = 40000.
    # This is likely what the seed is doing effectively (2000 steps might be too coarse for 1e-14 rel error).
    # Seed rel error might be high, hence score 1.0 is just baseline.
    
    # Strategy: Adaptive RK4 with step doubling.
    # 8 calls per step.
    # Target local error 1e-14.
    
    y = np.array(y0, dtype=float)
    t = t0
    h = 1e-3 # Initial guess
    max_h = (t1 - t0) / 10.0
    min_h = 1e-8
    
    while t < t1:
        if time.time() - start_time > time_limit:
            break
            
        # Limit step size to end of interval
        if t + h > t1:
            h = t1 - t
        
        # Step doubling error estimation
        y_full = rk4_step(t, y, h)
        
        y_half1 = rk4_step(t, y, h/2)
        y_half2 = rk4_step(t + h/2, y_half1, h/2)
        
        err = np.linalg.norm(y_half2 - y_full, ord=np.inf)
        scale = np.maximum(1e-15, np.maximum(atol, rtol * np.abs(y_half2)))
        err_norm = np.linalg.norm(err / scale, ord=np.inf)
        
        if err_norm <= 1.0:
            # Accept step
            y = y_half2
            t += h
            # Increase step size
            if err_norm > 0:
                factor = 0.9 * (err_norm ** (-0.2))
            else:
                factor = 5.0
            h_new = h * min(factor, 5.0)
            h = min(h_new, max_h)
            h = max(h, min_h)
        else:
            # Reject step, decrease h
            factor = 0.9 * (err_norm ** (-0.2))
            h = h * max(factor, 0.1)
            h = max(h, min_h)
            
    return y
# EVOLVE-BLOCK-END
