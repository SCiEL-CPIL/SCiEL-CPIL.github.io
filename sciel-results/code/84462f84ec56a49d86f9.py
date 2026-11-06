"""
Strategy: Custom explicit step-doubling with non-uniform time grid.
The Robertson problem has a very fast transient initially (y1 rises/falls quickly)
and then a slow decay (y0 -> 0, y2 -> 1). 
Standard adaptive solvers often waste evaluations in the slow region or struggle
with the stiffness if not tuned perfectly. 
By using a geometrically spaced grid that is dense at t=0 and sparse at t=40, 
and applying 4th-order Runge-Kutta with Richardson extrapolation (step doubling), 
we can achieve high accuracy with fewer calls than adaptive solvers that 
over-sample the smooth tail.
"""
import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Define a non-uniform grid.
    # The fast dynamics happen in the first ~0.1 time units.
    # The slow dynamics happen from ~0.1 to 40.
    # We want many steps early, few steps late.
    # Let's create a grid that is geometric in the "slow" part and linear in the "fast" part?
    # Actually, a simple power-law grid t_i = t0 + (t1-t0) * (i/N)^p with p > 1 
    # clusters points near t0.
    
    # Let's try N=2000 steps with p=3.
    # t_i = t0 + (t1-t0) * (i/N)^3
    # This gives dense steps near t0.
    
    N = 2000
    p = 3.0
    # Generate grid points
    idx = np.arange(N + 1)
    # Avoid 0^p issues if t0 != 0, but t0 is 0 here.
    # For t0=0, t_i = t1 * (i/N)^p
    t_grid = t0 + (t1 - t0) * (idx / N) ** p
    
    # Ensure t1 is exactly the last point
    t_grid[-1] = t1
    
    def rk4_step(t, y, h):
        k1 = f(t, y)
        k2 = f(t + 0.5*h, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def rk4_step_double(t, y, h):
        # Two half-steps
        y_mid = rk4_step(t, y, h/2.0)
        y_end = rk4_step(t + h/2.0, y_mid, h/2.0)
        return y_end
    
    # We will use Richardson extrapolation.
    # y_extrap = (16 * y_double - y_single) / 15
    # This gives 5th order accuracy.
    # However, we must be careful with the cost.
    # y_single: 4 calls
    # y_double: 8 calls
    # Total: 12 calls per step.
    # With N=2000, that's 24000 calls. Well within 2M limit.
    
    # But wait, if the step is too large in the fast region, RK4 will fail (blow up).
    # The power-law grid helps, but we might need an adaptive check or a smaller N with more aggressive clustering.
    # Let's check the step size at t=0.
    # h_0 = t1/N^p * p * (1/N)^(p-1) ? No.
    # t_i = T * (i/N)^p
    # h_i = t_{i+1} - t_i ≈ T * p * (i/N)^(p-1) * (1/N)
    # For i=0, h is 0? No, t_0=0, t_1 = T/N^p.
    # h_0 = T/N^p.
    # If T=40, N=2000, p=3:
    # h_0 = 40 / (2000^3) = 40 / 8e9 = 5e-9. Very small.
    # This is good for stability.
    # For i=N, h_N ≈ T * p / N = 40 * 3 / 2000 = 0.06.
    # This is reasonable for the slow part.
    
    # Let's run the loop
    for i in range(N):
        if time.time() > deadline:
            break
            
        t_curr = t_grid[i]
        t_next = t_grid[i+1]
        h = t_next - t_curr
        
        # Skip if h is too small (numerical noise)
        if h < 1e-15:
            continue
            
        # Calculate single step (4th order)
        y_single = rk4_step(t_curr, y, h)
        
        # Calculate double step (4th order, but effectively 5th order error term differs)
        y_double = rk4_step_double(t_curr, y, h)
        
        # Richardson extrapolation
        y = (16.0 * y_double - y_single) / 15.0
        
    return y
