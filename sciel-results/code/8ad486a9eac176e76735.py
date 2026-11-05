import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy: 
    1. Use a fixed-step 4th-order Runge-Kutta method (RK4).
    2. Choose the number of steps to balance accuracy and the penalty for function calls.
       The score is -log10(err) - 0.5*log10(calls).
       For RK4, error ~ C * h^4 = C * (T/N)^4.
       Calls ~ 4*N.
       Score ~ -log10(C) - 4*log10(T/N) - 0.5*log10(4*N)
             ~ 4*log10(N) - 0.5*log10(N) + const
             ~ 3.5 * log10(N) + const
       This suggests increasing N improves score, limited by stability and time.
       However, the constant C depends on the problem. For stiff problems, h must be small.
       For convection-diffusion with 100 points, let's estimate stability.
       Typical eigenvalues for 1D diffusion/convection on N points are O(1/h^2) for diffusion.
       h_space = L/N. lambda_max ~ -4*D/h_space^2.
       Stability for RK4: |1 + z + z^2/2 + z^3/6 + z^4/24| < 1 for z = lambda * h_time.
       For real negative lambda, limit is z > -2.78.
       So h_time < 2.78 / |lambda_max| ~ 2.78 * h_space^2 / (4*D).
       If D ~ 1, L ~ 1, N=100, h_space = 0.01.
       h_time < 2.78 * 1e-4 / 4 ~ 7e-5.
       T = 0.5. N_steps > 0.5 / 7e-5 ~ 7000.
       Calls = 4 * 7000 = 28000.
       log10(28000) ~ 4.4. Penalty ~ 2.2.
       
       Arnoldi used ~100 calls. log10(100) ~ 2. Penalty ~ 1.
       Arnoldi error is spectral gap dependent.
       
       Let's try a moderate N for RK4 to see if it's competitive.
       If N=1000, calls=4000. Penalty ~ 1.75.
       If N=500, calls=2000. Penalty ~ 1.65.
       
       Let's start with N=1000 and adapt if unstable or slow.
       Actually, the problem is "convection-diffusion". If diffusion is small, it's less stiff.
       But we don't know D.
       
       Let's use an adaptive-like approach within RK4? No, fixed step is requested.
       Let's pick a step size that is likely stable for a wide range of D.
       If D is large, we need small h. If D is small, we can use larger h.
       
       Alternative: Use the fact that we can probe f to estimate stiffness?
       No, the directive says "fixed-step 4th-order Runge-Kutta method with a step size chosen to balance...".
       
       Let's try N=2000. Calls=8000.
       Time cost: 8000 calls * 100 floats. Very fast.
       
       Let's implement RK4 with a fixed N.
       We will choose N=1000 as a starting point.
       
       Wait, the "inspiration" says reducing calls from 101 to 3 improved score.
       This implies the problem might be very simple or the penalty is significant.
       But RK4 requires 4 calls per step.
       
       If I use N=100, calls=400.
       If I use N=10, calls=40.
       
       Let's look at the score formula again.
       Score = -log10(err) - 0.5*log10(calls).
       
       If err = 1e-10, -log10(err) = 10.
       If calls = 400, penalty = 0.5 * 2.6 = 1.3. Score = 8.7.
       If calls = 100, penalty = 1.0. Score = 9.0 (assuming same err).
       
       So fewer calls is better IF error doesn't degrade too much.
       
       RK4 error is 4th order.
       If I reduce N by 10, error increases by 10^4 = 10000.
       log10(err) increases by 4.
       Score drops by 4.
       Penalty drops by 0.5 * log10(10) = 0.5.
       Net change: -3.5.
       So increasing N is always better for RK4, until stability limits.
       
       Therefore, we want the MAXIMUM N allowed by stability and time.
       
       How to find max N?
       We can estimate the spectral radius of A by probing f?
       Or just try a large N and check for blow-up?
       
       Let's try N=5000. Calls=20000.
       If it's stable, it will be very accurate.
       If it's unstable, y will explode.
       
       We can add a check: if norm(y) becomes huge, reduce N and retry.
       
       Strategy:
       1. Try N=10000.
       2. If norm blows up (e.g., > 1e10), try N=5000.
       3. If that blows up, try N=2500.
       4. Etc.
       
       This is a structural change from Krylov (which is exact for linear systems up to Krylov dim) to time-stepping.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290
    
    dt_total = t1 - t0
    if dt_total == 0:
        return y0
    
    # List of candidate step counts, starting high for accuracy, reducing if unstable
    # Stability limit for RK4 on diffusion: h < ~ 0.5 * dx^2 / D
    # dx = 1/n (assuming domain [0,1] or similar, typical for 100 points)
    # If D is large, we need small h.
    
    candidates = [20000, 10000, 5000, 2000, 1000, 500, 200, 100]
    
    y_final = None
    
    for N in candidates:
        if time.time() > deadline:
            break
            
        h = dt_total / N
        t = t0
        y = y0.copy()
        
        stable = True
        # We only need to run a few steps to check stability?
        # No, we need the final result.
        # But if it's unstable, it will blow up quickly.
        # Let's run the full loop but check norm periodically.
        
        for i in range(N):
            if time.time() > deadline:
                stable = False
                break
            
            k1 = f(t, y)
            k2 = f(t + 0.5*h, y + 0.5*h*k1)
            k3 = f(t + 0.5*h, y + 0.5*h*k2)
            k4 = f(t + h, y + h*k3)
            
            y = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
            # Check for explosion every 100 steps or at the end
            if i % 100 == 0 or i == N - 1:
                norm_y = np.linalg.norm(y)
                if norm_y > 1e10:
                    stable = False
                    break
        
        if stable:
            y_final = y
            break
    
    if y_final is None:
        # Fallback: if all failed, return y0 or a very coarse solution
        # This shouldn't happen for reasonable convection-diffusion
        return y0
        
    return y_final
