"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Adaptive-step Runge-Kutta with embedded error estimation.
# We use a 4th/5th order pair (Dormand-Prince) to estimate local error and adjust step size.
# This minimizes the number of function calls (f-evals) required to achieve high accuracy,
# which is crucial for the score = -log10(rel) - 0.5 * log10(calls).
#
# The Van der Pol oscillator with mu=5 is moderately stiff.
# An adaptive method will take larger steps in smooth regions and smaller steps
# near the sharp transitions, potentially using far fewer calls than a fixed-step
# method that must be small enough for the worst-case stiffness.

# Coefficients for Dormand-Prince 5(4)
# c_i
c = [0, 1/5, 3/10, 4/5, 8/9, 1]
# a_ij
a = [
    [0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
]
# b5 (5th order weights)
b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
# b4 (4th order weights)
b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 280s limit to be safe within 300s
    
    # Adaptive parameters
    rtol = 1e-11
    atol = 1e-13
    # Initial step size
    h = (t1 - t0) / 1000
    if h <= 0:
        return y
        
    t = t0
    calls = 0
    
    # Pre-compute constants for speed
    # We will store k1..k6 in a list for clarity, though unrolling is faster.
    # Given the 300s limit, unrolling is safer for performance.
    
    # Error control safety
    max_steps = 50000
    step_count = 0
    
    while t < t1:
        if time.time() > deadline:
            break
        if step_count > max_steps:
            break
            
        # Ensure we don't overshoot t1
        if t + h > t1:
            h = t1 - t
        if h <= 0:
            break
            
        # --- RK45 Step ---
        
        # k1
        k1 = f(t, y)
        calls += 1
        
        # k2
        y2 = y + h * (a[1][0] * k1)
        k2 = f(t + c[1] * h, y2)
        calls += 1
        
        # k3
        y3 = y + h * (a[2][0] * k1 + a[2][1] * k2)
        k3 = f(t + c[2] * h, y3)
        calls += 1
        
        # k4
        y4 = y + h * (a[3][0] * k1 + a[3][1] * k2 + a[3][2] * k3)
        k4 = f(t + c[3] * h, y4)
        calls += 1
        
        # k5
        y5 = y + h * (a[4][0] * k1 + a[4][1] * k2 + a[4][2] * k3 + a[4][3] * k4)
        k5 = f(t + c[4] * h, y5)
        calls += 1
        
        # k6
        y6 = y + h * (a[5][0] * k1 + a[5][1] * k2 + a[5][2] * k3 + a[5][3] * k4 + a[5][4] * k5)
        k6 = f(t + c[5] * h, y6)
        calls += 1
        
        # Compute 5th order solution (y_new) and 4th order solution (y_check)
        # y_new = y + h * sum(b5[i] * k[i+1])
        # y_check = y + h * sum(b4[i] * k[i+1])
        
        # Note: b4 has 7 elements, but k has 6. The last term in b4 is for the 7th stage in some formulations?
        # Standard DP4/5 uses 6 stages. 
        # Let's check the b4 vector length.
        # Standard DP5 b4: [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        # This implies a 7-stage method? No, Dormand-Prince is 6 stages.
        # Wait, the standard DP5 coefficients are:
        # b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        # b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        # Actually, the 4th order solution is often computed as y + h*(sum(b5) - sum(b4_diff))?
        # Or simply using the specific b4 weights.
        # Let's use the difference estimator: err = h * (b5 - b4) * k?
        # Standard DP error estimator: e = y5 - y4 = h * (b5 - b4) * k?
        # Actually, b4 in DP5 is defined such that y4 = y + h * sum(b4_i * k_i).
        # But b4 has 7 elements?
        # Let's verify the standard DP5 coefficients.
        # c = [0, 1/5, 3/10, 4/5, 8/9, 1]
        # a1 = [1/5]
        # a2 = [3/40, 9/40]
        # a3 = [44/45, -56/15, 32/9]
        # a4 = [19372/6561, -25360/2187, 64448/6561, -212/729]
        # a5 = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        # b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        # b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        # The b4 vector has 7 entries?
        # Ah, the 7th entry corresponds to k7? No, there are only 6 stages.
        # In many implementations, the 4th order solution is computed as:
        # y4 = y + h * (b5 - e) where e is the error vector?
        # Or y4 = y + h * sum(b4_i * k_i) for i=1..6?
        # Let's check the length of b4 again.
        # If b4 has 7 elements, it might be a typo in my memory or the standard list includes a 7th stage for the 4th order method?
        # No, DP5 is 6-stage.
        # Let's look at a reliable source.
        # b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        # This vector has 7 elements.
        # This is confusing.
        # Let's use the error estimation formula directly:
        # err = h * (b5 - b4) * k ?
        # If b4 has 7 elements and k has 6, we can't do that directly.
        # Actually, the 4th order solution is often just y + h * sum(b4_i * k_i) for the first 6?
        # Let's assume the standard implementation where err = h * (b5 - b4) * k is not used directly if lengths mismatch.
        # Instead, let's use the fact that y5 is the 5th order solution.
        # And we can estimate error as norm(y5 - y4).
        # To get y4, we can use the weights for the 4th order method.
        # If the b4 vector provided is correct, maybe it corresponds to a different indexing?
        # Let's try to use the standard DP5 error estimator:
        # e = h * (b5 - b4) * k
        # If I am unsure about the b4 length, I can use a simpler 3/2 order pair (RK4/RK2) or just use a very small fixed step?
        # But fixed step is what we are moving away from.
        
        # Let's hardcode the error estimator for DP5 which is well-known:
        # err = h * ( [ -2825/27648, 0, 12815/40494, -2197/4116, 2152/15625, 1/1440, 1/40 ] * k )
        # Wait, the vector [ -2825/27648, 0, 12815/40494, -2197/4116, 2152/15625, 1/1440, 1/40 ] has 7 elements.
        # This suggests there is a 7th stage k7?
        # No, the 7th term in the error estimator usually corresponds to the difference in the last stage?
        # Actually, in DP5, the 4th order solution uses the same 6 stages.
        # The error vector is often defined as e = y5 - y4.
        # y5 = y + h * sum(b5_i k_i)
        # y4 = y + h * sum(b4_i k_i)
        # If b4 has 7 terms, maybe the last term is for a 7th stage that is identical to the 6th?
        # Or maybe I should just use a fixed step with a very high order method?
        # The prompt says "STAGNATION... Make a STRUCTURAL change".
        # Using scipy.integrate.solve_ivp was successful in inspiration 1.
        # Let's just use scipy.integrate.solve_ivp with Radau, which is robust and handles stiffness.
        # It might use more calls, but it's guaranteed to be accurate and stable.
        # The score penalty for calls is log10. 100k calls is -5. 1M calls is -10.
        # If Radau uses 100k calls, penalty is -5.
        # If rel is 1e-14, gain is 14. Total 9.
        # Current fixed step DP5 h=0.002 uses 30k calls. Penalty -2.2.
        # If rel is 1e-14, total 11.8.
        # So fixed step DP5 is likely better IF it achieves 1e-14.
        # But does it?
        # With h=0.002, local error ~ h^5 = 3e-15. Global error ~ N * local error ~ 5000 * 3e-15 = 1.5e-11.
        # This is NOT 1e-14.
        # So fixed step DP5 with h=0.002 will have rel ~ 1e-11.
        # Score = 11 - 2.2 = 8.8.
        # To get rel 1e-14 with DP5, we need h^5 ~ 1e-14/5000 = 2e-18 => h ~ 1.8e-4.
        # N = 10/1.8e-4 = 55555 steps.
        # Calls = 6 * 55555 = 333330.
        # Penalty = -0.5 * log10(333330) = -0.5 * 5.52 = -2.76.
        # Score = 14 - 2.76 = 11.24.
        
        # So we need a smaller h for fixed step, or an adaptive step.
        # Adaptive step with DP5 will try to keep local error ~ rtol.
        # If we set rtol=1e-14, it will take small steps where needed.
        # Let's implement the adaptive DP5 properly.
        
        # Correct DP5 error estimator:
        # The error is estimated as e = h * (b5 - b4) * k
        # But b4 and b5 must be same length.
        # Standard DP5:
        # b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        # b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        # This is inconsistent.
        # Let's use the coefficients from a trusted source for DP5.
        # From Hairer et al.:
        # b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        # b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        # It seems b4 has 7 elements.
        # Ah, the 7th element of b4 is 1/40.
        # And the 6th element is 187/2100.
        # The error vector is e = [ -2825/27648, 0, 12815/40494, -2197/4116, 2152/15625, 1/1440, 1/40 ]
        # This also has 7 elements.
        # This implies there are 7 stages?
        # No, DP5 is 6 stages.
        # Maybe the last term in the error vector is for the 6th stage?
        # Let's check the indices.
        # k1..k6.
        # If the error vector has 7 terms, maybe the last one is 0?
        # Or maybe I am misremembering the number of stages.
        # Let's assume 6 stages and use the first 6 terms of the error vector?
        # err = h * ( e1*k1 + ... + e6*k6 )
        # e = [ -2825/27648, 0, 12815/40494, -2197/4116, 2152/15625, 1/1440 ]
        # And ignore the 7th term 1/40?
        # Or is the 7th term for a k7 that is equal to k6?
        # Let's try to find a definitive DP5 implementation online or in scipy source.
        # scipy.integrate.dopri5 uses:
        # c = [0, 1/5, 3/10, 4/5, 8/9, 1]
        # a = [[0,0,0,0,0,0], [1/5,0,0,0,0,0], [3/40,9/40,0,0,0,0], [44/45,-56/15,32/9,0,0,0], [19372/6561,-25360/2187,64448/6561,-212/729,0,0], [9017/3168,-355/33,46732/5247,49/176,-5103/18656,0]]
        # b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        # b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        # It really does have 7 elements for b4.
        # This is very strange.
        # Let's look at the error estimator in scipy:
        # err = h * (b5 - b4) * k
        # If b4 has 7 elements and k has 6, this will fail.
        # Unless k is padded?
        # Or maybe the 7th element of b4 is for the 6th stage, and the 6th element is for the 5th?
        # No, the order is k1..k6.
        # Let's assume the error estimator is:
        # err = h * ( [ -2825/27648, 0, 12815/40494, -2197/4116, 2152/15625, 1/1440 ] * k )
        # And the 7th term 1/40 is ignored or corresponds to something else.
        # Actually, let's just use a simpler method: RK4 with step doubling.
        # RK4 is 4th order. Step doubling gives 4th order error estimate.
        # 2 steps of h/2 vs 1 step of h.
        # Calls: 4 (for h) + 2*4 (for h/2) = 12 calls per step.
        # This is inefficient.
        
        # Alternative: Use scipy.integrate.solve_ivp with Radau.
        # It is robust, handles stiffness, and is likely to achieve high accuracy.
        # The call count might be higher, but the accuracy will be guaranteed.
        # Given the risk of coefficient errors in manual RK5, using scipy is safer.
        
        # Let's switch to scipy.integrate.solve_ivp.
        
        from scipy.integrate import solve_ivp
        
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-11, atol=1e-13)
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to fixed step if solve_ivp fails
            h = 0.0001
            N = int(np.ceil((t1 - t0) / h))
            h = (t1 - t0) / N
            y = np.array(y0, dtype=float)
            t = t0
            for _ in range(N):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y

# EVOLVE-BLOCK-END
