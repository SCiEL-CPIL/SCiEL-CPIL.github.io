import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    def extract_ab(t):
        """
        Extracts A(t) and b(t) from f(t, y) = A(t)y + b(t).
        Uses n+1 function calls.
        """
        b_t = np.array(f(t, np.zeros(n)), dtype=float)
        A_t = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            fi = np.array(f(t, ei), dtype=float)
            A_t[:, i] = fi - b_t
        return A_t, b_t

    try:
        # Strategy: Piecewise Constant Matrix Exponential Integration
        # We assume the system is linear: y' = A(t)y + b(t)
        # We divide [t0, t1] into K segments.
        # In each segment, we approximate A(t) and b(t) as constant (value at start of segment).
        # Solution: y_{k+1} = expm(h * A_k) * y_k + (expm(h * A_k) - I) * inv(A_k) * b_k
        # If A_k is singular or near singular, we use a Taylor expansion for the particular integral
        # or simply use the average value if b is constant and A is small, but for stiff systems A is large.
        # A more robust way to compute the particular integral integral_0^h exp((h-s)A) b ds:
        # If A is invertible: (expm(hA) - I) A^{-1} b
        # If A is not invertible, we can use the series expansion or solve the linear system.
        # Since n=15, we can precompute the matrix exponential and the "integral" matrix.
        
        # Let's first check if the system is constant over the whole domain.
        A0, b0 = extract_ab(t0)
        A1, b1 = extract_ab(t1)
        
        if np.allclose(A0, A1, atol=1e-10, rtol=1e-10) and np.allclose(b0, b1, atol=1e-10, rtol=1e-10):
            # Constant system: Single matrix exponential
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = A0
            M[:n, n] = b0
            Z0 = np.zeros(n + 1)
            Z0[:n] = y0
            Z0[n] = 1.0
            
            Z_end = expm(M) @ Z0
            return Z_end[:n]
            
        # Time-varying system: Use Piecewise Constant Approximation
        # We need to choose K (number of segments).
        # Too few K -> large approximation error.
        # Too many K -> high function call count (penalty) and compute time.
        # Let's start with a moderate K and increase if needed, or just pick a good K.
        # Given the score formula, we want high accuracy (low error) and low calls.
        # Error scales as O(h^2) for this method (first order in h for the approximation of A(t)? 
        # Actually, if A(t) is smooth, the local error is O(h^3) if we use midpoint, 
        # but using start point is O(h^2) local, O(h) global? 
        # Wait, this is a linear system. The error comes from approximating A(t) and b(t) as constant.
        # If A(t) is linear in t, the error is O(h^2).
        # Let's use a slightly higher order approximation: Midpoint rule for A and b.
        # y_{k+1} = expm(h * A_mid) y_k + ...
        # This requires evaluating A at t_mid.
        # Calls per segment: 2 * (n+1).
        
        # Let's try K=50 segments first.
        # Total calls approx 50 * 2 * 16 = 1600.
        # log10(1600) ~ 3.2. Penalty ~ 1.6.
        
        # If we use K=100, calls ~ 3200. Penalty ~ 1.8.
        # If we use K=20, calls ~ 640. Penalty ~ 1.4.
        
        # Let's implement a loop that increases K if time allows, or just pick a safe K.
        # Given stiffness, smaller h is better for stability/accuracy of the piecewise approximation.
        
        K = 100
        h = (t1 - t0) / K
        y = y0.copy()
        
        for k in range(K):
            t_start = t0 + k * h
            t_mid = t_start + h / 2.0
            
            # Check time
            if time.time() > deadline - 10:
                # Fallback to simple RK4 if running out of time, though unlikely for K=100
                # But we are in the loop, so we must finish.
                # Just break and return current y? No, that's inaccurate.
                # We should have chosen K such that it finishes fast.
                pass
            
            # Extract A and b at midpoint for better accuracy (2nd order)
            A_mid, b_mid = extract_ab(t_mid)
            
            # Compute matrix exponential
            E = expm(h * A_mid)
            
            # Compute particular integral: int_0^h exp((h-s)A_mid) b_mid ds
            # = (E - I) * inv(A_mid) * b_mid  if A_mid is invertible
            # To avoid inversion issues, we can augment the matrix or use solve.
            # Let's use the augmented matrix trick for the whole step?
            # Or just:
            # We need to solve: X = (E - I) A^{-1} b
            # This is equivalent to solving A X = (E - I) b
            try:
                # Solve A_mid * X = (E - I) * b_mid
                rhs = (E - np.eye(n)) @ b_mid
                X = np.linalg.solve(A_mid, rhs)
            except np.linalg.LinAlgError:
                # Fallback: use series expansion or pseudo-inverse
                # For stiff systems, A is usually well-conditioned enough for solve.
                # If it fails, use pinv
                X = np.linalg.pinv(A_mid) @ rhs
            
            y = E @ y + X
            
        return y

    except Exception:
        # Fallback: RK4 with moderate steps
        N = 1000
        h = (t1 - t0) / N
        t = t0
        y = y0.copy()
        for _ in range(N):
            k1 = np.array(f(t, y), dtype=float)
            k2 = np.array(f(t + h/2, y + h/2 * k1), dtype=float)
            k3 = np.array(f(t + h/2, y + h/2 * k2), dtype=float)
            k4 = np.array(f(t + h, y + h * k3), dtype=float)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
        return y
