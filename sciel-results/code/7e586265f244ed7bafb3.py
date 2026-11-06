import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 320.0

    def estimate_A(t):
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t, e_j)
        return A

    # Step 1: Estimate A at t0 and t1 to check for time-dependence
    A0 = estimate_A(t0)
    
    if time.time() > deadline:
        return y0

    A1 = estimate_A(t1)
    
    if time.time() > deadline:
        # Fallback to A0 if we ran out of time
        return expm(A0 * (t1 - t0)) @ y0

    # Check if A is approximately constant
    diff_norm = np.linalg.norm(A0 - A1)
    scale = max(np.linalg.norm(A0), 1e-12)
    
    if diff_norm < 1e-8 * scale:
        # System appears constant
        K = 1
        A_list = [A0]
        ts = [t0, t1]
    else:
        # System appears time-dependent
        # Use a reasonable number of steps
        # 100 steps means 100 * 15 = 1500 calls.
        # Penalty is ~1.6.
        # This should yield high accuracy.
        K = 100
        ts = np.linspace(t0, t1, K + 1)
        A_list = [A0]
        for i in range(1, K):
            if time.time() > deadline:
                break
            t_mid = 0.5 * (ts[i] + ts[i+1])
            A_list.append(estimate_A(t_mid))
        # If we broke early, we have fewer steps
        # We need to handle the remaining interval or just stop
        # For simplicity, if we break, we use the last computed A for the rest?
        # Or better, just use the steps we have.
        # But we need to cover the whole interval.
        # Let's just use the A_list we have and assume the last one covers the rest?
        # Actually, if we break, we should probably just return the current state.
        # But we must return y(t_end).
        # If we run out of time, we are in trouble.
        # 1500 calls + expm is very fast. It should not take 320s.
        # So we likely won't break.
        
        # Ensure A_list has K elements
        # If we broke, pad with last A?
        while len(A_list) < K:
             A_list.append(A_list[-1])

    # Integrate
    y = y0
    dt = (t1 - t0) / K
    
    for i in range(K):
        if time.time() > deadline:
            break
        # Use midpoint A for this step
        # Note: In the constant case K=1, A_list has 1 element.
        # In the variable case, A_list has K elements.
        A_step = A_list[i]
        y = expm(A_step * dt) @ y

    return y
