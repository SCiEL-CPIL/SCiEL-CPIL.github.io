import numpy as np
import time
import scipy.integrate
import scipy.linalg

def integrate(f, t_span, y0):
    """
    Integrates a stiff 15-dimensional linear ODE system.
    
    Strategy:
    1. Assume the system is linear: y' = A(t)y + b(t).
    2. Estimate A(t) and b(t) at several points to detect if it's time-varying.
    3. If A is constant, use expm (exact).
    4. If A is time-varying, use a **Piecewise Constant Approximation (PCA)** 
       with **Matrix Exponential** for each step. 
       
       This is a structural change from the current program's fallback (adaptive BDF/Radau).
       PCA with expm is a first-order method in the variation of A, but it is A-stable 
       and exploits the linearity exactly within each step. 
       
       By choosing a sufficient number of steps (e.g., 100-1000), we can achieve high accuracy 
       for slowly varying A while remaining stable for stiff A. 
       
       Number of calls:
       - Estimation of A at each step start: 2*n calls per step.
       - Total steps M. Total calls ~ M * 2*n.
       - For M=100, n=15, calls = 3000.
       - Score potential: -log10(error) - 0.5*log10(3000).
       
       If the error is ~1e-10 (likely with M=100 for smooth A), score ~ 10 - 2.5 = 7.5.
       This is worse than the constant case (13.5), but better than generic Radau if Radau struggles.
       
       However, if the system is actually constant A but the detection failed, PCA will still work 
       but with more calls. 
       
       Let's refine the detection and PCA.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0

    # Step 1: Estimate A and b at t0
    h = 1e-5
    
    f_zero_t0 = f(t0, np.zeros(n))
    is_homogeneous = np.linalg.norm(f_zero_t0) < 1e-12
    
    A_t0 = np.zeros((n, n))
    b_t0 = f_zero_t0.copy()
    
    if is_homogeneous:
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            A_t0[:, j] = (f(t0, e) - f(t0, -e)) / (2 * h)
        b_t0 = np.zeros(n)
    else:
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            A_t0[:, j] = (f(t0, e) - f(t0, -e)) / (2 * h)

    # Step 2: Check if A is constant
    A_t1 = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        A_t1[:, j] = (f(t1, e) - f(t1, -e)) / (2 * h)
        
    b_t1 = f(t1, np.zeros(n))
    
    norm_A = np.linalg.norm(A_t0)
    tol_A = 1e-6 * max(1.0, norm_A)
    is_A_constant = np.allclose(A_t0, A_t1, rtol=1e-5, atol=tol_A)
    
    is_b_constant = True
    if not is_homogeneous:
        norm_b = np.linalg.norm(b_t0)
        tol_b = 1e-6 * max(1.0, norm_b)
        is_b_constant = np.allclose(b_t0, b_t1, rtol=1e-5, atol=tol_b)

    if is_A_constant and is_b_constant:
        dt = t1 - t0
        try:
            E = scipy.linalg.expm(A_t0 * dt)
            if is_homogeneous:
                y_final = E @ y0
            else:
                try:
                    x = np.linalg.solve(A_t0, b_t0)
                    y_final = E @ y0 + (E - np.eye(n)) @ x
                except np.linalg.LinAlgError:
                    x = np.linalg.lstsq(A_t0, b_t0, rcond=None)[0]
                    y_final = E @ y0 + (E - np.eye(n)) @ x
            return y_final
        except Exception:
            pass

    # Step 3: Time-varying case -> Piecewise Constant Approximation with expm
    # We will divide [t0, t1] into M steps.
    # For each step, we estimate A at the start of the step and use expm(A*dt).
    # This is a first-order accurate method for time-varying A, but very stable.
    
    # Choose M based on a heuristic. If the system is stiff, we might need more steps.
    # Let's start with M=200.
    M = 200
    dt = (t1 - t0) / M
    
    y = y0.copy()
    t = t0
    
    for i in range(M):
        if time.time() > deadline:
            break
        
        # Estimate A at current t
        A_t = np.zeros((n, n))
        b_t = f(t, np.zeros(n))
        is_hom_t = np.linalg.norm(b_t) < 1e-12
        
        if is_hom_t:
            for j in range(n):
                e = np.zeros(n)
                e[j] = h
                A_t[:, j] = (f(t, e) - f(t, -e)) / (2 * h)
            b_t = np.zeros(n)
        else:
            for j in range(n):
                e = np.zeros(n)
                e[j] = h
                A_t[:, j] = (f(t, e) - f(t, -e)) / (2 * h)
        
        # Compute expm(A_t * dt)
        try:
            E = scipy.linalg.expm(A_t * dt)
            if is_hom_t:
                y = E @ y
            else:
                # y' = A y + b
                # Solution: y(t+dt) = expm(A dt) y + (expm(A dt) - I) A^-1 b
                try:
                    x = np.linalg.solve(A_t, b_t)
                    y = E @ y + (E - np.eye(n)) @ x
                except np.linalg.LinAlgError:
                    x = np.linalg.lstsq(A_t, b_t, rcond=None)[0]
                    y = E @ y + (E - np.eye(n)) @ x
        except Exception:
            # Fallback to simple Euler if expm fails (unlikely)
            y = y + dt * f(t, y)
        
        t += dt
    
    return y
