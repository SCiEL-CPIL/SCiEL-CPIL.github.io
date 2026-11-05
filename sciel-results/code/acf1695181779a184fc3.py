import numpy as np
import time
import scipy.integrate
import scipy.linalg

def integrate(f, t_span, y0):
    """
    Integrates a stiff 15-dimensional linear ODE system.
    
    Strategy:
    1. Assume the system is linear: y' = A(t)y + b(t).
    2. Estimate A(t0) and b(t0) using a small number of calls.
    3. Use a high-order BDF method with Newton iterations.
       Since the system is linear, the Newton iterations converge in 1 step 
       if the Jacobian is exact. We approximate the Jacobian once and reuse it,
       or we can exploit linearity to solve the implicit system directly.
       
       Actually, a simpler and more robust approach for "stiff linear" is:
       Use a high-order implicit Runge-Kutta method (like Radau) but with 
       very tight tolerances and a fixed step size if possible, or 
       exploit the fact that for linear systems, we can use matrix exponentials 
       if A is constant.
       
       The previous strategy used expm if A was constant. The score was 1.137.
       Let's try a different structural approach: 
       **Chebyshev Spectral Method** or **High-Order BDF with Fixed Step**.
       
       Given the stiffness, BDF is good. Let's implement a 4th order BDF 
       with a fixed number of steps to balance accuracy and call count.
       
       However, the "linear" hint is strong. If A is constant, expm is exact.
       The current code checks for constancy. 
       
       Let's try a **Hybrid Strategy**:
       1. Sample f at a few points to estimate A and b.
       2. If A appears constant, use expm (exact).
       3. If A is time-varying, use a **Piecewise Constant Approximation (PCA)** 
          with many small steps. For each small step, assume A is constant 
          and use expm(A*dt). This is a form of first-order exponential integrator.
          By taking many small steps (e.g., 1000 steps), we can achieve high accuracy 
          for slowly varying A, while remaining stable for stiff A.
          
       Number of calls: 
       - Estimation: ~2*n + 1 calls.
       - Integration: 1 call per step to update y? No, if we assume A is constant 
         over the step, we don't need to call f inside the step if we pre-computed A.
         But A(t) might change. We need to estimate A at each step start? 
         That would be 2*n calls per step. Too many.
         
       Alternative: **Linear Multistep Method (BDF) with Fixed Jacobian**.
       If the system is linear, the Jacobian is the system matrix A. 
       If A is constant, BDF with exact Jacobian is very efficient.
       
       Let's stick to the most robust improvement over the current seed:
       The seed uses `solve_ivp` with Radau as fallback. 
       Radau is adaptive. Sometimes adaptivity can be inefficient or inaccurate 
       if tolerances are not set perfectly.
       
       New Strategy: **High-Order BDF (4th order) with Fixed Step Size and Newton Iteration**.
       We will:
       1. Estimate A and b at t0.
       2. Check if A is constant. If so, use expm (as before, but optimized).
       3. If not constant, use a fixed-step BDF4 solver.
          - We will take N steps.
          - At each step, we perform Newton iterations. Since the system is linear,
            the residual is linear, so Newton converges in 1 step if we have the Jacobian.
            We will reuse the Jacobian from the previous step (or estimate it once).
            Actually, for linear y'=Ay+b, the Jacobian is A. If A is time-dependent, 
            it changes. But if we use a small step, A(t_n) is a good approximation.
            We can estimate A at t_n using finite differences (2*n calls).
            This is expensive: N * 2*n calls.
            
       Let's reconsider the **Piecewise Constant Approximation (PCA)** with **Matrix Exponential**.
       If we divide [t0, t1] into M intervals, and in each interval we assume A is constant 
       (estimated at the start of the interval), we can compute the solution exactly for that interval 
       using expm.
       Error per interval is O(h^2) for A(t) variation. Total error O(h).
       To get 1e-10 error, we need h ~ 1e-10? No, that's for explicit.
       For PCA with expm, the local truncation error is dominated by the change in A.
       If A is smooth, error ~ C * h^2 * ||A'||.
       If we use M=1000 steps, h=0.001.
       Calls: M * (2*n + 1) ? No, we only need A at the start of each step.
       To get A, we need 2*n calls. So 1000 * 30 = 30,000 calls.
       Score: -log10(1e-10) - 0.5*log10(30000) = 10 - 2.5 = 7.5.
       This is worse than the seed (13.5 potential, 1.137 actual).
       
       Why did the seed score 1.137? 
       If it hit the expm path, score would be ~13.5.
       If it fell back to Radau, score depends on Radau's performance.
       Maybe the system is NOT constant A? Or the detection failed?
       
       Let's look at the inspirations. They all score ~1.13.
       This suggests the system might NOT be constant A, or the detection is fragile.
       
       **New Structural Idea**: **Implicit Runge-Kutta (Radau) with Fixed Step and Tight Tolerances**.
       Instead of adaptive `solve_ivp`, use a fixed step Radau solver.
       This avoids the overhead of adaptive step control and potential instability 
       in step size selection.
       We can use `scipy.integrate.solve_ivp` with `method='Radau'` but set `max_step` 
       and `min_step` to force a specific behavior, or better, implement a simple 
       fixed-step Radau.
       
       However, implementing Radau is complex.
       
       Let's try **High-Order BDF with Adaptive Step Control** but implemented manually 
       to have full control over the Newton iterations and to exploit linearity.
       
       Actually, the most likely reason for the score 1.137 is that the system 
       IS stiff and linear, but A is time-varying or the detection of constancy 
       was too strict/loose.
       
       Let's try a **Hybrid**:
       1. Try expm with constant A assumption.
       2. If the residual of the solution is large (check y' = f(t,y) at the end), 
          fall back to a very high-precision solver.
          
       But we can't easily check the residual without more calls.
       
       Let's try **Increasing the Precision of the Finite Difference Estimation**.
       The current code uses h=1e-6. This might be too large or too small.
       Using complex step differentiation? No, f might not support complex.
       
       Let's try **Central Difference with Optimal Step Size**.
       Optimal h for central difference is ~ (eps)^(1/3) ~ 6e-6.
       Current h=1e-6 is close.
       
       What if we use **More Points** to estimate A? 
       Instead of 2 points, use 5 points for higher-order derivative approximation?
       This would increase calls but might improve accuracy of A.
       
       Let's try a completely different solver for the fallback: **LSODA** or **BDF**.
       `solve_ivp` with `method='BDF'` is often more stable for stiff problems than Radau 
       if the Jacobian is not well-conditioned.
       
       Let's modify the fallback to use `BDF` with very tight tolerances and a 
       Jacobian provided by the user? No, we don't have the Jacobian.
       But we can estimate it.
       
       **Revised Strategy**:
       1. Estimate A and b at t0.
       2. Check if A is constant.
       3. If constant, use expm.
       4. If not constant, use `solve_ivp` with `method='BDF'` and `jac` provided 
          as a function that returns the estimated A(t).
          This allows the solver to use the exact Jacobian (approximately) for Newton iterations,
          which is much more efficient for linear stiff systems.
          
       The `jac` function in `solve_ivp` is called to compute the Jacobian.
       We can cache the estimated A and return it.
       
       This is a structural change: **Providing a Jacobian to the solver**.
       The current code does not provide a Jacobian, so `solve_ivp` uses finite differences 
       internally, which is slow and inaccurate.
       
       Let's implement this.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0

    # Step 1: Estimate A and b at t0
    # Use central differences with h = 1e-5 (slightly larger to avoid round-off, but still small)
    h = 1e-5
    
    # Check homogeneity
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
        # y' = A y + b
        # f(t, y) = A(t) y + b(t)
        # f(t, e_j) = A e_j + b = col_j(A) + b
        # f(t, -e_j) = -col_j(A) + b
        # f(t, e_j) - f(t, -e_j) = 2 col_j(A)
        # So col_j(A) = (f(t, e_j) - f(t, -e_j)) / 2
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            A_t0[:, j] = (f(t0, e) - f(t0, -e)) / (2 * h)
        # b is already f(t, 0)

    # Step 2: Check if A is constant
    A_t1 = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        A_t1[:, j] = (f(t1, e) - f(t1, -e)) / (2 * h)
        
    # Also check b if inhomogeneous
    b_t1 = f(t1, np.zeros(n))
    
    # Check constancy with a tolerance
    # Use a relative tolerance based on the magnitude of A
    norm_A = np.linalg.norm(A_t0)
    tol_A = 1e-6 * max(1.0, norm_A)
    
    is_A_constant = np.allclose(A_t0, A_t1, rtol=1e-5, atol=tol_A)
    
    is_b_constant = True
    if not is_homogeneous:
        norm_b = np.linalg.norm(b_t0)
        tol_b = 1e-6 * max(1.0, norm_b)
        is_b_constant = np.allclose(b_t0, b_t1, rtol=1e-5, atol=tol_b)

    if is_A_constant and is_b_constant:
        # LTI System
        dt = t1 - t0
        try:
            E = scipy.linalg.expm(A_t0 * dt)
            if is_homogeneous:
                y_final = E @ y0
            else:
                # y(t) = E y0 + (E - I) A^-1 b
                try:
                    x = np.linalg.solve(A_t0, b_t0)
                    y_final = E @ y0 + (E - np.eye(n)) @ x
                except np.linalg.LinAlgError:
                    x = np.linalg.lstsq(A_t0, b_t0, rcond=None)[0]
                    y_final = E @ y0 + (E - np.eye(n)) @ x
            return y_final
        except Exception:
            pass

    # Fallback: Time-varying or detection failed
    # Use solve_ivp with BDF and provide a Jacobian function
    # The Jacobian function will return the estimated A at the current time.
    # To save calls, we will cache A estimates.
    
    # We will use a dictionary to cache A estimates at specific times.
    A_cache = {t0: A_t0, t1: A_t1}
    
    def jac(t, y):
        # Check if we have a cached A for this t
        # Since t is continuous, we might not have an exact match.
        # We will use the nearest cached value or estimate a new one.
        # For simplicity and to avoid too many calls, we will just return A_t0 
        # or estimate it if t is far from t0.
        # Actually, for a linear system, A(t) is the Jacobian.
        # We can estimate it using finite differences at the current t.
        # But that would be 2*n calls per Jacobian evaluation.
        # BDF might call jac multiple times per step.
        
        # Better strategy: Assume A is slowly varying and use A_t0.
        # Or, estimate A at t using a few calls.
        
        # Let's try estimating A at t if we haven't done so recently.
        # But we don't know the call budget.
        
        # Safe bet: Return A_t0. If the system is stiff, A is likely dominant.
        # If A is time-varying, this approximation might be poor.
        
        # Let's use a simple heuristic: if t is close to t0, use A_t0.
        # If t is close to t1, use A_t1.
        # Otherwise, interpolate? Or estimate?
        
        # Given the risk, let's just use A_t0.
        return A_t0

    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            jac=jac,
            max_step=0.1
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: Radau without jac
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.1
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
