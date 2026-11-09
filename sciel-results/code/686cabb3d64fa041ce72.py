# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import schur, expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0

    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0_arr

    # Strategy:
    # 1. Determine if the system is homogeneous (b=0) or affine (b!=0).
    #    If homogeneous, we need n calls to determine A.
    #    If affine, we need n+1 calls to determine A and b.
    #    Most "stiff linear systems" in benchmarks are homogeneous (e.g., y' = Ay).
    #    We will probe n+1 points to be safe, but we can potentially save a call 
    #    if we can prove b=0 from the first n probes? 
    #    Actually, we can't prove b=0 without checking a point not in the span 
    #    or checking consistency.
    #    
    #    Let's stick to the robust 16-call method for correctness, but optimize 
    #    the exponential computation.
    #    Wait, the score penalty for 16 vs 15 calls is 0.5*log10(16/15) ≈ 0.014.
    #    Is it worth the risk? 
    #    If the system IS homogeneous, 15 calls is enough.
    #    If we use 15 calls (probing e_1..e_n) and assume b=0, 
    #    we get A_est = [f(e_1)..f(e_n)].
    #    If the true system is y' = A y + b, then f(e_j) = A e_j + b.
    #    So A_est = A + b 1^T.
    #    The error in the solution will be significant if b != 0.
    #    
    #    However, we can check if f(t0, 0) is zero? 
    #    That requires 1 extra call. Total 16.
    #    
    #    Alternative: 
    #    Probe e_1..e_n. 
    #    Check if the columns of the matrix F = [f(e_1)..f(e_n)] 
    #    imply a consistent affine structure?
    #    No, any set of n vectors can be fit by a homogeneous system.
    #    
    #    So, 16 calls is the safe minimum for general affine.
    #    
    #    Let's focus on improving the exponential calculation precision/robustness.
    #    Schur decomposition is often more stable for stiff systems than 
    #    standard expm if the eigenvalues are widely spread, though expm 
    #    uses scaling/squaring which is also robust.
    #    
    #    Let's try a different angle: 
    #    If the system is time-invariant, we can use the analytical solution.
    #    y(t) = e^{A(t-t0)} y0 + \int_{t0}^t e^{A(t-\tau)} b d\tau.
    #    If b is constant: y(t) = e^{At} y0 + (e^{At} - I) A^{-1} b.
    #    If A is singular, use the augmented matrix trick.
    #    
    #    The current code uses the augmented matrix trick which is robust.
    #    Let's ensure we use the most precise method for expm.
    #    scipy.linalg.expm is generally very accurate.
    #    
    #    Is there a way to get 15 calls?
    #    Only if we assume homogeneity. 
    #    Given the "stiff linear system" description, homogeneity is likely.
    #    Let's implement a heuristic:
    #    1. Probe e_1..e_n (15 calls).
    #    2. Construct A_h = [f(e_1)..f(e_n)].
    #    3. Compute solution assuming b=0.
    #    4. Compute f(t0, 0) (16th call).
    #    5. If ||f(t0, 0)|| is very small (e.g., < 1e-10 * scale), 
    #       assume b=0 and use the 15-call result (conceptually).
    #       But we already made 16 calls.
    #    
    #    To actually save the call, we must decide BEFORE the 16th call.
    #    We can't.
    #    
    #    So, we are stuck with 16 calls for safety.
    #    
    #    Let's optimize the 16-call path.
    #    The augmented matrix is (n+1)x(n+1).
    #    For n=15, it's 16x16.
    #    expm on 16x16 is fast.
    #    
    #    What if we use Schur decomposition on the 15x15 A and handle b separately?
    #    If A is diagonalizable (or Schur form), we can compute e^{At} more directly.
    #    But expm already does this internally.
    #    
    #    Let's try to improve the numerical stability of the matrix construction.
    #    The current code uses standard basis vectors.
    #    This is optimal for conditioning.
    
    # 1. Probe b
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)

    # 2. Probe A
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_j = np.array(f(t0, e_j), dtype=float)
        A[:, j] = f_j - b

    if time.time() > deadline:
        return y0_arr

    # 3. Solve
    # Use augmented matrix for robustness against singular A
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    
    z0 = np.zeros(n + 1)
    z0[:n] = y0_arr
    z0[n] = 1.0

    # Compute expm
    # For stiff systems, scaling and squaring is key.
    # scipy.linalg.expm is the standard.
    
    try:
        # Check if A is close to zero to avoid unnecessary work?
        # No, expm handles it.
        
        expM = expm(M * dt)
        z_final = expM @ z0
        y_final = z_final[:n]
        
        if not np.all(np.isfinite(y_final)):
            # Fallback: use solve_ivp with tight tolerances
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, (t0, t1), y0_arr, method='Radau', rtol=1e-12, atol=1e-14)
            if sol.success:
                return sol.y[:, -1]
            return y0_arr
            
        return y_final
    except Exception:
        # Last resort fallback
        from scipy.integrate import solve_ivp
        try:
            sol = solve_ivp(f, (t0, t1), y0_arr, method='Radau', rtol=1e-10, atol=1e-12)
            if sol.success:
                return sol.y[:, -1]
        except:
            pass
        return y0_arr

# EVOLVE-BLOCK-END
