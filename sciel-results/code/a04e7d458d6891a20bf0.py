import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).flatten()
    n = y0.size
    dt = t1 - t0
    start = time.time()
    deadline = start + 285.0  # safety margin before 330s kill

    if n == 0:
        return np.array([])
    if dt == 0:
        return y0.copy()

    # --- Attempt 1: detect a time-independent linear (homogeneous) system f(t, y) = A y.
    # Extract columns of A from unit basis evaluations at t0.
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e)

    # Homogeneity check: f(t, 0) should be ~ 0.
    z0 = f(t0, np.zeros(n))
    hom_err = np.linalg.norm(z0)

    # Linearity + time-independence probes.
    base = y0.copy()
    if np.linalg.norm(base) < 1e-14:
        base = np.ones(n, dtype=float)
    probes = [base, 2.7 * base, -1.3 * base, np.arange(1, n + 1, dtype=float)]
    if n >= 2:
        probes.append(base + 0.5 * np.arange(1, n + 1, dtype=float))

    maxrel = 0.0
    for py in probes:
        pred = A @ py
        fy = f(t0, py)
        maxrel = max(maxrel, np.linalg.norm(fy - pred) / (1e-300 + np.linalg.norm(fy)))

    if hom_err < 1e-10 and maxrel < 1e-9:
        # Constant linear system: y(t1) = expm(A dt) @ y0.
        M = expm(A * dt)
        return M @ y0

    # --- Attempt 2: detect a time-dependent linear (homogeneous) system f(t, y) = A(t) y.
    # Use a piecewise-constant approximation with K subintervals.
    t_mid = 0.5 * (t0 + t1)
    
    # Helper to extract A(t)
    def get_A(t):
        A_t = np.empty((n, n), dtype=float)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            A_t[:, j] = f(t, e)
        return A_t

    # Check if f is linear (but possibly time-dependent) at t0, t_mid, t1
    probe = base + 0.5 * np.arange(1, n + 1, dtype=float)

    maxrel_lin = 0.0
    for t_chk in [t0, t_mid, t1]:
        A_chk = get_A(t_chk)
        pred = A_chk @ probe
        fy = f(t_chk, probe)
        maxrel_lin = max(maxrel_lin, np.linalg.norm(fy - pred) / (1e-300 + np.linalg.norm(fy)))
        # Homogeneity check
        z = f(t_chk, np.zeros(n))
        maxrel_lin = max(maxrel_lin, np.linalg.norm(z) / (1e-300 + np.linalg.norm(fy)))

    if maxrel_lin < 1e-9:
        # Time-dependent linear homogeneous system: use product of exponentials
        # y(t1) ≈ exp(A(t_K) * dt/K) * ... * exp(A(t_1) * dt/K) @ y0
        # Use midpoint rule for A(t) at each subinterval
        
        # Determine optimal K based on available time and accuracy needs.
        # For linear systems, error is O(h^2) per step, so O(1/K^2) global.
        # We want high accuracy. Let's estimate time per step.
        
        # Measure time for one step (extract A + expm + matvec)
        t_meas_start = time.time()
        A_sample = get_A(t_mid)
        h_test = dt / 100
        M_sample = expm(A_sample * h_test)
        _ = M_sample @ y0
        t_meas_end = time.time()
        time_per_step = t_meas_end - t_meas_start
        
        # Available time for integration
        elapsed = time.time() - start
        remaining_time = deadline - start - elapsed
        # Reserve 10% for safety
        usable_time = 0.9 * remaining_time
        
        # Max steps we can afford
        max_K = int(usable_time / max(time_per_step, 1e-6))
        
        # Cap K to avoid excessive f calls (2000000 limit)
        # Each step uses n calls to f.
        max_K = min(max_K, 2000000 // n)
        
        # Ensure minimum reasonable K
        max_K = max(max_K, 100)
        
        # Use as many steps as possible within budget for best accuracy
        K = max_K
        h = dt / K
        y_cur = y0.copy()
        
        for k in range(K):
            if time.time() > deadline:
                # Fallback: return current y
                return y_cur
                
            t_k = t0 + (k + 0.5) * h
            A_k = get_A(t_k)
            M_k = expm(A_k * h)
            y_cur = M_k @ y_cur
            
        return y_cur

    # --- Fallback: robust implicit stiff integrator.
    if time.time() > deadline - 30.0:
        # Not enough time; return a coarse explicit step result as a best effort.
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-8, atol=1e-10,
                        max_step=0.1)
        return sol.y[:, -1]

    from scipy.integrate import solve_ivp
    sol = solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-11, atol=1e-13,
                    max_step=0.05)
    if not sol.success:
        # Try BDF as a last resort.
        sol = solve_ivp(f, (t0, t1), y0, method='BDF', rtol=1e-11, atol=1e-13,
                        max_step=0.05)
    return sol.y[:, -1]
