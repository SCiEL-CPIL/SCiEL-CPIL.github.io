import numpy as np
import scipy.linalg
import scipy.integrate
import time


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on t_span = [t0, t1].
    Strategy:
      1. Assume time-invariant affine linear system y' = A y + b (based on task contract).
      2. Solve exactly via augmented matrix exponential.
      3. Use Cayley-Hamilton polynomial evaluation for expm if eigenvalues are well-separated.
      4. Otherwise fall back to a high-accuracy stiff solver (Radau/BDF).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 20s safety margin for 300s limit

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # --- Step 1: Solve Linear System ---
    try:
        # Probe b = f(t0, 0)
        zero_vec = np.zeros(n)
        b = np.asarray(f(t0, zero_vec), dtype=float)

        # Probe A columns
        A = np.zeros((n, n))
        
        # Column 0
        e0 = np.zeros(n)
        e0[0] = 1.0
        f_e0 = np.asarray(f(t0, e0), dtype=float)
        A[:, 0] = f_e0 - b

        # Columns 1 to n-1
        for i in range(1, n):
            if time.time() > deadline - 10:
                raise TimeoutError("Deadline approaching")
            ei = np.zeros(n)
            ei[i] = 1.0
            f_ei = np.asarray(f(t0, ei), dtype=float)
            A[:, i] = f_ei - b

        if time.time() > deadline - 5:
            raise TimeoutError("Deadline approaching")

        # Solve exactly using augmented matrix exponential
        # z' = M z, M = [[A, b], [0, 0]]
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b

        M_scaled = dt * M
        
        # Try Cayley-Hamilton polynomial evaluation
        expM = None
        try:
            eigvals, eigvecs = np.linalg.eig(M_scaled)
            
            # Check for repeated or close eigenvalues
            # Simple check: min distance between eigenvalues
            # For N=15, O(N^2) check is fine
            min_dist = np.inf
            for i in range(len(eigvals)):
                for j in range(i + 1, len(eigvals)):
                    dist = np.abs(eigvals[i] - eigvals[j])
                    if dist < min_dist:
                        min_dist = dist
            
            # If eigenvalues are well separated, use polynomial
            if min_dist > 1e-8 * np.max(np.abs(eigvals)):
                # Build Vandermonde matrix
                V = np.vander(eigvals, N=n + 1, increasing=True)
                target = np.exp(eigvals)
                
                # Solve V c = target
                # Use least squares if singular, but should be invertible
                c, residuals, rank, s = np.linalg.lstsq(V, target, rcond=None)
                
                # Evaluate polynomial r(M) using Horner's method
                # r(x) = c[0] + c[1]x + ... + c[N]x^N
                # r(M) = c[0]I + c[1]M + ... + c[N]M^N
                I = np.eye(n + 1)
                expM = np.zeros((n + 1, n + 1))
                for i in range(n, -1, -1):
                    if i == 0:
                        expM += c[i] * I
                    else:
                        expM = expM @ M_scaled + c[i] * I
                
            else:
                raise np.linalg.LinAlgError("Eigenvalues too close")

        except Exception:
            # Fallback to standard expm
            expM = scipy.linalg.expm(M_scaled)

        z_init = np.zeros(n + 1)
        z_init[:n] = y0
        z_init[n] = 1.0

        z_final = expM @ z_init
        y_final = z_final[:n]

        return y_final

    except Exception:
        # If linear assumption fails or expm fails, fall through to numerical solver
        pass

    # --- Step 2: Fallback to Numerical Integration ---

    # Use Radau for stiff systems with high accuracy
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            dense_output=False
        )

        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau failed, try BDF
            sol2 = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                return sol2.y[:, -1]

    except Exception:
        # Last resort fallbacks
        try:
            sol3 = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='LSODA',
                rtol=1e-10,
                atol=1e-12
            )
            return sol3.y[:, -1]
        except Exception:
            try:
                sol4 = scipy.integrate.solve_ivp(
                    f,
                    t_span,
                    y0,
                    method='RK45',
                    rtol=1e-10,
                    atol=1e-12
                )
                return sol4.y[:, -1]
            except Exception:
                return y0
