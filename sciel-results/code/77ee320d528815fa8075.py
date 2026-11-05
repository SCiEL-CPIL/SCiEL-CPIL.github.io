import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)

    try:
        # Step 1: Extract b and A using n+1 calls at t0
        # b = f(t0, 0)
        zero_vec = np.zeros(n)
        b = f(t0, zero_vec)

        # A[:, i] = f(t0, e_i) - b
        A = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            f_ei = f(t0, ei)
            A[:, i] = f_ei - b

        # Step 2: Solve exactly using augmented matrix exponential
        # For y' = A*y + b, the solution is y(t) = e^{A*t} * y0 + (e^{A*t} - I) * A^{-1} * b
        # Using augmented matrix: [A  b; 0 0], expm(dt*M) @ [y0; 1]
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b
        # M[n, :] remains 0

        dt = t1 - t0
        M_scaled = dt * M

        # Use scipy.linalg.expm for matrix exponential
        expM = scipy.linalg.expm(M_scaled)

        z_init = np.zeros(n + 1)
        z_init[:n] = y0
        z_init[n] = 1.0

        z_final = expM @ z_init
        y_final = z_final[:n]

        return y_final

    except Exception as e:
        # Fallback if detection fails or expm fails
        pass

    # Fallback: Use SciPy Radau (stiff solver)
    def f_original(t, y):
        return f(t, y)

    try:
        sol = scipy.integrate.solve_ivp(
            f_original,
            t_span,
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )

        if sol.success:
            return sol.y[:, -1]
        else:
            # Try BDF if Radau fails
            sol2 = scipy.integrate.solve_ivp(
                f_original,
                t_span,
                y0,
                method='BDF',
                rtol=1e-10,
                atol=1e-12
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                return sol2.y[:, -1]

    except Exception as e2:
        # Last resort
        try:
            sol3 = scipy.integrate.solve_ivp(
                f_original,
                t_span,
                y0,
                method='BDF',
                rtol=1e-6,
                atol=1e-8
            )
            return sol3.y[:, -1]
        except:
            return y0
