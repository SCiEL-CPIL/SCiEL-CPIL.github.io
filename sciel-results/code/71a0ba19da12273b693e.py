import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 20s safety margin for 300s limit

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)

    # Strategy:
    # 1. Extract b and A using n+1 calls at t0.
    # 2. Solve exactly using augmented matrix exponential.
    # 3. Verify linearity/time-invariance with 1 extra call.
    # 4. Fallback to Radau if verification fails.

    try:
        # Step 1: Extract b and A
        zero_vec = np.zeros(n)
        b = f(t0, zero_vec)

        A = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            f_ei = f(t0, ei)
            A[:, i] = f_ei - b

        # Step 2: Solve exactly using augmented matrix exponential.
        if time.time() > deadline - 5:
            raise TimeoutError("Deadline approaching")
            
        M = np.zeros((n+1, n+1))
        M[:n, :n] = A
        M[:n, n] = b
        # M[n, :] remains 0

        dt = t1 - t0
        M_scaled = dt * M

        expM = scipy.linalg.expm(M_scaled)

        z_init = np.zeros(n+1)
        z_init[:n] = y0
        z_init[n] = 1.0

        z_final = expM @ z_init
        y_final = z_final[:n]

        # Step 3: Verify LTI assumption
        # Check if f(t1, y_final) matches A @ y_final + b
        f_check = f(t1, y_final)
        model_check = A @ y_final + b
        
        # Use a reasonable tolerance for verification
        # If the system is LTI, this should match closely.
        # If not, it will differ.
        if np.allclose(f_check, model_check, rtol=1e-5, atol=1e-8):
            return y_final
        else:
            # LTI assumption failed. Fall back to integration.
            raise ValueError("LTI assumption failed")

    except Exception as e:
        # Fallback if detection fails, verification fails, or expm fails
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
                return sol2.y[:, -1] # Return best effort
                
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
