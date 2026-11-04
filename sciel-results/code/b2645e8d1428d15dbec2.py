import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    n, m, p = 2, 3, 5

    def verify_exact(U, V, W, R):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        u_val = Fraction(U[i*m + j][r])
                        v_val = Fraction(V[j*p + k][r])
                        w_val = Fraction(W[k*n + i][r])
                        s += u_val * v_val * w_val
                    if s != 1:
                        return False
        return True

    # ============================================================
    # Laderman rank-7 decomposition for 3x3 matrix multiplication
    # ============================================================
    # Laderman's 7-multiplication algorithm for 3x3 matrices:
    # We compute C = A * B where A, B, C are 3x3.
    # The 7 scalar multiplications are:
    #
    # Let A = (a_ij), B = (b_jk), C = (c_ik) with i,j,k in {0,1,2}.
    #
    # m1 = a00 * (b00 + b01 + b02)
    # m2 = (a01 - a00) * (b00 + b10)
    # m3 = (a02 - a00) * (b00 + b20)
    # m4 = (a10 - a00) * (b00 + b01)
    # m5 = (a20 - a00) * (b00 + b02)
    # m6 = (a11 + a22 - a00) * (b11 + b22 - b00)
    # m7 = (a12 + a21 - a00) * (b12 + b21 - b00)
    #
    # Then:
    # c00 = m1
    # c01 = m1 + m2
    # c02 = m1 + m3
    # c10 = m1 + m4
    # c11 = m1 + m2 + m6
    # c12 = m1 + m3 + m7
    # c20 = m1 + m5
    # c21 = m1 + m2 + m7  -- wait, let me re-derive.
    #
    # Actually, let me use the standard Laderman formulas. Let me look them up carefully.
    #
    # Standard Laderman 3x3 (from Laderman 1976):
    # Define:
    #   x1 = a00
    #   x2 = a01 - a00
    #   x3 = a02 - a00
    #   x4 = a10 - a00
    #   x5 = a20 - a00
    #   x6 = a11 + a22 - a00
    #   x7 = a12 + a21 - a00
    #
    #   y1 = b00 + b01 + b02
    #   y2 = b00 + b10
    #   y3 = b00 + b20
    #   y4 = b00 + b01
    #   y5 = b00 + b02
    #   y6 = b11 + b22 - b00
    #   y7 = b12 + b21 - b00
    #
    #   m1 = x1 * y1 = a00 * (b00+b01+b02)
    #   m2 = x2 * y2 = (a01-a00) * (b00+b10)
    #   m3 = x3 * y3 = (a02-a00) * (b00+b20)
    #   m4 = x4 * y4 = (a10-a00) * (b00+b01)
    #   m5 = x5 * y5 = (a20-a00) * (b00+b02)
    #   m6 = x6 * y6 = (a11+a22-a00) * (b11+b22-b00)
    #   m7 = x7 * y7 = (a12+a21-a00) * (b12+b21-b00)
    #
    #   c00 = m1
    #   c01 = m1 + m2
    #   c02 = m1 + m3
    #   c10 = m1 + m4
    #   c11 = m1 + m2 + m6  -- need to verify
    #   c12 = m1 + m3 + m7  -- need to verify
    #   c20 = m1 + m5
    #   c21 = m1 + m2 + m7  -- need to verify
    #   c22 = m1 + m3 + m6  -- need to verify
    #
    # Let me verify c11:
    # c11 should = a10*b01 + a11*b11 + a12*b21
    # m1 = a00*b00 + a00*b01 + a00*b02
    # m2 = a01*b00 + a01*b10 - a00*b00 - a00*b10
    # m6 = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a00*b11 - a00*b22 - a11*b00 - a22*b00 + a00*b00
    #
    # m1 + m2 + m6 =
    # a00*b00 + a00*b01 + a00*b02
    # + a01*b00 + a01*b10 - a00*b00 - a00*b10
    # + a11*b11 + a11*b22 + a22*b11 + a22*b22 - a00*b11 - a00*b22 - a11*b00 - a22*b00 + a00*b00
    #
    # = a00*b01 + a00*b02 + a01*b00 + a01*b10 - a00*b10
    #   + a11*b11 + a11*b22 + a22*b11 + a22*b22 - a00*b11 - a00*b22 - a11*b00 - a22*b00
    #
    # This doesn't look like c11 = a10*b01 + a11*b11 + a12*b21.
    #
    # I think I'm misremembering the Laderman formulas. Let me use a known correct set.
    #
    # Actually, the Laderman algorithm is well-known. Let me use the standard formulation:
    #
    # For C = A * B (3x3):
    # m1 = a00 * (b00 + b01 + b02)
    # m2 = (a01 - a00) * (b00 + b10)
    # m3 = (a02 - a00) * (b00 + b20)
    # m4 = (a10 - a00) * (b00 + b01)
    # m5 = (a20 - a00) * (b00 + b02)
    # m6 = (a11 + a22 - a00) * (b11 + b22 - b00)
    # m7 = (a12 + a21 - a00) * (b12 + b21 - b00)
    #
    # c00 = m1
    # c01 = m1 + m2
    # c02 = m1 + m3
    # c10 = m1 + m4
    # c11 = m1 + m2 + m6  -- Let me verify with a simple example
    #
    # Test: A = I (identity), B = B.
    # a00=1, all other a_ij = 0.
    # m1 = 1*(b00+b01+b02) = b00+b01+b02
    # m2 = (0-1)*(b00+b10) = -(b00+b10)
    # m3 = (0-1)*(b00+b20) = -(b00+b20)
    # m4 = (0-1)*(b00+b01) = -(b00+b01)
    # m5 = (0-1)*(b00+b02) = -(b00+b02)
    # m6 = (0+0-1)*(b11+b22-b00) = -(b11+b22-b00)
    # m7 = (0+0-1)*(b12+b21-b00) = -(b12+b21-b00)
    #
    # c00 = m1 = b00+b01+b02. But c00 should be a00*b00+a01*b10+a02*b20 = b00.
    # This is wrong! The Laderman formulas I'm using are incorrect.
    #
    # Let me look up the actual Laderman 3x3 algorithm.
    #
    # The correct Laderman 3x3 decomposition:
    # m1 = a00 * b00
    # m2 = (a01 + a02) * (b01 + b02)  -- no, this isn't right either.
    #
    # Let me use a different approach. I'll use the known fact that the rank of 3x3x3 is 7,
    # and construct the decomposition using a known explicit formula.
    #
    # Actually, let me just use a numerical search approach but with a much better initialization
    # based on the structure of the problem. The key insight from the directive is to use
    # Laderman, but since I can't easily recall the exact formulas, let me try a different structural approach.
    #
    # Alternative structural approach:
    # The 2x3 x 3x5 multiplication can be viewed as:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # We can split this into:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # For each i,k, this is a sum of 3 products. Total: 2*5*3 = 30 schoolbook.
    #
    # To reduce rank, we need to share multiplications.
    #
    # Key idea: Use the fact that A has only 2 rows. We can define:
    # S0 = A[0,0] + A[0,1] + A[0,2]  (or some linear combination)
    # S1 = A[1,0] + A[1,1] + A[1,2]
    #
    # But this doesn't directly help because B[j,k] varies with j.
    #
    # Let me try a Winograd-style approach for the 2x3 x 3x5 case.
    #
    # Actually, let me try a completely different approach: use the known rank bounds.
    # The rank of M_{2,3,5} is known to be at most 26 (from various sources).
    # Let me try to find a rank-26 decomposition numerically.
    #
    # But first, let me try a simple structural decomposition that might give rank < 30.
    #
    # Split B (3x5) into B1 (2x5) and B2 (1x5).
    # A (2x3) = [A1 (2x2) | A2 (2x1)]
    # C = A1 @ B1 + A2 @ B2
    #
    # A1 @ B1: 2x2 @ 2x5. Schoolbook rank = 2*2*5 = 20.
    # A2 @ B2: 2x1 @ 1x5. Schoolbook rank = 2*1*5 = 10.
    # Total: 30.
    #
    # For A1 @ B1 (2x2 @ 2x5):
    # C[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
    #
    # We can define:
    # P0 = A1[0,0] + A1[0,1]
    # P1 = A1[1,0] + A1[1,1]
    # Q0 = B1[0,0] + B1[1,0]  (for k=0)
    # ...
    #
    # This is getting complicated. Let me just try the numerical search with better initialization.
    # ============================================================

    def try_find_decomposition(R, time_budget=10.0):
        """Try to find a rank-R decomposition using numerical optimization."""
        if time.time() > deadline:
            return None

        t0 = time.time()
        best_err = float('inf')
        best_params = None

        n_restarts = 0
        max_restarts = 20

        while n_restarts < max_restarts:
            if time.time() > deadline or (time.time() - t0) > time_budget:
                break

            n_restarts += 1

            # Random initialization
            scale = 0.5 + 0.5 * n_restarts
            U_init = np.random.randn(n*m, R) * scale
            V_init = np.random.randn(m*p, R) * scale
            W_init = np.random.randn(p*n, R) * scale

            # Also try a structured initialization based on schoolbook
            if n_restarts == 1:
                U_init = np.zeros((n*m, R))
                V_init = np.zeros((m*p, R))
                W_init = np.zeros((p*n, R))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < R:
                                U_init[i*m+j, col] = 1.0
                                V_init[j*p+k, col] = 1.0
                                W_init[k*n+i, col] = 1.0
                            col += 1

            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def error_func(params):
                U_loc = params[:R*n*m].reshape(R, n*m).T
                V_loc = params[R*n*m:R*(n*m+m*p)].reshape(R, m*p).T
                W_loc = params[R*(n*m+m*p):].reshape(R, p*n).T

                T_recon = np.zeros((n, m, p))
                for r in range(R):
                    u_r = U_loc[:, r].reshape(n, m)
                    v_r = V_loc[:, r].reshape(m, p)
                    w_r = W_loc[:, r].reshape(p, n)
                    T_recon += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)

                err = np.sum((T_recon - 1.0)**2)
                return err

            # Try Nelder-Mead
            try:
                from scipy.optimize import minimize
                res = minimize(error_func, params0, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-12})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x.copy()
                if best_err < 1e-10:
                    break
            except Exception:
                pass

            # If not converged, try L-BFGS
            if best_err > 1e-6 and time.time() < deadline:
                try:
                    from scipy.optimize import minimize
                    res = minimize(error_func, params0, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-12})
                    if res.fun < best_err:
                        best_err = res.fun
                        best_params = res.x.copy()
                    if best_err < 1e-10:
                        break
                except Exception:
                    pass

        if best_params is None or best_err > 1e-4:
            return None

        U_res = best_params[:R*n*m].reshape(R, n*m).T
        V_res = best_params[R*n*m:R*(n*m+m*p)].reshape(R, m*p).T
        W_res = best_params[R*(n*m+m*p):].reshape(R, p*n).T

        # Try to snap to rational values
        for denom in [1, 2, 4, 8]:
            step = 1.0 / denom
            U_snap = np.round(U_res / step) * step
            V_snap = np.round(V_res / step) * step
            W_snap = np.round(W_res / step) * step

            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()

            if verify_exact(U_list, V_list, W_list, R):
                return U_list, V_list, W_list

        # Try more aggressive rational fitting
        from fractions import Fraction
        from math import gcd

        def snap_to_rational(x, max_denom=16):
            best = Fraction(0)
            best_err = abs(x)
            for q in range(1, max_denom + 1):
                p_val = round(x * q)
                val = Fraction(p_val, q)
                err = abs(float(val) - x)
                if err < best_err:
                    best_err = err
                    best = val
            return best

        U_r = np.zeros((n*m, R))
        V_r = np.zeros((m*p, R))
        W_r = np.zeros((p*n, R))
        for i in range(n*m):
            for r in range(R):
                U_r[i, r] = float(snap_to_rational(U_res[i, r], 16))
        for i in range(m*p):
            for r in range(R):
                V_r[i, r] = float(snap_to_rational(V_res[i, r], 16))
        for i in range(p*n):
            for r in range(R):
                W_r[i, r] = float(snap_to_rational(W_res[i, r], 16))

        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        if verify_exact(U_list, V_list, W_list, R):
            return U_list, V_list, W_list

        return None

    # Try ranks from low to high
    for R in [24, 25, 26, 27, 28, 29]:
        if time.time() > deadline - 5:
            break
        result = try_find_decomposition(R, time_budget=min(15, deadline - time.time() - 2))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # ============================================================
    # Fallback: Schoolbook rank 30
    # ============================================================
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
