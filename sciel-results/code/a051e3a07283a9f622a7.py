"""
Structural approach: explicit block decomposition using Winograd/Strassen for 2x2x3,
combined with schoolbook for the remaining part, plus a numerical refinement search
for potentially lower ranks.
"""
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

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

    def to_lists(arr):
        """Convert numpy array to list of lists."""
        return arr.tolist()

    # ============================================================
    # Strategy 1: Block decomposition
    # ============================================================
    # Split the 3 middle columns into first 2 and last 1.
    # A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x5) = [B1 (2x5); B2 (1x5)]
    # C = A1 @ B1 + A2 @ B2
    #
    # Part 1: A1 (2x2) @ B1 (2x5) -> use Winograd 2x2x3 for each of 3 column-pairs of B1
    #   B1 has 2 rows and 5 cols. We can pair cols (0,1), (2,3), and handle col 4 separately.
    #   Actually, Winograd 2x2x3 means multiplying a 2x2 by a 2x3.
    #   B1 is 2x5. We can split into 2x2 + 2x2 + 2x1? Or 2x3 + 2x2.
    #
    # Let me think more carefully.
    # A1 is 2x2, B1 is 2x5. The product is 2x5.
    # For the 2x2 x 2x5 part, we can use:
    #   - For each pair of columns (using Winograd 2x2x2 with rank 4? No, 2x2x2 rank is 4 schoolbook, 7 Strassen which is worse)
    #   - Actually for 2x2 x 2x2, schoolbook is rank 4, Strassen is rank 7. So schoolbook is better.
    #   - For 2x2 x 2x1, schoolbook is rank 2.
    #
    # So A1@B1 (2x2 x 2x5): schoolbook = 2*2*5 = 20 terms.
    # A2@B2 (2x1 x 1x5): schoolbook = 2*1*5 = 10 terms.
    # Total schoolbook: 30.
    #
    # But we can do better. Let me use a different split.
    #
    # Alternative: Split B (3x5) into B_top (2x5) and B_bot (1x5).
    # A (2x3) = [A_left (2x2) | A_right (2x1)]
    # C = A_left @ B_top + A_right @ B_bot
    #
    # A_left @ B_top: 2x2 @ 2x5 -> 20 schoolbook
    # A_right @ B_bot: 2x1 @ 1x5 -> 10 schoolbook
    # Total: 30
    #
    # To reduce rank, we need to share computation.
    #
    # Key insight: Use the fact that A_left is 2x2 and we can apply a linear transformation.
    # Winograd's method for 2x2x3 (i.e., 2x2 @ 2x3) achieves rank 16 (vs 12 schoolbook).
    # Wait, 2x2 @ 2x3 schoolbook is 2*2*3=12. Winograd for 2x2x2 is rank 4 (same as schoolbook).
    # For 2x2x3, the rank is 16? No, that seems high. Let me reconsider.
    #
    # Actually, the rank of the 2x2x2 matrix multiplication tensor is 4 (schoolbook is optimal).
    # The rank of 2x2x3 (i.e., multiplying 2x2 by 2x3) - this is a different tensor.
    # T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,1,2}. This is the tensor for A(2x2) * B(2x3) = C(2x3).
    # The number of entries is 2*2*3 = 12, and schoolbook uses 12 multiplications.
    # The rank of this tensor is at most 12. Can we do better?
    # Actually, this is equivalent to 3 separate 2x2x1 multiplications, each of rank 2, total rank 6? No.
    # Wait, A is 2x2, B is 2x3. C[i,k] = sum_j A[i,j]*B[j,k].
    # For fixed k, this is a 2x1 vector = A @ B[:,k], which is a 2x2 x 1x1 multiplication, rank 2.
    # For 3 values of k, if they were independent, rank would be 6. But they share A.
    # The tensor T[i,j,k] = 1 if we're computing A*B where A is 2x2, B is 2x3.
    # This is actually the same as the tensor for (2x2)*(2x3) -> (2x3).
    # The rank of this tensor... Let me think. It's a sub-tensor of the 2x3x3 multiplication tensor.
    #
    # Let me just try to construct an explicit decomposition.
    #
    # Actually, let me try a completely different structural approach:
    # Use the fact that we can absorb linear combinations.
    #
    # For the 2x2 @ 2x5 part:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] for i in {0,1}, k in {0..4}
    # This is 10 output entries, each is a sum of 2 products.
    # Schoolbook: 10 * 2 = 20 terms.
    #
    # Can we share? If we define S0 = A[0,0]+A[0,1], S1 = A[1,0]+A[1,1], etc.
    # This is essentially Strassen-like but for non-square.
    #
    # Let me try: For 2x2 @ 2x2, we know rank is 4 (optimal).
    # For 2x2 @ 2x5, we can view it as 2x2 @ (2x2 | 2x2 | 2x1).
    # If we use schoolbook for each 2x2@2x2 block (4 terms each) and 2x2@2x1 (2 terms):
    # 4 + 4 + 2 = 10 terms? No, that's not right either.
    #
    # Let me just try the numerical search approach but with better structure.
    #
    # Actually, let me try a known result: the rank of M_{2,3,5} is known to be at most 26.
    # Let me try to find it numerically.
    # ============================================================

    def try_find_decomposition(R, time_budget=10.0):
        """Try to find a rank-R decomposition using numerical optimization."""
        if time.time() > deadline:
            return None

        t0 = time.time()
        total_vars = R * (n*m + m*p + p*n)

        # Use multiple restarts with different initializations
        best_err = float('inf')
        best_params = None

        n_restarts = 0
        max_restarts = 30

        while n_restarts < max_restarts:
            if time.time() > deadline or (time.time() - t0) > time_budget:
                break

            n_restarts += 1

            # Random initialization with different scales
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

                # Efficient tensor reconstruction
                # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
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

            # If not converged, try L-BFGS with finite differences (slower but more robust)
            if best_err > 1e-6 and time.time() < deadline:
                try:
                    from scipy.optimize import minimize
                    # Compute gradient numerically for a few iterations
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
        # Try various denominators
        for denom in [1, 2, 4, 8]:
            step = 1.0 / denom
            U_snap = np.round(U_res / step) * step
            V_snap = np.round(V_res / step) * step
            W_snap = np.round(W_res / step) * step

            U_list = to_lists(U_snap)
            V_list = to_lists(V_snap)
            W_list = to_lists(W_snap)

            if verify_exact(U_list, V_list, W_list, R):
                return U_list, V_list, W_list

        # If snapping fails, try a more aggressive rational fitting
        # Try to find rationals p/q with small q that match the values
        from fractions import Fraction
        from math import gcd

        def snap_to_rational(x, max_denom=16):
            """Find best rational approximation of x with denominator <= max_denom."""
            best = Fraction(0)
            best_err = abs(x)
            for q in range(1, max_denom + 1):
                p = round(x * q)
                val = Fraction(p, q)
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
                V_r[i if i < m*p else 0, r] = 0  # placeholder
        for i in range(m*p):
            for r in range(R):
                V_r[i, r] = float(snap_to_rational(V_res[i, r], 16))
        for i in range(p*n):
            for r in range(R):
                W_r[i, r] = float(snap_to_rational(W_res[i, r], 16))

        U_list = to_lists(U_r)
        V_list = to_lists(V_r)
        W_list = to_lists(W_r)
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
