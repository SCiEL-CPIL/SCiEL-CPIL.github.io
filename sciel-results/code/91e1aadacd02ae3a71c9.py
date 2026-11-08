import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 135.0  # 15s safety margin

    # --- Precompute tensor indices ---
    # T[i,j,k, l] where l = 1 if A[i,j]*B[j,k]*W[k*n+i] contributes
    # For each (i,j,k): the triple (u_idx, v_idx, w_idx) = (i*m+j, j*p+k, k*n+i)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all (i,j,k)

    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    T_len = len(triples)  # = n*m*p = 32

    def reconstruct(params, rank):
        U_flat = params[:n * m * rank].reshape(n * m, rank)
        V_flat = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
        W_flat = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
        return U_flat, V_flat, W_flat

    def compute_errors(U, V, W, rank):
        errs = np.empty(T_len)
        for t, (ui, vi, wi) in enumerate(triples):
            errs[t] = np.dot(U[ui] * V[vi], W[wi]) - 1.0
        return errs

    def objective(params):
        U, V, W = reconstruct(params, rank)
        errs = compute_errors(U, V, W, rank)
        return np.sum(errs * errs)

    def check_decomposition(U, V, W, rank):
        # Integrality (half-integer)
        if not np.allclose(2 * U, np.round(2 * U), atol=1e-4): return False
        if not np.allclose(2 * V, np.round(2 * V), atol=1e-4): return False
        if not np.allclose(2 * W, np.round(2 * W), atol=1e-4): return False
        # Correctness
        U_r = np.round(2 * U) / 2.0
        V_r = np.round(2 * V) / 2.0
        W_r = np.round(2 * W) / 2.0
        errs = compute_errors(U_r, V_r, W_r, rank)
        if not np.allclose(errs, 0.0, atol=1e-4): return False
        return True

    def strassen_28():
        """Return a valid rank-28 decomposition via 4x4x4 Strassen tiling."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        # Standard 7-term Strassen for 2x2 * 2x2
        # We tile 4x4 into four 2x2 blocks
        # A is 2x4: rows 0,1; cols 0..3  -> split into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
        # B is 4x4: split into 4 blocks of 2x2: B00, B01, B10, B11
        # C is 2x4: split into C_left (2x2) and C_right (2x2)

        # Strassen terms for 2x2*2x2 -> 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M2+M3-M4+M6
        # C10 = M2+M5
        # C11 = M1-M3+M4+M6

        strassen_defs = [
            # (A_idx_dict, B_idx_dict, C_idx_dict) for 2x2 blocks
            # A indices: 00,01,10,11 ; B indices: 00,01,10,11 ; C indices: 00,01,10,11
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),   # M1
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,0):1}),            # M2
            ({(0,0):1}, {(1,0):1,(1,1):-1}, {(0,1):-1,(1,1):-1}),         # M3
            ({(1,1):1}, {(0,1):1,(1,0):-1}, {(0,0):-1,(1,0):-1}),         # M4  (note: -M4 in C00, +M4 in C11)
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}),           # M5  (-M5 in C00, +M5 in C10)
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(0,1):1,(1,1):1}),   # M6
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),           # M7
        ]

        # For 4x4 tiling: A has 2 row-blocks of 1 row each (n=2, split into 1x1? No, n=2 so A is 2x4)
        # Actually A is 2x4. We split A's columns into two groups of 2: A_L (2x2), A_R (2x2)
        # B is 4x4: split rows into two groups of 2, cols into two groups of 2
        # C is 2x4: split cols into two groups of 2

        # A_L: rows 0,1; cols 0,1  -> A_L[i,j] = A[i, j]
        # A_R: rows 0,1; cols 2,3  -> A_R[i,j] = A[i, j+2]
        # B00: rows 0,1; cols 0,1
        # B01: rows 0,1; cols 2,3
        # B10: rows 2,3; cols 0,1
        # B11: rows 2,3; cols 2,3
        # C_L: rows 0,1; cols 0,1  = A_L @ B_L where B_L = [[B00,B01]]... 

        # Actually for 2x4 * 4x4:
        # C = A * B where A is 2x4, B is 4x4
        # Split B into left (cols 0-1) and right (cols 2-3)
        # C_left = A * B_left (2x4 * 4x2 = 2x2)
        # C_right = A * B_right (2x4 * 4x2 = 2x2)
        # For C_left: A * B_left, split A into A[:, 0:2] and A[:, 2:4], and B_left into B[0:2, :] and B[2:4, :]
        #   C_left = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
        # This is a 2x2 * 2x2 + 2x2 * 2x2 product, which needs 2*7=14 Strassen terms
        # Similarly C_right needs 14 terms. Total = 28.

        # For C_left = A0 * B00 + A1 * B10 where A0=A[:,0:2], A1=A[:,2:4], B00=B[0:2,0:2], B10=B[2:4,0:2]
        # Strassen for A0*B00: 7 terms, for A1*B10: 7 terms
        # C_right = A0 * B01 + A1 * B11

        def apply_strassen_2x2(a_col_base, b_row_base, b_col_base, c_col_base, col_offset):
            """Apply one 2x2*2x2 Strassen with 7 terms starting at col_offset."""
            for t_idx, (a_map, b_map, c_map) in enumerate(strassen_defs):
                col = col_offset + t_idx
                for (ar, ac), coeff in a_map.items():
                    # A row ar (0 or 1), col within 2x2: a_col_base + ac
                    # A is 2x4, so global A index: row ar, col a_col_base + ac
                    u_idx = ar * m + (a_col_base + ac)
                    U[u_idx, col] += coeff
                for (br, bc), coeff in b_map.items():
                    # B row b_row_base + br, col b_col_base + bc
                    b_global_row = b_row_base + br
                    b_global_col = b_col_base + bc
                    v_idx = b_global_row * p + b_global_col
                    V[v_idx, col] += coeff
                for (cr, cc), coeff in c_map.items():
                    # C row cr (0 or 1), col c_col_base + cc
                    # W index: c_col_base + cc is the global col, cr is the row
                    w_idx = (c_col_base + cc) * n + cr
                    W[w_idx, col] += coeff

        # C_left = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
        apply_strassen_2x2(0, 0, 0, 0, 0)    # A[:,0:2] * B[0:2,0:2] -> C[0:2, 0:2]
        apply_strassen_2x2(2, 2, 0, 0, 7)    # A[:,2:4] * B[2:4,0:2] -> C[0:2, 0:2]
        # C_right = A[:,0:2] * B[0:2,2:4] + A[:,2:4] * B[2:4,2:4]
        apply_strassen_2x2(0, 0, 2, 2, 14)   # A[:,0:2] * B[0:2,2:4] -> C[0:2, 2:4]
        apply_strassen_2x2(2, 2, 2, 2, 21)   # A[:,2:4] * B[2:4,2:4] -> C[0:2, 2:4]

        return U, V, W

    # --- Try to find lower-rank decompositions ---
    best_result = None

    # Quick rank search from 20 upward
    for rank in range(20, 28):
        if time.time() > DEADLINE:
            break
        found = False
        n_restarts = 30
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                break
            np.random.seed(hash((rank, restart)) % (2**31))
            params0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.3

            def obj_local(params):
                U, V, W = reconstruct(params, rank)
                errs = compute_errors(U, V, W, rank)
                return np.sum(errs * errs)

            try:
                res = minimize(obj_local, params0, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
                if res.fun < 1e-8:
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    if check_decomposition(U_opt, V_opt, W_opt, rank):
                        best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                        found = True
                        break
            except:
                pass
        if found:
            break

    if best_result is not None:
        return best_result

    # Fallback: rank 28 Strassen
    U, V, W = strassen_28()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
