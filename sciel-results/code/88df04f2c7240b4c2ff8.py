import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W, rank

    def try_rank_sub(n, m, p, rank, max_iters=200, trials=5):
        """Try to find a decomposition of given rank for n x m x p."""
        for trial in range(trials):
            if time.time() > deadline - 10:
                return None
                
            np.random.seed(2000 + trial * 137)
            # Initialize W with sparse ±1 patterns
            W = np.zeros((p*n, rank))
            for i in range(p*n):
                # Each row of W gets a few non-zero entries
                nz = np.random.choice(rank, size=min(3, rank), replace=False)
                for idx in nz:
                    W[i][idx] = np.random.choice([-1.0, 1.0])
            
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            
            for iteration in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Solve for V given U, W
                A_v = np.zeros((n*m*p, m*p*rank))
                b_v = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n+i][c]
                                if abs(w_val) > 1e-12:
                                    v_idx = (j*p + k) * rank + c
                                    A_v[eq_idx, v_idx] = U[i*m+j][c] * w_val
                            eq_idx += 1
                V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_sol.reshape(m*p, rank)
                
                # Solve for U given V, W
                A_u = np.zeros((n*m*p, n*m*rank))
                b_u = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n+i][c]
                                if abs(w_val) > 1e-12:
                                    u_idx = (i*m + j) * rank + c
                                    A_u[eq_idx, u_idx] = V[j*p+k][c] * w_val
                            eq_idx += 1
                U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_sol.reshape(n*m, rank)
                
                # Check error periodically
                if iteration % 10 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(rank):
                                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-6:
                        # Try rounding to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, n, m, p):
                            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
                        # Also try integer rounding
                        U_snap2 = np.round(U)
                        V_snap2 = np.round(V)
                        W_snap2 = np.round(W)
                        if verify_decomposition(U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist(), rank, n, m, p):
                            return U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist()
                            
        return None

    def combine_two_sub(U_sub, V_sub, W_sub, rank_sub):
        """Combine two 2x3x3 decompositions for 2x3x6."""
        # U is 2x3 (6 rows) x (2*rank_sub) cols
        U_full = []
        for row in range(6):
            U_full.append(U_sub[row] + U_sub[row])
            
        # V is 3x6 (18 rows) x (2*rank_sub) cols
        V_full = []
        for j in range(3):
            for k in range(3):
                row_idx = j * 3 + k
                V_full.append(V_sub[row_idx] + [0]*rank_sub)
            for k in range(3):
                row_idx = j * 3 + k
                V_full.append([0]*rank_sub + V_sub[row_idx])
                
        # W is 6x2 (12 rows) x (2*rank_sub) cols
        W_full = []
        for k in range(3):
            for i in range(2):
                row_idx = k * 2 + i
                W_full.append(W_sub[row_idx] + [0]*rank_sub)
        for k in range(3):
            for i in range(2):
                row_idx = k * 2 + i
                W_full.append([0]*rank_sub + W_sub[row_idx])
                
        return U_full, V_full, W_full

    # Try to find 2x3x3 decomposition of rank 10
    sub_res = try_rank_sub(2, 3, 3, 10)
    if sub_res:
        U_sub, V_sub, W_sub = sub_res
        rank_sub = 10
        U_full, V_full, W_full = combine_two_sub(U_sub, V_sub, W_sub, rank_sub)
        if verify_decomposition(U_full, V_full, W_full, 20, 2, 3, 6):
            return {"U": U_full, "V": V_full, "W": W_full}

    # Try rank 9 for 2x3x3
    sub_res9 = try_rank_sub(2, 3, 3, 9)
    if sub_res9:
        U_sub9, V_sub9, W_sub9 = sub_res9
        rank_sub9 = 9
        U_full9, V_full9, W_full9 = combine_two_sub(U_sub9, V_sub9, W_sub9, rank_sub9)
        if verify_decomposition(U_full9, V_full9, W_full9, 18, 2, 3, 6):
            return {"U": U_full9, "V": V_full9, "W": W_full9}

    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook(2, 3, 6)
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
