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

    def try_rank_direct(n, m, p, rank, max_trials=20, max_iters=100):
        """Try to find a decomposition of given rank for n x m x p directly."""
        n_eq = n * m * p
        n_u = n * m
        n_v = m * p
        n_w = p * n
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(42 + trial * 17)
            
            # Initialize W with sparse random +/-1
            W = np.zeros((n_w, rank))
            for r in range(n_w):
                nz_count = min(rank, 2 + np.random.randint(2))
                nz_idx = np.random.choice(rank, size=nz_count, replace=False)
                for idx in nz_idx:
                    W[r][idx] = np.random.choice([-1.0, 1.0])
            
            # Initialize U and V with small integers
            U = np.random.randint(-1, 2, size=(n_u, rank)).astype(float)
            V = np.random.randint(-1, 2, size=(n_v, rank)).astype(float)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for V
                A_v = np.zeros((n_eq, n_v * rank))
                b_v = np.ones(n_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n + i][c]
                                if w_val != 0:
                                    u_val = U[i*m + j][c]
                                    if u_val != 0:
                                        v_idx = (j*p + k) * rank + c
                                        A_v[eq_idx, v_idx] = u_val * w_val
                            eq_idx += 1
                            
                V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_sol.reshape(n_v, rank)
                
                # Solve for U
                A_u = np.zeros((n_eq, n_u * rank))
                b_u = np.ones(n_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n + i][c]
                                if w_val != 0:
                                    v_val = V[j*p + k][c]
                                    if v_val != 0:
                                        u_idx = (i*m + j) * rank + c
                                        A_u[eq_idx, u_idx] = v_val * w_val
                            eq_idx += 1
                            
                U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_sol.reshape(n_u, rank)
                
                # Check convergence
                if it % 5 == 0:
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
                        
                        # Try integer rounding
                        U_snap2 = np.round(U)
                        V_snap2 = np.round(V)
                        W_snap2 = np.round(W)
                        if verify_decomposition(U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist(), rank, n, m, p):
                            return U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist()
                            
        return None

    # Try ranks from low to high
    # Schoolbook is 36.
    # Try 26, 28, 30, 32, 34, 35
    for r in [26, 28, 30, 32, 34, 35]:
        if time.time() > deadline - 10:
            break
        res = try_rank_direct(2, 3, 6, r, max_trials=10, max_iters=50)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook(2, 3, 6)
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
