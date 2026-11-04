"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-8):
    T = _build_tensor(n, m, p)
    T_d = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(T_d, T, atol=tol, rtol=0)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115
    
    U, V, W = _schoolbook(n, m, p)
    best = (U, V, W)
    best_rank = U.shape[1]
    
    # Try to reduce rank via ALS from the schoolbook init
    # Merge pairs of columns that are similar
    T = _build_tensor(n, m, p)
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    # Try various target ranks with ALS starting from schoolbook
    for target_rank in range(110, 27, -1):
        if time.time() > deadline - 5:
            break
        
        # Initialize from schoolbook by averaging random pairs
        rng = np.random.default_rng(target_rank)
        # Create mapping from original columns to new columns
        orig_cols = list(range(112))
        new_col_map = {}
        for oc in orig_cols:
            new_col_map[oc] = rng.integers(0, target_rank)
        
        U_init = np.zeros((dim_a, target_rank))
        V_init = np.zeros((dim_b, target_rank))
        W_init = np.zeros((dim_c, target_rank))
        for oc in range(112):
            nc = new_col_map[oc]
            i = oc // (m * p)
            j = (oc // p) % m
            k = oc % p
            U_init[i*m + j, nc] += 1.0
            V_init[j*p + k, nc] += 1.0
            W_init[k*n + i, nc] += 1.0
        
        # ALS refinement
        U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
        for it in range(200):
            if time.time() > deadline - 5:
                break
            # Update U
            T_d = np.einsum('ac,bc,dc->abd', U_cur, V_cur, W_cur)
            R = T - T_d
            for c in range(target_rank):
                P = np.einsum('abd,bc,dc->a', R, V_cur[:, c], W_cur[:, c])
                Q = np.sum((V_cur[:, c] * W_cur[:, c]) ** 2)
                if Q > 1e-12:
                    U_cur[:, c] = P / Q
            # Update V
            T_d = np.einsum('ac,bc,dc->abd', U_cur, V_cur, W_cur)
            R = T - T_d
            for c in range(target_rank):
                P = np.einsum('abd,ac,dc->b', R, U_cur[:, c], W_cur[:, c])
                Q = np.sum((U_cur[:, c] * W_cur[:, c]) ** 2)
                if Q > 1e-12:
                    V_cur[:, c] = P / Q
            # Update W
            T_d = np.einsum('ac,bc,dc->abd', U_cur, V_cur, W_cur)
            R = T - T_d
            for c in range(target_rank):
                P = np.einsum('abd,ac,bc->d', R, U_cur[:, c], V_cur[:, c])
                Q = np.sum((U_cur[:, c] * V_cur[:, c]) ** 2)
                if Q > 1e-12:
                    W_cur[:, c] = P / Q
            
            T_d = np.einsum('ac,bc,dc->abd', U_cur, V_cur, W_cur)
            err = np.linalg.norm(T_d - T)
            if err < 1e-8:
                # Try to round
                U_r = np.round(U_cur * 2) / 2
                V_r = np.round(V_cur * 2) / 2
                W_r = np.round(W_cur * 2) / 2
                if _verify(U_r, V_r, W_r, n, m, p, 1e-7):
                    best = (U_r, V_r, W_r)
                    best_rank = target_rank
                    break
                break
        else:
            continue
        
        if best_rank <= target_rank:
            break
    
    U_f, V_f, W_f = best
    return {"U": [[float(x) for x in row] for row in U_f],
            "V": [[float(x) for x in row] for row in V_f],
            "W": [[float(x) for x in row] for row in W_f]}
# EVOLVE-BLOCK-END
