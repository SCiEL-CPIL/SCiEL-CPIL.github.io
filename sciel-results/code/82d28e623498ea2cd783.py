"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Build the matrix multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute reshaped tensors for ALS
    nm, mp, pn = T.shape
    T_mat_U = T.reshape(nm, mp * pn)
    T_mat_V = T.reshape(mp, nm * pn)
    T_mat_W = T.reshape(pn, nm * mp)
    
    # Try ranks from low to high
    for rank in range(30, 61):
        if time.time() > deadline:
            break
        result = _find_decomposition(T, T_mat_U, T_mat_V, T_mat_W, rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}

def _find_decomposition(T, T_mat_U, T_mat_V, T_mat_W, rank):
    nm, mp, pn = T.shape
    
    for restart in range(15):
        np.random.seed(restart * 7919 + rank * 104729)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        for it in range(80):
            for c in range(rank):
                VW = np.outer(V[:, c], W[:, c])
                U[:, c] = T_mat_U @ VW
            for c in range(rank):
                UW = np.outer(U[:, c], W[:, c])
                V[:, c] = T_mat_V @ UW
            for c in range(rank):
                UV = np.outer(U[:, c], V[:, c])
                W[:, c] = T_mat_W @ UV
        
        # Check error
        err = 0.0
        for c in range(rank):
            outer = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            err += np.sum((T - outer) ** 2)
        
        if err < 1e-10:
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            err_snap = 0.0
            for c in range(rank):
                outer = np.einsum('i,j,k->ijk', U_snap[:, c], V_snap[:, c], W_snap[:, c])
                err_snap += np.sum((T - outer) ** 2)
            
            if err_snap < 1e-8:
                return _to_list(U_snap, V_snap, W_snap)
    
    return None

def _to_list(U, V, W):
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
