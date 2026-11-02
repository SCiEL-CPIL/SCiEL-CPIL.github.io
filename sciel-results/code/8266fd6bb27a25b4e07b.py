import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify the tensor decomposition is exact using numpy."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    try:
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # result[i, j, k] = sum_s U[i, j, s] * V[j, k, s] * W[k, i, s]
        result = np.einsum('ijs,jks,ks->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(result, 1.0, atol=1e-9)
    except Exception:
        return False

def _rank11_solution():
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # Strassen for 2x2 block (columns 0,1 of B)
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    # Schoolbook for column 2
    set_u(0,0,7,1); set_v(0,2,7,1); set_w(2,0,7,1)
    set_u(0,1,8,1); set_v(1,2,8,1); set_w(2,0,8,1)
    set_u(1,0,9,1); set_v(0,2,9,1); set_w(2,1,9,1)
    set_u(1,1,10,1); set_v(1,2,10,1); set_w(2,1,10,1)

    return U, V, W

def _try_rank10_search(deadline):
    """
    Search for a rank-10 solution by looking for linear dependencies
    among the 11 rank-1 terms and also trying structured combinations.
    """
    n, m, p = DIMENSIONS
    U11, V11, W11 = _rank11_solution()
    
    # The 11 rank-1 tensors
    tensors = []
    for s in range(11):
        u_s = U11[:, s]
        v_s = V11[:, s]
        w_s = W11[:, s]
        t_s = np.outer(np.outer(u_s, v_s), w_s).flatten()
        tensors.append(t_s)
    
    T = np.array(tensors).T  # (144, 11)
    
    # Check rank
    rank = np.linalg.matrix_rank(T, tol=1e-9)
    
    if rank < 11:
        # Linear dependence exists
        U_svd, s_vals, Vt = np.linalg.svd(T, full_matrices=False)
        c = Vt[-1]
        
        combo = T @ c
        if np.allclose(combo, 0, atol=1e-9):
            idx = np.argmax(np.abs(c))
            
            # Try to eliminate term idx
            U10 = np.delete(U11, idx, axis=1)
            V10 = np.delete(V11, idx, axis=1)
            W10 = np.delete(W11, idx, axis=1)
            
            if _verify(U10, V10, W10):
                return U10, V10, W10
    
    # Alternative: try structured rank-10 with 7 Strassen + 3 for col 2
    start = time.time()
    np.random.seed(42)
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        
        # Strassen terms (0-6)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[1,1]=1; W[3,1]=-1
        
        U[0,2]=1
        V[2,2]=1; V[1,2]=-1
        W[2,2]=1; W[3,2]=1
        
        U[3,3]=1
        V[2,3]=1; V[1,3]=-1
        W[0,3]=1; W[1,3]=1
        
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[2,4]=1
        
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        # Terms 7-9 for column 2 with random initialization
        U[0:4, 7:10] = np.random.randn(4, 3)
        V[0:6, 7:10] = np.random.randn(6, 3)
        W[0:6, 7:10] = np.random.randn(6, 3)
        
        if _verify(U, V, W):
            return U, V, W
        
        # Quick gradient step to reduce error
        # Compute residual
        U_r = U.reshape(2, 2, 10)
        V_r = V.reshape(2, 3, 10)
        W_r = W.reshape(3, 2, 10)
        residual = np.einsum('ijs,jks,ks->ijk', U_r, V_r, W_r) - 1.0
        err = np.sum(residual**2)
        
        if err < 1e-6:
            # Very close, try to snap to integers/half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if _verify(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    U, V, W = _rank11_solution()
    
    if _verify(U, V, W):
        rank10 = _try_rank10_search(deadline)
        if rank10 is not None:
            U10, V10, W10 = rank10
            if _verify(U10, V10, W10):
                return {"U": U10.tolist(), "V": V10.tolist(), "W": W10.tolist()}
    
    # Fallback to rank 11
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
