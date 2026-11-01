import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    Attempts rank 10, falls back to rank 11 if verification fails.
    """
    n, m, p = DIMENSIONS
    
    # Rank 10 candidate
    # U: (4, 10), V: (6, 10), W: (6, 10)
    # W index convention: k*n + i
    
    U10 = np.array([
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    ], dtype=int)
    
    V10 = np.array([
        [1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=int)
    
    # W indices: 0:C00, 1:C10, 2:C01, 3:C11, 4:C02, 5:C12
    W10 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C00
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # C10
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0], # C01
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # C11
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0], # C02
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # C12
    ], dtype=int)
    
    def verify(U, V, W):
        # Check if sum_r U[i,j] V[j,k] W[i,k] == delta
        # U[i,j] is U[i*m+j]
        # V[j,k] is V[j*p+k]
        # W[i,k] is W[k*n+i]
        
        n, m, p = 2, 2, 3
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for jp in range(m): # j' for B
                    for k in range(p):
                        # Target: 1 if j==jp and this term contributes to C[i,k]
                        # Actually, the condition is:
                        # sum_r U_r[i,j] V_r[jp,k] W_r[i,k] == 1 if j==jp else 0?
                        # No, the tensor is T[i,j,k] = delta_{i,i'}...
                        # For fixed i, k, the coefficient of A[i,j] B[jp,k] in C[i,k]
                        # should be 1 if j==jp, 0 otherwise.
                        
                        target = 1 if j == jp else 0
                        val = 0
                        for r in range(rank):
                            u_val = U[i*m + j, r]
                            v_val = V[jp*p + k, r]
                            w_val = W[k*n + i, r]
                            val += u_val * v_val * w_val
                        if val != target:
                            return False
        return True

    if verify(U10, V10, W10):
        return {"U": U10.tolist(), "V": V10.tolist(), "W": W10.tolist()}
    else:
        # Fall back to Rank 11
        rank = 11
        U = np.zeros((4, rank), dtype=int)
        V = np.zeros((6, rank), dtype=int)
        W = np.zeros((6, rank), dtype=int)
        
        def set_u(i, j, r, val): U[i*m+j, r] = val
        def set_v(j, k, r, val): V[j*p+k, r] = val
        def set_w(k, i, r, val): W[k*n+i, r] = val

        # Term 1
        set_u(0,0,0,1); set_u(1,1,0,1)
        set_v(0,0,0,1); set_v(1,1,0,1)
        set_w(0,0,0,1); set_w(1,1,0,1)
        
        # Term 2
        set_u(1,0,1,1); set_u(1,1,1,1)
        set_v(0,0,1,1)
        set_w(0,1,1,1); set_w(1,1,1,-1)
        
        # Term 3
        set_u(0,0,2,1)
        set_v(0,1,2,1); set_v(1,1,2,-1)
        set_w(1,0,2,1); set_w(1,1,2,1)
        
        # Term 4
        set_u(1,1,3,1)
        set_v(1,0,3,1); set_v(0,0,3,-1)
        set_w(0,0,3,1); set_w(0,1,3,1)
        
        # Term 5
        set_u(0,0,4,1); set_u(0,1,4,1)
        set_v(1,1,4,1)
        set_w(0,0,4,-1); set_w(1,0,4,1)
        
        # Term 6
        set_u(1,0,5,1); set_u(0,0,5,-1)
        set_v(0,0,5,1); set_v(0,1,5,1)
        set_w(1,1,5,1)
        
        # Term 7
        set_u(0,1,6,1); set_u(1,1,6,-1)
        set_v(1,0,6,1); set_v(1,1,6,1)
        set_w(0,0,6,1)
        
        # Term 8
        set_u(0,0,7,1)
        set_v(0,2,7,1)
        set_w(2,0,7,1)
        
        # Term 9
        set_u(0,1,8,1)
        set_v(1,2,8,1)
        set_w(2,0,8,1)
        
        # Term 10
        set_u(1,0,9,1)
        set_v(0,2,9,1)
        set_w(2,1,9,1)
        
        # Term 11
        set_u(1,1,10,1)
        set_v(1,2,10,1)
        set_w(2,1,10,1)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
