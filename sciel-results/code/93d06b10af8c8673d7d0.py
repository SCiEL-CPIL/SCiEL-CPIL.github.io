import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    I, J, K = n*m, m*p, p*n # 9, 18, 18
    
    # Target: Rank 46 decomposition for 3x3x6
    # Strategy: Find rank 23 decomposition for 3x3x3, then expand.
    
    # 1. Build 3x3x3 tensor
    n3, m3, p3 = 3, 3, 3
    I3, J3, K3 = n3*m3, m3*p3, p3*n3 # 9, 9, 9
    T3 = np.zeros((I3, J3, K3), dtype=float)
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                a = i * m3 + j
                b = j * p3 + k
                c = k * n3 + i
                T3[a, b, c] = 1.0
                
    T3_I = T3.reshape(I3, J3*K3)
    T3_J = T3.transpose(1,0,2).reshape(J3, I3*K3)
    T3_K = T3.transpose(2,0,1).reshape(K3, I3*J3)
    
    R3 = 23
    found_3 = False
    U3 = None
    V3 = None
    W3 = None
    
    start_time = time.time()
    timeout = 100.0
    
    # Try to find rank 23 decomposition for 3x3x3
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(42 + restart)
        U = np.random.randn(I3, R3)
        V = np.random.randn(J3, R3)
        W = np.random.randn(K3, R3)
        
        for it in range(200):
            # Update U
            V_kron_W = (V[:, None, :] * W[None, :, :]).reshape(J3*K3, R3)
            try:
                U = np.linalg.lstsq(V_kron_W, T3_I, rcond=None)[0]
            except:
                pass
                
            # Update V
            U_kron_W = (U[:, None, :] * W[None, :, :]).reshape(I3*K3, R3)
            try:
                V = np.linalg.lstsq(U_kron_W, T3_J, rcond=None)[0]
            except:
                pass
                
            # Update W
            U_kron_V = (U[:, None, :] * V[None, :, :]).reshape(I3*J3, R3)
            try:
                W = np.linalg.lstsq(U_kron_V, T3_K, rcond=None)[0]
            except:
                pass
                
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = np.linalg.norm(T3 - T_approx)
        
        if residual < 1e-6:
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            T_approx_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
            if np.allclose(T3, T_approx_int, atol=1e-9):
                U3, V3, W3 = U_int, V_int, W_int
                found_3 = True
                break
                
    if found_3:
        # Expand to 3x3x6
        # U shape (9, 46)
        # V shape (18, 46)
        # W shape (18, 46)
        
        U = np.zeros((I, 2*R3), dtype=float)
        V = np.zeros((J, 2*R3), dtype=float)
        W = np.zeros((K, 2*R3), dtype=float)
        
        # First 23 columns: correspond to first 3x3 block of B and C
        U[:, :R3] = U3
        V[:9, :R3] = V3
        W[:9, :R3] = W3
        
        # Last 23 columns: correspond to second 3x3 block of B and C
        U[:, R3:] = U3
        V[9:, R3:] = V3
        W[9:, R3:] = W3
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook 54
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
