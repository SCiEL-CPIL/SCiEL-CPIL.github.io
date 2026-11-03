import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
        
    # Try ranks from 9 upwards
    for R in range(9, 25):
        # Multiple restarts for each rank
        for restart in range(20):
            # Check time
            if time.time() > 140:
                break
            
            # Initialize
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # ALS iterations
            for _ in range(50):
                # Solve for U
                # T = U V^T W^T ? No, T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
                # Reshape T to (nm, mp*pn)
                T_reshaped = T.reshape(nm, mp * pn)
                VW = (V * W).reshape(mp * pn, R) # This is not quite right for LS
                # Actually, T = U @ (V * W)^T is not correct.
                # T_{i, j, k} = sum_r U_{i,r} V_{j,r} W_{k,r}
                # For fixed i, T_{i, :, :} = sum_r U_{i,r} (V_r \otimes W_r)
                # Let M be a matrix of size (mp*pn, R) where M_{(j,k), r} = V_{j,r} W_{k,r}
                M = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, R)
                # T_reshaped[i, :] = U[i, :] @ M^T
                U, _, _, _ = lstsq(M.T, T_reshaped.T, lapack_driver='gelsd')
                
                # Solve for V
                # T_{:, j, k} = sum_r V_{j,r} (U_r \otimes W_r)
                M = np.einsum('ir,kr->ikr', U, W).reshape(nm * pn, R)
                T_reshaped = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
                V, _, _, _ = lstsq(M.T, T_reshaped.T, lapack_driver='gelsd')
                
                # Solve for W
                # T_{i, j, k} = sum_r W_{k,r} (U_r \otimes V_r)
                M = np.einsum('ir,jr->ijr', U, V).reshape(nm * mp, R)
                T_reshaped = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
                W, _, _, _ = lstsq(M.T, T_reshaped.T, lapack_driver='gelsd')
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify
            Tc = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            if np.allclose(Tc, T, atol=1e-10):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        if time.time() > 140:
            break

    # Fallback to schoolbook
    R_fb = n * m * p
    U = np.zeros((nm, R_fb))
    V = np.zeros((mp, R_fb))
    W = np.zeros((pn, R_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
