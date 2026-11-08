<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U3[i, j, :]
                v_vec = V3[j, k, :]
                w_vec = W3[k, i, :]
                total = np.sum(u_vec * v_vec * w_vec)
                if abs(total - 1.0) > tol:
                    return False
    return True
=======
import numpy as np
import time

DIMENSIONS = [2, 4, 7]
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    
    # Vectorized verification: compute sum over rank for each (i,j,k)
    # U3[i,j,:] * V3[j,k,:] * W3[k,i,:] summed over last axis
    # Shape: (n, m, p)
    result = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_slice = U3[:, :, r]  # (n, m)
        v_slice = V3[:, :, r]  # (m, p)
        w_slice = W3[:, :, r]  # (p, n)
        # outer product-like: u[i,j] * v[j,k] * w[k,i]
        # Need to broadcast: (n,m,1) * (1,m,p) * (p,1,n) -> (n,m,p)
        result += u_slice[:, :, np.newaxis] * v_slice[np.newaxis, :, :] * w_slice.T[np.newaxis, np.newaxis, :]
    
    return np.all(np.abs(result - 1.0) <= tol)
>>>>>>> REPLACE

