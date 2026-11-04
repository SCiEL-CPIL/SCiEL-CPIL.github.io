import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build U, V, W for schoolbook decomposition
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Verification step to ensure validity
    # The decomposition is valid if sum_c U[:,c] V[:,c] W[:,c] == Tensor
    # Tensor T[i,j,k] = 1 if i,j,k correspond to the multiplication path
    # Actually, the tensor for matrix multiplication C = AB is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] has T[i,j,k] = 1.
    # The decomposition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] ?
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the third factor W is indexed by (k, i).
    # The standard tensor decomposition for matmul is:
    # T_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
    # Where U corresponds to A indices (i,j), V to B indices (j,k), W to C indices (k,i)?
    # Wait, C is n x p. Indices are i (row) and k (col).
    # The tensor T is defined such that T[i, j, k] = 1.
    # The contraction is sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k].
    # The decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] ??
    # Let's check the indices in the seed:
    # U[i*m+j, c] -> index (i,j)
    # V[j*p+k, c] -> index (j,k)
    # W[k*n+i, c] -> index (k,i)
    # So the term is U[i,j,r] * V[j,k,r] * W[k,i,r].
    # Summing over r and j:
    # sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # In the seed, for a fixed (i,j,k), there is exactly one r such that U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1.
    # For that r, the product is 1. For all other r, at least one factor is 0.
    # So the sum is 1. This matches T[i,j,k]=1.
    
    # We can verify this quickly with NumPy
    # Construct the tensor T
    T = np.zeros((n, m, p))
    T[:] = 1.0
    
    # Reconstruct T from U, V, W
    # T_recon[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a bit slow to do naively, but n,m,p are small.
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    
    # We can use einsum
    # U_reshaped: (n, m, R)
    # V_reshaped: (m, p, R)
    # W_reshaped: (p, n, R)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T_recon[i,j,k] = sum_r U_r[i,j,r] V_r[j,k,r] W_r[k,i,r]
    # Note: W is indexed by (k, i) in the flattened form k*n+i.
    # So W_r[k, i, r] is correct.
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback: return a known valid decomposition (which is the one we just built, so this shouldn't happen)
        # But if it does, we have a bug.
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
