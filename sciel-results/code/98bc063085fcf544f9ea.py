import numpy as np

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if j is the middle index (i.e., T[i,j,k] = delta_{j, j} for the multiplication tensor)
    # Actually, the matrix multiplication tensor T[i,j,k] = 1 if the index j matches
    # For M_{n,m,p}: T[i,j,k] = 1 if j is in 0..m-1, and it's the (i,j) entry of A times (j,k) entry of B
    # So T[i,j,k] = 1 for all valid i,j,k where j is the shared index
    # Wait, the tensor is T[i,j,k] = 1 if j is the middle index
    # Actually, T[i,j,k] = delta_{j,j} which is always 1 for valid indices
    # No, the matrix multiplication tensor has T[i,j,k] = 1 if and only if j is the index that A[i,j] and B[j,k] share
    # So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # And the decomposition is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # Wait, the W index is W[k*n+i, r]
    
    # Let me just check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                expected = 1
                if abs(val - expected) > 1e-10:
                    return False, (i, j, k, val, expected)
    return True, None
