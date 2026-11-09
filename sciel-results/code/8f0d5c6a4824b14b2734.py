<<<<<<< SEARCH
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the matrix multiplication tensor: T[i, j, k, l] = 1 if j==k else 0
    # In the factorization: T[i, j, k, l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we check: for all i, j, k, l: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == (1 if j==k else 0)
    # We can do this efficiently by reshaping.
    # Let's verify on a random matrix product to be safe and fast.
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Reconstruct C from the tensor factors
    # C[i, k] = sum_{j, l} A[i, j] * B[j, k] * (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r])
    # But the tensor entry T[i, j, k, l] should be 1 if j==k else 0.
    # So C[i, k] = sum_j A[i, j] * B[j, k] * (1 if j==k else 0) = sum_j A[i, j] * B[j, k] * delta(j,k)
    # Wait, the standard matrix multiplication is C[i, k] = sum_j A[i, j] * B[j, k].
    # The tensor for matrix multiplication is T_{i, j, k} = delta_{j, k}? No.
    # The tensor T is such that C_{i, k} = sum_{j} A_{i, j} B_{j, k}.
    # In tensor notation, the multiplication tensor has indices (i, j, k) and value 1 if the "middle" indices match?
    # Actually, the tensor M for matrix multiplication of size n x m x p is:
    # M[i, j, k] = 1 if j == k (assuming m=p), else 0.
    # A rank-r decomposition is M[i, j, k] = sum_{t=1}^r U[i, j, t] V[j, k, t] W[k, i, t].
    # Here U[i, j, t] is U[i*m+j, t], V[j, k, t] is V[j*p+k, t], W[k, i, t] is W[k*n+i, t].
    # So we need to check: for all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == (1 if j==k else 0).
    # Let's verify this condition.
    U_flat = U
    V_flat = V
    W_flat = W
    # We can check a subset or use random sampling if rank is large, but for 49 it's small.
    # Let's check all i, j, k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over t
                s = 0.0
                for t in range(rank):
                    s += U_flat[i*m+j, t] * V_flat[j*p+k, t] * W_flat[k*n+i, t]
                expected = 1.0 if j == k else 0.0
                if abs(s - expected) > 1e-9:
                    return False
    return True
=======
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    # Convert to numpy arrays to ensure .shape and indexing work correctly
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Check: for all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == (1 if j==k else 0)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
                expected = 1.0 if j == k else 0.0
                if abs(s - expected) > 1e-9:
                    return False
    return True
>>>>>>> REPLACE

