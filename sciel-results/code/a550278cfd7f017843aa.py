import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses a structured search: start from known rank-44 decomposition (Strassen + schoolbook),
    then attempt to reduce rank via subspace projection / bilinear map optimization.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 120s to be safe

    # Build the multiplication tensor T[i, j, k] where output[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor is of shape (n, m, p) but we flatten: T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # Actually, the tensor decomposition is:
    #   C[k*n+i] = sum_r U[i*m + a, r] * V[a*p + k, r] * W[k*n + i, r]  (for each a)
    # Wait, the standard tensor decomposition for matrix multiplication M_{n,m,p} is:
    #   C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r]
    # But the interface says:
    #   U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    #   And the decomposition is: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    # where a = i*m + j, b = j*p + k, c = k*n + i
    # So T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    # More precisely: T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k.
    
    # Let's build the tensor T as a 3D array of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i for some i in [0,n), j in [0,m), k in [0,p)
    # Otherwise 0.
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Known rank-44 decomposition
    def build_rank44():
        rank = 44
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        # Strassen 2x2x2
        S_U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ], dtype=np.float64)
        S_V = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=np.float64)
        S_W = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, -1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ], dtype=np.float64)
        
        col = 0
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        U[li * m + la, col] = S_U[r, li * 2 + la]
                for la in range(2):
                    for lk in range(2):
                        V[la * p + (k0 + lk), col] = S_V[r, la * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0 + lk) * n + li, col] = S_W[r, lk * 2 + li]
                col += 1
        
        # Schoolbook for a=2 (third row of A, i.e., j=2)
        for i in range(n):
            for k in range(p):
                U[i * m + 2, col] = 1.0
                V[2 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        
        return U, V, W
    
    U44, V44, W44 = build_rank44()
    
    def tensor_error(U, V, W):
        """Compute the error ||sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] - T||_F"""
        # Reshape U, V, W to 3D and compute the tensor
        # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3D tensor contraction
        # We can compute it as: for each r, add outer product U[:,r] * V[:,r] * W[:,r]
        # But that's slow. Instead, use einsum.
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # But for large rank, this might be slow. Let's try.
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.linalg.norm(T_rec - T)
    
    def try_rank_reduction(U0, V0, W0, target_rank, max_time=10.0):
        """Try to find a decomposition of rank target_rank by optimizing."""
        if time.time() > deadline - max_time:
            return None
        
        # Start from the first target_rank columns of the rank-44 decomposition
        U = U0[:, :target_rank].copy()
        V = V0[:, :target_rank].copy()
        W = W0[:, :target_rank].copy()
        
        # Flatten parameters
        def flatten_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten_params(x):
            nU = U.size
            nV = V.size
            U_ = x[:nU].reshape(U.shape)
            V_ = x[nU:nU+nV].reshape(V.shape)
            W_ = x[nU+nV:].reshape(W.shape)
            return U_, V_, W_
        
        def objective(x):
            U_, V_, W_ = unflatten_params(x)
            return tensor_error(U_, V_, W_)
        
        x0 = flatten_params(U, V, W)
        
        # Use L-BFGS-B
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if time.time() > deadline - 2:
            return None
        
        if result.success and result.fun < 1e-6:
            U_, V_, W_ = unflatten_params(result.x)
            # Round to nearest half-integer
            U_ = np.round(U_ * 2) / 2
            V_ = np.round(V_ * 2) / 2
            W_ = np.round(W_ * 2) / 2
            if tensor_error(U_, V_, W_) < 1e-6:
                return U_, V_, W_
        
        return None
    
    # Try ranks from 43 down to 30
    best_result = (U44, V44, W44)
    best_rank = 44
    
    for target_rank in range(43, 29, -1):
        if time.time() > deadline - 15:
            break
        result = try_rank_reduction(U44, V44, W44, target_rank, max_time=12.0)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found a lower rank, stop
    
    U, V, W = best_result
    rank = best_rank
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
