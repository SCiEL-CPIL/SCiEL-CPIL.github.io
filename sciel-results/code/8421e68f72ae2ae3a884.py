"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Uses a structured search over low-rank tensor decompositions with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    
    # The tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # T[i, j, k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # In the output convention W[k*n + i, r], the tensor index is (k*n+i, i*m+j, j*p+k)
    # Actually, let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where a indexes (i,j) pairs, b indexes (j,k) pairs, c indexes (k,i) pairs? 
    # Wait, the output factor W[k*n+i, r] suggests c = k*n + i.
    # So T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # The tensor entry is 1 when (a,b,c) corresponds to a valid multiplication term, i.e.,
    # A[i,j] * B[j,k] contributes to C[i,k].
    # In the standard matrix multiplication tensor, T[i, j, k] = 1 for all i, j, k,
    # and the multiplication is sum_j A[i,j] * B[j,k] = C[i,k].
    # So the tensor T has entries T[i, j, k] = 1, and the decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i, j, k) where j is shared.
    # Wait, but not all combinations of (i,j,k) are independent - j must be the same index.
    # Actually, the tensor is T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # and T[a, b, c] = 1 if j = j' and i = i' and k = k', else 0.
    # Hmm, that doesn't seem right either. Let me think again.
    
    # Standard matrix multiplication: C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) with T[i, j, k] = 1.
    # In the flattened form: a = i*m + j, b = j*p + k, c = k*n + i
    # The condition is that the middle index j matches between a and b, and i matches between a and c, and k matches between b and c.
    # So T[a, b, c] = 1 iff (a % m == b // p) and (a // m == c % n) and (b % p == c // n).
    # Wait, let me re-derive.
    # a = i*m + j, so i = a // m, j = a % m
    # b = j'*p + k, so j' = b // p, k = b % p
    # c = k'*n + i', so k' = c // n, i' = c % n
    # T[a, b, c] = 1 iff j == j' and i == i' and k == k'
    # i.e., a % m == b // p and a // m == c % n and b % p == c // n
    
    # Let me construct the tensor and try to find a low-rank decomposition.
    
    def build_tensor():
        """Build the matrix multiplication tensor in flattened form."""
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T = build_tensor()
    
    # Try to find rank-r decomposition using random initialization and optimization
    # We want to minimize sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
    # subject to U, V, W having integer or half-integer entries.
    
    # First, let's try small ranks with random restarts.
    # The known minimum rank for 3x3 * 3x3 is 19 (Strassen-like), and for 3x3 * 3x5 it should be similar.
    # Actually, let me just try to find a good decomposition.
    
    best_rank = n * m * p
    best_U = best_V = best_W = None
    
    def try_rank(r, n_restarts=50):
        """Try to find a rank-r decomposition."""
        global best_rank, best_U, best_V, best_W
        for restart in range(n_restarts):
            if time.time() > deadline:
                return
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize to avoid scale issues
            for col in range(r):
                norm = np.linalg.norm(U[:, col]) * np.linalg.norm(V[:, col]) * np.linalg.norm(W[:, col])
                if norm > 0:
                    scale = norm ** (1/3)
                    U[:, col] /= scale
                    V[:, col] /= scale
                    W[:, col] /= scale
            
            # Use gradient descent on the flattened parameters
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            U_size = n * m * r
            V_size = m * p * r
            W_size = p * n * r
            
            def loss_and_grad(params):
                U = params[:U_size].reshape(n * m, r)
                V = params[U_size:U_size + V_size].reshape(m * p, r)
                W = params[U_size + V_size:].reshape(p * n, r)
                
                # Compute tensor product
                # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # = einsum('ar,br,cr->abc', U, V, W)
                T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                loss = np.sum((T - T_approx) ** 2)
                
                # Gradient
                # dL/dU[a,r] = 2 * sum_{b,c} (T[a,b,c] - T_approx[a,b,c]) * V[b,r] * W[c,r]
                # = 2 * einsum('abc,br,cr->ar', (T - T_approx), V, W)
                dT = T - T_approx
                dU = 2 * np.einsum('abc,br,cr->ar', dT, V, W, optimize=True)
                dV = 2 * np.einsum('abc,ar,cr->br', dT, U, W, optimize=True)
                dW = 2 * np.einsum('abc,ar,br->cr', dT, U, V, optimize=True)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return loss, grad
            
            # Simple gradient descent with adaptive step size
            step_size = 0.01
            for it in range(2000):
                if time.time() > deadline:
                    return
                loss, grad = loss_and_grad(params)
                if loss < 1e-6:
                    break
                params -= step_size * grad
                # Adaptive step size
                loss_new, _ = loss_and_grad(params)
                if loss_new > loss:
                    step_size *= 0.5
                    params += step_size * grad
                elif loss_new > loss * 0.9:
                    step_size *= 0.8
                else:
                    step_size *= 1.1
            
            U = params[:U_size].reshape(n * m, r)
            V = params[U_size:U_size + V_size].reshape(m * p, r)
            W = params[U_size + V_size:].reshape(p * n, r)
            
            # Round to nearest half-integer and verify
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_approx = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded, optimize=True)
            if np.max(np.abs(T - T_approx)) < 1e-9:
                if r < best_rank:
                    best_rank = r
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    return
    
    # Try ranks from 18 upward
    for r in range(18, n * m * p + 1):
        if time.time() > deadline:
            break
        n_restarts = 30 if r <= 25 else 10
        try_rank(r, n_restarts)
        if best_rank <= r:
            break
    
    # Convert to list of lists
    U_list = best_U.tolist() if best_U is not None else None
    V_list = best_V.tolist() if best_V is not None else None
    W_list = best_W.tolist() if best_W is not None else None
    
    if U_list is None:
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
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

