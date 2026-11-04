<<<<<<< SEARCH
import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
        
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check tensor decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a 3-tensor contraction.
    # We can reshape and use einsum or matrix multiplication.
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # Let's just loop in Python for small n,m,p but use floats for speed? 
    # Or use einsum.
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Compute T[i, j, k]
    # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    # This is not a standard matmul. 
    # We can compute it by looping over i, j, k which is 2*5*5=50 iterations.
    # Each iteration is a dot product of length rank.
    # 50 * rank operations. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Dot product of slices
                val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
=======
import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
        
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return 0.0
        
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    total_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                total_err += (val - 1.0) ** 2
    return total_err

def _gradient_descent_refine(U, V, W, n, m, p, max_iter=200, lr=0.01, tol=1e-10):
    """Refine the decomposition using gradient descent on the sum of squared errors."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return U, V, W
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    prev_err = _compute_error(U, V, W, n, m, p)
    
    for it in range(max_iter):
        # Compute gradients via finite differences or analytical
        # For speed, use a simple approach: perturb each variable slightly
        
        # Compute current error
        cur_err = _compute_error(U, V, W, n, m, p)
        if cur_err < tol:
            break
        
        # Compute gradients for U, V, W using central differences
        # This is expensive but rank is small (40-50) and dimensions are small
        eps = 1e-6
        
        # Gradient for U
        grad_U = np.zeros_like(U_np)
        for idx in range(nm):
            for c in range(rank):
                orig = U_np[idx, c]
                U_np[idx, c] = orig + eps
                err_plus = _compute_error(U_np.reshape(-1, rank).tolist(), V, W, n, m, p)
                U_np[idx, c] = orig - eps
                err_minus = _compute_error(U_np.reshape(-1, rank).tolist(), V, W, n, m, p)
                U_np[idx, c] = orig
                grad_U[idx, c] = (err_plus - err_minus) / (2 * eps)
        
        # Gradient for V
        grad_V = np.zeros_like(V_np)
        for idx in range(mp):
            for c in range(rank):
                orig = V_np[idx, c]
                V_np[idx, c] = orig + eps
                err_plus = _compute_error(U, V_np.reshape(-1, rank).tolist(), W, n, m, p)
                V_np[idx, c] = orig - eps
                err_minus = _compute_error(U, V_np.reshape(-1, rank).tolist(), W, n, m, p)
                V_np[idx, c] = orig
                grad_V[idx, c] = (err_plus - err_minus) / (2 * eps)
        
        # Gradient for W
        grad_W = np.zeros_like(W_np)
        for idx in range(pn):
            for c in range(rank):
                orig = W_np[idx, c]
                W_np[idx, c] = orig + eps
                err_plus = _compute_error(U, V, W_np.reshape(-1, rank).tolist(), n, m, p)
                W_np[idx, c] = orig - eps
                err_minus = _compute_error(U, V, W_np.reshape(-1, rank).tolist(), n, m, p)
                W_np[idx, c] = orig
                grad_W[idx, c] = (err_plus - err_minus) / (2 * eps)
        
        # Update
        U_np -= lr * grad_U
        V_np -= lr * grad_V
        W_np -= lr * grad_W
        
        new_err = _compute_error(U_np.reshape(-1, rank).tolist(), V_np.reshape(-1, rank).tolist(), W_np.reshape(-1, rank).tolist(), n, m, p)
        
        if abs(new_err - prev_err) < 1e-12:
            break
        prev_err = new_err
    
    U_out = U_np.reshape(-1, rank).tolist()
    V_out = V_np.reshape(-1, rank).tolist()
    W_out = W_np.reshape(-1, rank).tolist()
    
    # Round to nearest half-integer if close
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for c in range(rank):
                val = row[c]
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) < 1e-6:
                    row[c] = nearest
    
    return U_out, V_out, W_out
>>>>>>> REPLACE

