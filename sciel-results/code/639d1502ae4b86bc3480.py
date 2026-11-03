"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if i,j,k form a valid triple (which they always do in matmul)
    # Actually, the tensor for matmul is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Let's verify by checking that the reconstructed tensor matches the identity pattern
    # For matmul (n x m) * (m x p) -> (n x p), the tensor is T[i,k,j] = delta_{j,j} = 1
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), indexed by [i*m+j, c]
    # V has shape (m*p, rank), indexed by [j*p+k, c]  
    # W has shape (p*n, rank), indexed by [k*n+i, c]
    
    # The tensor being decomposed is the matmul tensor where T[i,j,k] = 1
    # (meaning C[i,k] = sum_j A[i,j] * B[j,k])
    
    # So we need: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's compute this efficiently
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can compute this as a tensor contraction
    
    # For each i, j, k: sum over c
    # This is equivalent to: for fixed i, j: sum_c U3[i,j,c] * (sum_k' V3[j,k',c] * W3[k',i,c])
    # Let's just do it directly with einsum
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-10):
        return False
    return True

def _try_rank_decomposition(rank, max_iters=5000, tol=1e-6):
    """Try to find a rank-r decomposition using gradient descent on a smooth objective."""
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Initialize with small random values
    np.random.seed(42)
    U = np.random.randn(dim_U, rank) * 0.5
    V = np.random.randn(dim_V, rank) * 0.5
    W = np.random.randn(dim_W, rank) * 0.5
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    momentum = 0.9
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    for it in range(max_iters):
        # Compute T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        loss = np.sum((T - target) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        # dL/dT[i,j,k] = 2*(T[i,j,k] - 1)
        dT = 2.0 * (T - target)
        
        # dL/dU3[i,j,c] = sum_k dT[i,j,k] * V3[j,k,c] * W3[k,i,c]
        dU3 = np.einsum('ijk,jkc,kic->ijc', dT, V3, W3)
        dU = dU3.reshape(dim_U, rank)
        
        # dL/dV3[j,k,c] = sum_i dT[i,j,k] * U3[i,j,c] * W3[k,i,c]
        dV3 = np.einsum('ijk,ijc,kic->jkc', dT, U3, W3)
        dV = dV3.reshape(dim_V, rank)
        
        # dL/dW3[k,i,c] = sum_j dT[i,j,k] * U3[i,j,c] * V3[j,k,c]
        dW3 = np.einsum('ijk,ijc,jkc->kic', dT, U3, V3)
        dW = dW3.reshape(dim_W, rank)
        
        # Adam-style update with momentum
        vU = momentum * vU + (1 - momentum) * dU
        vV = momentum * vV + (1 - momentum) * dV
        vW = momentum * vW + (1 - momentum) * dW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
    
    if best_loss < tol:
        return best_U, best_V, best_W, best_loss
    return None, None, None, best_loss

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Leave margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Try ranks from a lower bound up to schoolbook
    # Known lower bound for matmul rank: at least n*p (for 3x3x3, this is 9, but actual is 23 for 4x4x4 etc.)
    # For 3x3x5, let's try from 20 up to 45
    
    # Actually, let's be smarter. The schoolbook is 45.
    # Try a few ranks below 45 to see if we can find a better one.
    
    for rank in range(25, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_decomposition(rank, max_iters=2000, tol=1e-8)
        if result[0] is not None:
            U, V, W, loss = result
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                # Convert to integer/half-integer if possible
                U_list = [[round(x, 10) for x in row] for row in U.tolist()]
                V_list = [[round(x, 10) for x in row] for row in V.tolist()]
                W_list = [[round(x, 10) for x in row] for row in W.tolist()]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
