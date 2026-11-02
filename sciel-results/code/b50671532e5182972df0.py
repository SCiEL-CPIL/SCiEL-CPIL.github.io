"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Check if the decomposition exactly represents matrix multiplication.
    For each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Efficient check: for each c, the outer product U_c ⊗ V_c ⊗ W_c contributes
    # We need sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k
    # This is equivalent to: the tensor T must equal the all-ones tensor of shape (n,m,p)
    # Check by flattening: T_flat = einsum('ic,jc,kc->ijk', U, V, W)
    # But that's expensive. Instead, check a random subset or use the full check.
    # Full check:
    for c in range(rank):
        pass  # We'll do the full tensor product check
    # Use einsum for the full check
    T = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _try_rank(rank, max_iters=2000, lr=0.01, seed=42):
    """Try to find a rank-rank decomposition via gradient descent."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Target tensor
    target = np.ones((n, m, p))
    
    for it in range(max_iters):
        # Compute current tensor
        T = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
        diff = T - target
        
        # Loss = 0.5 * ||T - target||^2
        loss = 0.5 * np.sum(diff * diff)
        
        # Gradients
        # dL/dU[i*m+j, c] = sum_{k} diff[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        # dL/dV[j*p+k, c] = sum_{i} diff[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
        # dL/dW[k*n+i, c] = sum_{j} diff[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
        
        # Efficient computation
        # dU[i*m+j, c] = sum_k diff[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        # This is a bit complex. Let's use einsum.
        
        # For U: dU[i*m+j, c] = sum_k T_diff[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape V to (m, p, rank) and W to (p, n, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        U3 = U.reshape(n, m, rank)
        
        # dU3[i, j, c] = sum_k diff[i,j,k] * V3[j,k,c] * W3[k,i,c]
        dU3 = np.einsum('ijk,jkc,kic->ijc', diff, V3, W3, optimize=True)
        dU = dU3.reshape(n * m, rank)
        
        # dV3[j, k, c] = sum_i diff[i,j,k] * U3[i,j,c] * W3[k,i,c]
        dV3 = np.einsum('ijk,ijc,kic->jkc', diff, U3, W3, optimize=True)
        dV = dV3.reshape(m * p, rank)
        
        # dW3[k, i, c] = sum_j diff[i,j,k] * U3[i,j,c] * V3[j,k,c]
        dW3 = np.einsum('ijk,ijc,jkc->kic', diff, U3, V3, optimize=True)
        dW = dW3.reshape(p * n, rank)
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if loss < 1e-12:
            break
    
    if _verify(U, V, W):
        return U, V, W
    return None

def _kronecker_product_decomposition():
    """Use Kronecker product structure.
    
    2x3 * 3x9: We can view the 3x9 as (3x3) x (1x3) in some sense, or use
    the fact that 9 = 3*3.
    
    Actually, let's think about this differently. The matrix multiplication tensor
    for (n,m,p) has rank at most n*m*p. We want to reduce this.
    
    One approach: use the fact that the tensor is a sum of rank-1 tensors.
    The schoolbook gives rank n*m*p = 2*3*9 = 54.
    
    Can we do better? Let's try to find a lower rank decomposition.
    
    For 2x3 * 3x9, note that the middle dimension is 3. We can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a bilinear map. The tensor is T[i,j,k] = 1 (in the sense that each
    term A[i,j]*B[j,k] contributes to C[i,k]).
    
    Let's try a rank-30 decomposition or lower.
    """
    n, m, p = DIMENSIONS
    
    # Try to use the structure: for each j, the term A[i,j]*B[j,k] gives a
    # rank-1 contribution in the (i,k) space with weight B[j,k]... no, that's not right.
    
    # Actually, the standard decomposition is:
    # For each (i,j,k): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # This gives rank n*m*p.
    
    # To reduce rank, we need to find linear dependencies.
    
    # Let's try: for fixed j, the terms over i and k form a matrix of size n x p.
    # The contribution to C[i,k] from A[i,j]*B[j,k] is a rank-1 matrix (outer product
    # of column j of A and row j of B).
    
    # So C = sum_j A[:,j] * B[j,:]
    
    # This is already the standard form. The tensor rank is the minimum number of
    # rank-1 tensors that sum to the multiplication tensor.
    
    # For now, let's just return the schoolbook decomposition.
    U, V, W = _schoolbook()
    return U, V, W

def construct():
    deadline = time.time() + 120
    best_result = None
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    if _verify(U, V, W):
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to improve with gradient descent at various ranks
    for rank in range(30, 55):
        if time.time() > deadline - 10:
            break
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = _try_rank(rank, max_iters=500, lr=0.05, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if best_result is None:
        U, V, W = _schoolbook()
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result

# EVOLVE-BLOCK-END
