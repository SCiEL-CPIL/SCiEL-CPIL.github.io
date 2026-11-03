"""
Optimized matrix multiplication tensor decomposition for 3x3 * 3x6.
Uses numerical optimization (Adam) to find rank-minimal exact decomposition.
Falls back to schoolbook if no valid decomposition found within time budget.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, tol=1e-7):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # D[c, i, j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T[i, j, k] = 1 if k == i*j (element-wise product contributes) -- wait no.
    # The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1.
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[c]_something?
    # Wait, the task says: W[k*n+i, r]. So W is indexed by (k, i).
    # The standard tensor decomposition for matmul:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    
    # Let's verify: for each c, we have a term U[i,j,c] * V[j,k,c] * W[k,i,c].
    # Sum over c should give 1 for all i,j,k? No, that's not right.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k].
    # The "tensor" of the multiplication map is not just T[i,j,k]=1.
    # Actually, the bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # This can be viewed as a trilinear form or as a tensor in (A*B) space.
    # The standard decomposition uses:
    # C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_l V[j,k,c] B[j,k]) ... no.
    
    # Let's re-read the task: "U, V, W factor matrices ... that exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T is defined such that:
    # For matrices A (n x m) and B (m x p), the product C (n x p) has entries C[i,k] = sum_j A[i,j] B[j,k].
    # This is a trilinear map? No, it's bilinear in A and B, linear in C.
    # In tensor decomposition terms, we want to express the multiplication as:
    # C[i,k] = sum_c f_c(A) * g_c(B)
    # where f_c(A) = sum_j U[i,j,c] A[i,j] and g_c(B) = sum_j V[j,k,c] B[j,k].
    # But then we also have W. Let's look at the schoolbook code:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1.
    # The term for column c (corresponding to i,j,k) is:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 * 1 * 1 = 1.
    # And this term contributes to C[i,k] only if we sum over j?
    # Wait, the decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] 
    # This sum is over c. For a fixed i,j,k, how many c's contribute?
    # In schoolbook, each c corresponds to a unique (i,j,k).
    # So for a fixed (i,j,k), only one c (the one for that i,j,k) has all three nonzero.
    # So the sum is 1.
    # But how does this relate to C[i,k]?
    # The full expression is: sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_l V[j*p+k, c] B[j,k])? No.
    
    # Let's think of it as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c [ (sum_j U[i*m+j, c] A[i,j]) * (sum_l V[j*p+k, c] B[j,k]) * W[k*n+i, c] ]?
    # No, that doesn't work dimensionally.
    
    # Actually, the standard way to view matrix multiplication as tensor decomposition:
    # We have a tensor T of shape (n*m, m*p, p*n) where T[i*m+j, j*p+k, k*n+i] = 1? No.
    # The tensor is T[i*m+j, j*p+k, k*n+i] = delta_{i,j,k}? No.
    
    # Let's use the identity:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Multiply both sides by W[k*n+i, c] and sum over c? No.
    
    # The correct interpretation:
    # We want to find U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k.
    # This is NOT the matrix multiplication tensor. This is the identity tensor.
    
    # Wait, I think I'm confusing myself. Let's re-read the task carefully.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for C = A * B is:
    # T[A, B, C] = sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] (sum_j A[i,j] B[j,k])
    # = sum_{i,j,k} (sum_j' A[i,j'] B[j',k]) A[i,j] B[j,k]
    # This is a cubic form. But tensor decomposition usually refers to the bilinear map.
    
    # In the context of arithmetic complexity, the "rank" of matrix multiplication refers to the bilinear complexity.
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # This can be written as: C = sum_c u_c(A) * v_c(B) where u_c(A) and v_c(B) are linear forms.
    # But here we have three matrices U, V, W. This suggests a trilinear form or a different convention.
    
    # Looking at the schoolbook code again:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c corresponding to (i,j,k).
    # The product U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    # This looks like the decomposition of the "diagonal" tensor or the identity.
    
    # Actually, I think the task is using a specific convention where the "matrix multiplication tensor" is defined as:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # And we want to decompose this tensor as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # If this is the case, then the schoolbook decomposition is indeed valid with rank n*m*p.
    
    # Let's verify: for a given (i,j,k), the only c that contributes is the one where (i,j,k) is the index.
    # So sum_c ... = 1. This matches T[i*m+j, j*p+k, k*n+i] = 1.
    
    # So the verification should check:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute D = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # This is expensive, so let's do it efficiently.
    # D[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # We can compute this as: for each r, add outer(U[:,r], outer(V[:,r], W[:,r]))
    # Or use einsum: D = np.einsum('ar,br,cr->abc', U, V, W)
    D = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check if D matches T
    return np.allclose(D, T, atol=tol)

def _try_optimization(rank, time_budget=10):
    """Try to find a decomposition of given rank using numerical optimization."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Initialize with random values
    np.random.seed(42)  # Fixed seed for reproducibility
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Adam optimizer
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    t = 0
    
    while time.time() - start_time < time_budget:
        t += 1
        
        # Compute D = einsum('ar,br,cr->abc', U, V, W)
        D = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target tensor T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Loss: ||D - T||^2
        diff = D - T
        loss = np.sum(diff**2)
        
        # Gradient with respect to U:
        # d/dU[a,r] ||D - T||^2 = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        # Update
        t += 1
        mU = beta1 * mU + (1 - beta1) * grad_U
        vU = beta2 * vU + (1 - beta2) * grad_U**2
        mU_hat = mU / (1 - beta1**t)
        vU_hat = vU / (1 - beta2**t)
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        
        mV = beta1 * mV + (1 - beta1) * grad_V
        vV = beta2 * vV + (1 - beta2) * grad_V**2
        mV_hat = mV / (1 - beta1**t)
        vV_hat = vV / (1 - beta2**t)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        
        mW = beta1 * mW + (1 - beta1) * grad_W
        vW = beta2 * vW + (1 - beta2) * grad_W**2
        mW_hat = mW / (1 - beta1**t)
        vW_hat = vW / (1 - beta2**t)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        # Check convergence
        if loss < 1e-10:
            return U, V, W, True
    
    # Check if final result is valid
    final_D = np.einsum('ar,br,cr->abc', U, V, W)
    final_loss = np.sum((final_D - T)**2)
    if final_loss < 1e-7:
        return U, V, W, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s budget, leaving 10s safety margin
    
    # Try ranks from 40 upwards
    for rank in range(40, 55):
        if time.time() > deadline - 5:
            break
        U, V, W, success = _try_optimization(rank, time_budget=10)
        if success:
            # Round to nearest half-integer
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Verify
            if _verify_decomposition(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
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
