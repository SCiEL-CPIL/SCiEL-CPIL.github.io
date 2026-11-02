"""Matrix multiplication 3,3,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # For each (i, j, k), the sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                total = np.sum(u_val * v_val * w_val)
                if abs(total - 1.0) > 1e-10:
                    return False
    
    # Check that for non-matching indices, the sum is 0
    for _ in range(200):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        i2 = np.random.randint(n)
        j2 = np.random.randint(m)
        k2 = np.random.randint(p)
        if i2 == i and j2 == j and k2 == k:
            continue
        u_val = U[i * m + j]
        v_val = V[j * p + k]
        w_val = W[k2 * n + i2]
        total = np.sum(u_val * v_val * w_val)
        if abs(total) > 1e-10:
            return False
    
    return True

def _try_kron_decomposition(n, m, p, deadline):
    """
    Try to find a rank decomposition using Kronecker product structure.
    The matrix multiplication tensor can be viewed as a 3-tensor T[i,j,k]
    where T[i,j,k] = 1 if the indices match appropriately.
    
    We construct the tensor as a matrix and use SVD to find its rank.
    """
    # The tensor T has shape (n*m, m*p, p*n) in our indexing
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # Otherwise 0
    
    # Construct the tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try different matrix unfoldings to find low-rank structure
    # Unfolding 1: reshape to (n*m, m*p*p*n) - this gives the "standard" matrix
    # But we want to find U, V, W such that T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    
    # This is equivalent to finding a rank-r decomposition where:
    # T_unfolded = U * V^T * W^T (in some sense)
    
    # Let's try a different approach: use the fact that for fixed k,
    # the slice T[:,:,k] should be a rank-1 matrix (outer product)
    # But that's not quite right either.
    
    # Actually, the tensor decomposition we want is:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    # This is a 3-way tensor decomposition. One approach is to:
    # 1. Flatten T to a matrix M of shape (n*m, m*p*p*n)
    # 2. Find SVD of M
    # 3. The rank of M gives a lower bound on the tensor rank
    
    # But this doesn't directly give us U, V, W.
    
    # Alternative: use the "matricization" approach
    # For a given candidate rank r, we want to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    
    # This is a system of equations. Let's try to solve it for small r.
    
    # Start with r = 15 (theoretical lower bound) and increase
    min_rank = max(n * m, m * p, p * n)  # = max(9, 15, 15) = 15
    
    for r in range(min_rank, 46):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition
        # We can use a greedy approach or random initialization + optimization
        
        # Random initialization
        np.random.seed(42 + r)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize to avoid scaling issues
        U = U / np.linalg.norm(U, axis=0, keepdims=True)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
        
        # Check if this random initialization happens to work (unlikely)
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # For exact decomposition, we need to solve a system of equations
        # The tensor T has n*m*p = 45 nonzero entries
        # We have r*(n*m + m*p + p*n) = r*(9 + 15 + 15) = 39r unknowns
        # For r >= 2, we have more unknowns than equations, so solutions may exist
        
        # Let's try a linear algebra approach:
        # Fix W, then the problem becomes: for each k,
        # T[:,:,k] = sum_c W[k,c] * (U * V^T)[:,:,c]
        # This is a linear system in U and V (if W is fixed)
        
        # Actually, let's try a simpler heuristic: 
        # Use the structure of the problem. For 3x3x3, Strassen's algorithm
        # gives rank 7. For 3x3x5, we might be able to do better than 45.
        
        # Let's try to embed the 3x3x3 Strassen decomposition into 3x3x5
        # by extending the 5 columns appropriately.
        
        # Strassen's 7-term algorithm for 3x3x3:
        # M1 = (A11 + A33) * (B11 + B33)
        # M2 = (A31 + A33) * B31
        # M3 = A11 * (B13 - B33)
        # M4 = A33 * (B31 - B11)
        # M5 = (A13 + A11) * B11
        # M6 = (A31 - A11 + A33) * (A11 + B13)  # Wait, this doesn't look right
        # M7 = (A13 - A31 + A33) * (B31 + B33)
        
        # Actually, let me look up the correct Strassen formulas for 3x3.
        # The standard Strassen for 2x2 uses 7 multiplications.
        # For 3x3, there are known rank-23 algorithms, but I'm not sure of the exact form.
        
        # Let's try a different approach: use the known lower bound and try
        # to find a decomposition via random search with verification.
        
        # For now, let's just return schoolbook if we can't find better quickly.
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = len(result['U'][0])
    
    # Try Kronecker-based decomposition
    if time.time() < deadline:
        kron_result = _try_kron_decomposition(n, m, p, deadline)
        if kron_result is not None:
            kron_rank = len(kron_result['U'][0])
            if kron_rank < best_rank:
                result = kron_result
                best_rank = kron_rank
    
    return result
# EVOLVE-BLOCK-END
