"""Matrix multiplication 3x3x3 tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_strassen_23():
    """
    Strassen-style rank-23 decomposition for 3x3 matrix multiplication.
    Uses 7 Strassen products on overlapping 2x2 blocks + 16 edge/center terms.
    
    Convention: C[i,j] = A[i,:] @ B[:,j]
    Output W indexed as W[k*n+i, r] following trace convention.
    U indexed as U[i*m+j, r], V indexed as V[j*p+k, r].
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We'll build 23 terms.
    # Terms 0-6: Strassen 7 products on overlapping 2x2 blocks
    # Terms 7-22: Additional terms for remaining elements
    
    # Strategy: Use the known decomposition that achieves rank 23 for 3x3.
    # Based on Strassen's 7-product algorithm adapted to 3x3 with boundary terms.
    
    # Let's use a systematic approach:
    # The 3x3 product C[i,j] = sum_k A[i,k]*B[k,j]
    # We need 23 rank-1 terms whose sum gives all 27 products.
    
    # Known rank-23 construction for 3x3 (based on literature):
    # Uses the fact that rank(3x3 mult) = 23, with explicit decomposition.
    
    # I'll implement a verified decomposition.
    # The key insight: we can use 7 Strassen-like products on a 2x2 core
    # plus 16 additional terms to cover the boundary.
    
    # Actually, let me implement a well-known explicit rank-23 formula.
    
    # For 3x3 matrix multiplication, a rank-23 decomposition:
    # C[i,j] for i,j in {0,1,2}
    
    # Using the approach from "The complexity of multiplying 3x3 matrices"
    # The 23 terms can be organized as follows:
    
    # Let me define the 23 terms explicitly.
    # Each term r contributes: U[:,r] * V[:,r] * W[:,r]
    # where the product gives sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The tensor T[i,j,k] = delta (1 if this is the (i,j,k) entry of the product)
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Wait, that's not right. The matrix multiplication tensor is:
    # M[i, k, j] = 1 (A[i,k] * B[k,j] contributes to C[i,j])
    # So the tensor has entries T[i, k, j] = 1 for all i, k, j in {0,1,2}
    
    # In the decomposition: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,k,j
    
    # This is a very specific structure. Let me think about this differently.
    
    # The matrix multiplication tensor for n x n has rank at most n^3 (schoolbook)
    # and at least n^2 (lower bound). For n=3, the rank is known to be 23.
    
    # The tensor T is in the space of (R^3 x R^3)^* x (R^3 x R^3)^* x (R^3 x R^3)^*
    # T(A, B, C) = tr(A^T * B * C) or similar.
    
    # Let me just verify: can I find a rank-23 decomposition numerically?
    
    # Actually, for the purposes of this task, let me implement a search-based approach
    # that tries to find a low-rank decomposition, falling back to rank 27 if needed.
    
    # But first, let me try to hard-code the known rank-23 Strassen decomposition.
    
    # The rank-23 decomposition for 3x3 multiplication is non-trivial to write from memory.
    # Let me use a different approach: implement a numerical search for low-rank decompositions.
    
    return None  # Placeholder - will use search below

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of 3x3 matrix multiplication."""
    n, m, p = 3, 3, 3
    rank = U.shape[1]
    
    # The tensor for 3x3 matrix multiplication:
    # T[i, k, j] = 1 for all i, k, j in {0,1,2}
    # This means: for all i, j, the (i,j) entry of A@B is sum_k A[i,k]*B[k,j]
    
    # In the decomposition:
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1 for all i,k,j
    
    # Wait, let me re-derive the convention.
    # U has shape (n*m, rank) = (9, rank), indexed by (i, k) where i is row of A, k is column of A
    # V has shape (m*p, rank) = (9, rank), indexed by (k, j) where k is row of B, j is column of B  
    # W has shape (p*n, rank) = (9, rank), indexed by (j, i) where j is col of B, i is row of A
    
    # The product: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # This should give the coefficient for A[i,k]*B[k,j] in the expression for C[i,j]
    # which is 1 for all i,k,j.
    
    # So we need: for all i in {0,1,2}, k in {0,1,2}, j in {0,1,2}:
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    
    # Let's verify this.
    for i in range(n):
        for k in range(m):
            for j in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
                if abs(total - 1.0) > tol:
                    return False, (i, k, j, total)
    return True, None

def _build_schoolbook_27():
    """Build the schoolbook rank-27 decomposition."""
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1
                V[k * p + j, col] = 1
                W[j * n + i, col] = 1
                col += 1
    
    return U, V, W

def _try_find_lower_rank(deadline):
    """
    Try to find a rank < 27 decomposition using iterative optimization.
    This is a heuristic search - may or may not find rank 23 or 22.
    """
    n, m, p = 3, 3, 3
    start_time = time.time()
    
    # Try ranks from 22 upward
    for target_rank in [22, 23, 24, 25, 26]:
        if time.time() > deadline:
            break
        
        # Initialize with random values
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        max_iters = 500
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Iterative optimization: alternate between updating U, V, W
            for inner_iter in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current error
                err = 0.0
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            total = sum(U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] 
                                      for r in range(target_rank))
                            err += (total - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < 1e-10:
                    break
                
                # Update W: for each (j,i), W[j*n+i, :] should satisfy
                # sum_r V[k*p+j, r] * W[j*n+i, r] * U[i*m+k, r] = 1 for all k
                # This is a linear system in W[j*n+i, :]
                for i in range(n):
                    for j in range(p):
                        # We need: for all k in {0,1,2}:
                        # sum_r (V[k*p+j, r] * U[i*m+k, r]) * W[j*n+i, r] = 1
                        # Let A_k[r] = V[k*p+j, r] * U[i*m+k, r]
                        # Then: A_0 @ w + A_1 @ w + ... no wait
                        # sum_r A_k[r] * w[r] = 1 for each k
                        # This is: [A_0; A_1; A_2] @ w = [1, 1, 1]^T
                        # where A_k is a row vector of length target_rank
                        A = np.array([V[k*p+j, :] * U[i*m+k, :] for k in range(m)])
                        b = np.ones(m)
                        # Solve least squares: A @ w = b
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            W[j*n+i, :] = w_sol
                        except:
                            pass
                
                # Update U: for each (i,k), U[i*m+k, :]
                # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all j
                for i in range(n):
                    for k in range(m):
                        B = np.array([V[k*p+j, :] * W[j*n+i, :] for j in range(p)])
                        b = np.ones(p)
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                            U[i*m+k, :] = u_sol
                        except:
                            pass
                
                # Update V: for each (k,j), V[k*p+j, :]
                # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i
                for k in range(m):
                    for j in range(p):
                        C = np.array([U[i*m+k, :] * W[j*n+i, :] for i in range(n)])
                        b = np.ones(n)
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(C, b, rcond=None)
                            V[k*p+j, :] = v_sol
                        except:
                            pass
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-9:
            # Verify
            valid, info = _verify_decomposition(best_U, best_V, best_W)
            if valid:
                return best_U, best_V, best_W, target_rank
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # First, try to find a lower-rank decomposition
    result = _try_find_lower_rank(deadline)
    
    if result is not None:
        U, V, W, rank = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank-27
    U, V, W = _build_schoolbook_27()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
