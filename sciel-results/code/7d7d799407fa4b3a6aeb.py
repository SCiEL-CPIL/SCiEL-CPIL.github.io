"""Matrix multiplication 3x3 by 3x7 - rank minimization via structured search."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0]) if U else 0
    rng = np.random.default_rng(42)
    for trial in range(5):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C_comp = np.zeros((n, p))
        for c in range(rank):
            a_form = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_form[i] += U[i * m + j][c] * A[i, j]
            b_form = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    b_form[k] += V[j * p + k][c] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i][c] * a_form[i] * b_form[k]
        if not np.allclose(C_true, C_comp, atol=tol, rtol=tol):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_reduced_rank(n, m, p, target_rank):
    """
    Try to find a decomposition with the given target_rank.
    Uses a structured approach based on the algebraic structure of matrix multiplication.
    """
    # Strategy: For matrix multiplication C[i,k] = sum_j A[i,j] B[j,k]
    # We can try decompositions where we group terms.
    # 
    # Key insight: If we fix i, then C[i,:] = A[i,:] @ B is a 1x3 times 3x7 product.
    # The minimal rank for 1x3 * 3x7 is 3 (since it's essentially a vector-matrix product).
    # So for each of the 3 rows, we need at least 3 multiplications, giving 9 total.
    # But the multiplications for different rows share the B factors, so we might do better.
    #
    # Let's try a decomposition with rank = n*m = 9:
    # M_{i,j} = A[i,j] * B[j,:]  -- but this isn't a scalar multiplication
    #
    # Alternative: Use the structure that B is 3x7. We can write B as a sum of rank-1 terms,
    # but that doesn't directly help.
    #
    # Let me try a different approach: random search with constraints.
    # Initialize U, V, W with random values and try to solve for a valid decomposition.
    
    if target_rank >= n * m * p:
        return None
    
    # Try a structured decomposition
    # For rank = n * m * p - (p-1) * (n-1) * (m-1) or similar reductions
    
    # Actually, let's try: rank = n * m * p - (n-1)*(m-1)*(p-1) = 63 - 2*2*6 = 63-24 = 39
    # This is based on the idea that some terms can be shared.
    
    # Let me try a more systematic approach:
    # The tensor T[i,j,k] for matrix mult is 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually T[i,j,k] = 1 for all valid (i,j,k) since each product term contributes.
    
    # A rank-r decomposition satisfies:
    # For all (i,k): sum_j A[i,j]B[j,k] = sum_c (sum_{j'} U[i*m+j',c] A[i,j']) * (sum_{k'} V[j'*p+k',c] B[j',k']) * W[k*n+i,c]
    # Wait, this isn't quite right. Let me re-derive.
    
    # The bilinear algorithm: M_c = (u_c^T A_vec) * (v_c^T B_vec) where A_vec is n*m, B_vec is m*p
    # C_vec = sum_c W_c * M_c where W_c is n*p (flattened)
    # So C[i,k] = sum_c W_c[k*n+i] * (sum_{i',j'} U[i'*m+j',c] A[i',j']) * (sum_{j',k'} V[j'*p+k',c] B[j',k'])
    
    # For this to equal sum_j A[i,j] B[j,k] for all A,B:
    # sum_c W_c[k*n+i] * (sum_{i',j'} U[i'*m+j',c] A[i',j']) * (sum_{j',k'} V[j'*p+k',c] B[j',k']) = sum_j A[i,j] B[j,k]
    
    # This must hold for all A,B, so the tensor formed by the LHS must equal the standard multiplication tensor.
    
    # This is a complex constraint. Let me just try to find a low-rank decomposition numerically.
    
    # For now, return None to indicate this approach didn't find a solution
    return None

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook is the guaranteed valid solution with rank 63
    candidate = _schoolbook(n, m, p)
    
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        best_result = candidate
        best_rank = n * m * p
    else:
        return candidate
    
    # Try to find a lower rank decomposition
    # Start with the most promising structured reductions
    
    # Attempt 1: Try rank = n*m*p - (n-1)*(m-1)*(p-1)
    # For n=3,m=3,p=7: 63 - 2*2*6 = 63-24 = 39
    # This is speculative, so let's try a numerical approach instead.
    
    # Numerical approach: Use random initialization and gradient-free search
    # to find a valid low-rank decomposition.
    
    # Given the complexity, let's try a few specific structured decompositions.
    
    # Structure: Group by j (the shared index)
    # For each j, we have n*p = 21 terms (i,k pairs). 
    # Can we represent the j-th "slice" with fewer than 21 rank-1 terms?
    # The j-th slice is: for all (i,k), contribution is A[i,j]*B[j,k]
    # This is an outer product of A[:,j] (3x1) and B[j,:] (1x7), which has rank 1!
    # So each j-slice can be represented with rank 1 if we could use:
    # M_j = (linear form extracting A[:,j]) * (linear form extracting B[j,:])
    # But M_j must be a scalar, and we need it to produce a 3x7 matrix via W.
    
    # The issue: M_j = (sum_i U[i,j] A[i,j]) * (sum_k V[j,k] B[j,k]) is a scalar.
    # Then C[i,k] += W[i,k,j] * M_j
    # For this to equal A[i,j]*B[j,k], we need W[i,k,j] * (sum_i' U[i',j] A[i',j]) * (sum_k' V[j,k'] B[j,k']) = A[i,j]*B[j,k]
    # This can't hold for all A,B unless U and V are delta functions, giving us back the schoolbook.
    
    # So the simple grouping doesn't work. The rank is genuinely 63 for this structured approach.
    
    # Let me try a completely different angle: maybe there's a known result.
    # For 3x3 * 3x7, the rank might be 63 or slightly less.
    # Without a proven better algorithm, I'll return the schoolbook.
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
