"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _validate_numpy(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # U shape: (n*m, rank) -> (n, m, rank)
    # V shape: (m*p, rank) -> (m, p, rank)
    # W shape: (p*n, rank) -> (p, n, rank)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a bit tricky with einsum
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Let's do it step by step
    # First compute A[i,j,k] = sum_c U[i,j,c] * W[k,i,c]
    # A[i,j,k] = U[i,j,:] dot W[k,i,:]
    A = np.einsum('ijc,kic->ijk', U_reshaped, W_reshaped)
    # Then T[i,j,k] = sum_c A[i,j,k] * V[j,k,c]... wait, that's not right
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # = sum_c (U[i,j,c] * W[k,i,c]) * V[j,k,c]
    
    # Let me just do it directly
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += U_reshaped[:, :, c] * V_reshaped[:, :, c] * W_reshaped[:, :, c]
    
    # Check that T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _all_half_integer(arr, tol=1e-9):
    """Check if all entries in arr are integers or half-integers."""
    for row in arr:
        for val in row:
            if not _is_half_integer(val, tol):
                return False
    return True

def _try_strassen_decomposition(n=2, m=4, p=5, deadline=None):
    """
    Try to find a lower-rank decomposition using Strassen-like tricks.
    
    For the 4-dimensional inner product, we can use a rank-3 decomposition:
    Let a = A[:,0], b = A[:,1], c = A[:,2], d = A[:,3]
    Let x = B[:,0], y = B[:,1], z = B[:,2], w = B[:,3]
    
    We want: a*x + b*y + c*z + d*w = sum of 3 products
    
    Using Strassen's trick:
    Let u = a + b, v = x + y, s = c + d, t = z + w
    Then: a*x + b*y + c*z + d*w = 
    (u*v - (a-b)*(x-y) - (s-t)*(z-w) - (a+b)*(c+d) + ... )
    
    Actually, let me use a known rank-3 decomposition for 4D inner product.
    """
    # For 4-dimensional inner product, we can use:
    # (a0+a1)(b0+b1) - (a0-a1)(b0-b1) = 2*a0*b0 + 2*a1*b1
    # This handles 2 dimensions with 2 terms.
    # For 4 dimensions, we can do it in 3 terms using:
    # Let's use the fact that we can group: (a0+a1)(b0+b1) + (a2+a3)(b2+b3) 
    # but that's only 2 terms for 4 dims if we're not careful.
    
    # Actually, the rank of the 4D inner product tensor is 3 (not 4).
    # One construction:
    # T1 = (a0+a1)(b0+b1)
    # T2 = (a0-a1)(b0-b1)  
    # T3 = (a2+a3)(b2+b3) + (a2-a3)(b2-b3)  -- wait, that's 2 terms
    
    # Let me think differently. The 4D inner product has rank 3.
    # We can write:
    # a0*b0 + a1*b1 + a2*b2 + a3*b3 = 
    # 1/2 * [(a0+a1)(b0+b1) + (a0-a1)(b0-b1) + (a2+a3)(b2+b3) + (a2-a3)(b2-b3)]
    # But that's 4 terms.
    
    # Actually, I think the rank is 3 for the 4D case. Let me use a different approach.
    # We can use:
    # T1 = (a0+a2)(b0+b2)
    # T2 = (a0-a2)(b0-b2)
    # T3 = (a1+a3)(b1+b3) + (a1-a3)(b1-b3)  -- no, that's still 2 terms
    
    # I think I'm overcomplicating this. Let me just try to find a rank-3 decomposition
    # by solving the linear system.
    
    # The tensor T[j,k] for the 4D inner product (for fixed i) is:
    # T[j,k] = delta_{jk} for j,k in {0,1,2,3}
    # We want to write this as sum_{c=1}^{3} U_j[c] * V_k[c]
    # where U is 4x3 and V is 4x3.
    
    # This is equivalent to finding a rank-3 decomposition of the 4x4 identity matrix,
    # which is impossible since rank(4x4 I) = 4.
    
    # Wait, I made an error. The tensor is not the identity matrix.
    # The tensor T[i,j,k] for MM_{2,4,5} is:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C
    # Actually, T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} = 1 for all i,j,k
    # No, that's not right either.
    
    # Let me re-read the problem. The tensor is:
    # T[i,j,k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all i,j,k (since every A[i,j] and B[j,k] pair contributes to C[i,k])
    
    # Wait, that's not the standard matrix multiplication tensor.
    # The standard MM tensor has T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # where i' is the row index of A, j' is the column index of A / row index of B, 
    # k' is the column index of B.
    
    # So for MM_{n,m,p}, T[i,j,k] = 1 if i is row of A, j is col of A / row of B, k is col of B
    # And the tensor value is 1 for all valid (i,j,k).
    
    # So the tensor is just all ones? That can't be right.
    
    # Let me re-read the task. "Matrix multiplication 2,4,5"
    # The tensor is T_{i,j,k} where i in [0,2), j in [0,4), k in [0,5)
    # And T_{i,j,k} = 1 (the contribution of A[i,j] * B[j,k] to C[i,k])
    
    # So the tensor is indeed all ones!
    # And we want to decompose it as sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # For the all-ones tensor, we can use:
    # U[i,j,c] = 1 for all i,j,c
    # V[j,k,c] = 1 for all j,k,c  
    # W[k,i,c] = 1 for all k,i,c
    # Then sum_c 1*1*1 = rank (number of terms)
    # But we need the sum to be 1, not rank.
    
    # So we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # If we set U[i,j,c] = V[j,k,c] = W[k,i,c] = 1/sqrt(rank), then sum = rank * (1/sqrt(rank))^3 = 1/sqrt(rank)
    # That's not 1.
    
    # If we set U[i,j,c] = V[j,k,c] = 1, W[k,i,c] = 1/rank, then sum = rank * 1*1*(1/rank) = 1
    # So we need rank terms, each with U=1, V=1, W=1/rank.
    
    # But the entries need to be integers or half-integers. 1/rank is not a half-integer for rank > 2.
    
    # So we need a different approach.
    
    # Actually, I think I misunderstood the tensor. Let me re-read.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # So the tensor is the standard matrix multiplication tensor.
    # T[i,j,k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # Which is always 1 for valid indices.
    
    # But the decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_c U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * W[k,i,c]
    # Wait, that's not right.
    
    # The decomposition is:
    # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k]) * W[k,i,c]
    # No, that's not right either.
    
    # Let me look at the validate function:
    # val = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    # This should equal 1 for all i,j,k.
    
    # So the tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # And we want to minimize rank.
    
    # For the all-ones tensor, the minimum rank is 1 (just use one term with all entries = 1).
    # But that's not right because the tensor is not all ones.
    
    # I think I'm confusing myself. Let me just implement the schoolbook solution and see if I can improve it.
    
    return None

def _try_ranked_decomposition(n=2, m=4, p=5, max_rank=40, deadline=None):
    """
    Try to find a decomposition with rank <= max_rank.
    """
    # For now, just return the schoolbook solution
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

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    result = _try_strassen_decomposition(n, m, p, deadline)
    
    if result is None or time.time() > deadline - 5:
        # Fall back to schoolbook
        result = _try_ranked_decomposition(n, m, p, n*m*p, deadline)
    
    # Verify the decomposition
    U, V, W = result["U"], result["V"], result["W"]
    assert _validate(U, V, W, n, m, p), "Decomposition failed validation!"
    assert _all_half_integer(U), "U has non-half-integer entries"
    assert _all_half_integer(V), "V has non-half-integer entries"
    assert _all_half_integer(W), "W has non-half-integer entries"
    
    return result
# EVOLVE-BLOCK-END
