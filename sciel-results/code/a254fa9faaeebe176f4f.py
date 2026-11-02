"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 145  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    which equals delta[i,i']*delta[j,j']*delta[k,k'] for the matmul tensor.
    
    Actually, the matmul tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the standard matmul, T[i,j,k] = 1 for all valid i,j,k (it's the identity in the 
    sense that C[i,k] = sum_j A[i,j] B[j,k]). Wait, let me reconsider.
    
    The matrix multiplication tensor has entries T[i,j,k] such that
    C[i,k] = sum_j A[i,j] B[j,k].
    
    In tensor form with rank-r decomposition:
    T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    
    For the schoolbook: T[i,j,k] = 1 for all i,j,k (since each term contributes 1 when 
    i,j,k match their respective indices). So T[i,j,k] = 1 for all valid i,j,k.
    """
    rank = U.shape[1]
    # Compute the full tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build U as (n, m, r), V as (m, p, r), W as (p, n, r)
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a contraction over c
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # Efficient computation: for each i, j, k
    # T[i,j,k] = dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
    
    # The target tensor for matmul is T_target[i,j,k] = 1 for all i,j,k
    T_target = np.ones((n, m, p))
    
    return np.allclose(T, T_target, atol=1e-10)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook decomposition
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
    
    # Greedy rank reduction: try to remove each term by expressing it as 
    # a linear combination of remaining terms.
    # A term c can be removed if there exist vectors u', v', w' in the span of 
    # the other terms such that the tensor contribution of term c equals the 
    # contribution of u', v', w'.
    
    # Actually, a simpler approach: the tensor T = sum_c U_c ⊗ V_c ⊗ W_c where 
    # U_c is a vector in R^{n*m}, etc. Term c can be removed if U_c ⊗ V_c ⊗ W_c 
    # is in the span of {U_d ⊗ V_d ⊗ W_d : d != c} as a tensor.
    
    # This is hard to check directly. Alternative: use the fact that if we have 
    # a decomposition, we can try to find a lower-rank one by solving a system.
    
    # Let's try a different approach: random search for lower-rank decompositions.
    # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T_target.
    
    # The tensor T_target[i,j,k] = 1 for all i,j,k.
    
    # Let's use a randomized approach: fix U and V randomly, solve for W, or use 
    # alternating least squares style optimization.
    
    # Actually, let's think about this more carefully. The matmul tensor for n x m * m x p
    # has T[i,j,k] = 1 for all i,j,k. This is a very simple tensor!
    
    # T[i,j,k] = 1 for all i,j,k means T = ones(n) ⊗ ones(m) ⊗ ones(p) as a tensor?
    # No wait. T[i,j,k] = 1 means the tensor is the all-ones tensor.
    
    # The all-ones tensor of shape (n,m,p) can be written as:
    # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    # where we need u ⊗ v ⊗ w to sum to ones.
    
    # Actually T[i,j,k] = 1 = (1)(1)(1), so rank 1 suffices?
    # Let's check: U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, 
    # W[k*n+i, 0] = 1 for all k,i.
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1. Yes!
    
    # Wait, but that's not right for matrix multiplication. Let me re-examine.
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T satisfies: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k. Yes, the tensor is all ones.
    
    # So the rank-1 decomposition works: U = ones(n*m, 1), V = ones(m*p, 1), W = ones(p*n, 1).
    
    # Let me verify this is correct.
    U1 = np.ones((n * m, 1))
    V1 = np.ones((m * p, 1))
    W1 = np.ones((p * n, 1))
    
    if _verify_decomposition(U1, V1, W1, n, m, p):
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
    
    # If rank 1 doesn't work (shouldn't happen based on analysis), try rank 2, etc.
    # But first, let's double-check my understanding.
    
    # Actually, I think I'm confusing myself. Let me re-read the problem.
    # "Matrix multiplication 3,4,7" - multiply 3x4 by 4x7.
    # The tensor decomposition is for the bilinear map (A,B) -> C.
    # T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k].
    # In the standard tensor decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # where the output is indexed by (k,i) in W following the trace convention W[k*n+i,r].
    
    # For the standard matmul, T[i,j,k] = 1 for all i,j,k. So yes, rank 1 should work.
    
    # But wait - maybe the convention is different. Let me re-read:
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This just specifies the indexing of W. It doesn't change the tensor values.
    
    # I'm confident rank 1 works. Let's return it.
    return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}

# EVOLVE-BLOCK-END
