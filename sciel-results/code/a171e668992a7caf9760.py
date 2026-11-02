"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import itertools

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a decomposition with rank < 63
    # Use structured approaches based on the tensor structure
    
    # Strategy 1: Try rank = 3 * p = 21 (one "block" per j, each of rank p)
    # For each j, we have T[:,j,:] = A[:,j] ⊗ B[j,:]
    # We can write this as sum over k of A[:,j] ⊗ e_k ⊗ B[j,k]
    # But that's rank p per j, total 3*p = 21
    
    # Let's try a different approach: exploit that B[j,:] has 7 entries
    # We can try to factor the 7-dimensional space more efficiently
    
    # Try various candidate ranks
    for candidate_rank in [21, 24, 27, 30, 33, 36, 42, 48, 54]:
        if time.time() > deadline - 20:
            break
        result = _try_structured_rank(n, m, p, candidate_rank)
        if result is not None:
            return result
    
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

def _try_structured_rank(n, m, p, rank):
    """
    Try to find a rank-`rank` decomposition using structured patterns.
    
    The tensor T[i,j,k] = A[i,j] * B[j,k]
    We want: T = sum_c U_c ⊗ V_c ⊗ W_c
    
    where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
    
    Key insight: For each j, the "slice" T[:,j,:] is a 3x7 matrix of rank ≤ 3.
    We can try to share structure across different j values.
    """
    # Approach: Use the fact that for fixed i and k, T[i,j,k] = A[i,j] * B[j,k]
    # This is a bilinear form in j
    
    # Let's try a "sum over j" approach with rank-1 terms in the j dimension
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (but W should depend on k and i)
    
    # Actually, the decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_k V[j,k,c] * B[j,k]) * W[k,i,c]
    # This doesn't quite work directly.
    
    # Let me think differently. The standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    # where the sum over c replaces the sum over j
    
    # For rank r, we need r triples (U_c, V_c, W_c)
    
    # Simple structured approach: use the j index directly
    # For each j, create a term that captures A[:,j] and B[j,:]
    # Term j: U_j[i,j',k] = 1 if j'=j else 0, V_j[j',k] = B[j,k] (but B is variable!)
    
    # The issue is that U, V, W must be fixed, not depend on A and B.
    
    # So we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = A[i,j] * B[j,k] for all A, B
    # This means the tensors must satisfy:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = A[i,j] * B[j,k]
    
    # This is a system of equations. Let's try to find a sparse solution.
    
    # For rank = m * p = 21:
    # Use one term for each (j, k) pair
    # U[i, j', c(j,k)] = delta_{j',j} * A[i,j] ... but A is variable!
    
    # I see the issue. U, V, W must be constant tensors, not depending on A, B.
    
    # So we need: for all A, B:
    # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = A[i,j] * B[j,k]
    
    # This means the coefficient of A[i,j] * B[j,k] must be 1, and all other coefficients 0.
    
    # Let's try rank = m * p = 21 with a structured pattern:
    # For each (j, k), create a term where:
    #   U_c[i, j'] = delta_{j',j} * (some function of i)
    #   V_c[j', k'] = delta_{j',j} * delta_{k',k}
    #   W_c[k'', i'] = delta_{k'',k} * (some function of i')
    
    # But this won't work because we need the product to give A[i,j] * B[j,k]
    
    # Let me try a different approach. Use the fact that:
    # A[i,j] * B[j,k] = sum over some basis
    
    # For rank = n * m * p / something, try sparse patterns.
    
    # Actually, let's just try the schoolbook rank first and see if we can improve.
    
    # Schoolbook: rank = n * m * p = 63
    # Each term corresponds to a (i, j, k) triple
    
    # Can we do better? Let's try rank = 3 * 7 = 21
    # For each (j, k), one term:
    #   U_c[i, j'] = delta_{j',j}
    #   V_c[j', k'] = delta_{j',j} * delta_{k',k}
    #   W_c[k'', i'] = delta_{k'',k}
    
    # Then sum_c U_c[i,j'] * V_c[j',k'] * W_c[k'',i'] 
    # = sum_{j,k} delta_{j',j} * delta_{j',j} * delta_{k',k} * delta_{k'',k}
    # This gives 1 if j'=j and k'=k, which is not what we want.
    
    # I think I need to be more careful. Let me use the correct formulation.
    
    # The matrix multiplication tensor is:
    # T[i,j,k] = e_{ij} ⊗ e_{jk} ⊗ e_{ki} in some sense
    
    # Actually, the standard formulation is:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # As a tensor: T(A,B) = sum_{i,j,k} A[i,j] * B[j,k] * E_{i,k}
    # where E_{i,k} is the (i,k) entry of the identity-like output
    
    # The rank is the minimum r such that we can write:
    # sum_{i,j,k} A[i,j] * B[j,k] * E_{i,k} = sum_c f_c(A) * g_c(B) * h_c
    # where f_c, g_c are bilinear forms and h_c is a matrix
    
    # For rank r, we have r such decompositions.
    
    # Let's try to find a rank-21 decomposition:
    # For each (j, k) pair, define:
    #   f_c(A) = A[i,j] for the specific i ... no, f_c must be a single bilinear form
    
    # I think I'm overcomplicating this. Let me just return None and use schoolbook.
    
    return None
# EVOLVE-BLOCK-END
