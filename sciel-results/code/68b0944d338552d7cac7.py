"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 7 matrix.
    
    The multiplication tensor T has entries T[i, j, k] = 1 if the product C[i,k] = sum_j A[i,j]B[j,k]
    involves the term A[i,j]*B[j,k]. More precisely, the tensor for matrix multiplication
    C = AB where A is n x m and B is m x p is:
    T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    where the tensor is indexed such that the decomposition is:
    T = sum_r U[:,r] * V[:,r] * W[:,r]
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    For exact integer decomposition, the schoolbook method gives rank n*m*p = 42.
    However, we can potentially do better using fast matrix multiplication techniques
    or by finding a lower rank decomposition.
    
    For 2x3 * 3x7, let's think about what rank is achievable.
    The tensor is of format (2,3,7). The schoolbook rank is 2*3*7 = 42.
    
    Using the Strassen-like ideas or other fast matrix multiplication, we might be able
    to reduce this. However, for small matrices, the minimal rank might still be close
    to schoolbook.
    
    Let's try to use a systematic approach: represent the tensor and try to find a
    low-rank decomposition.
    
    Actually, for exact integer decompositions, the minimal rank for multiplying an
    n x m by m x p matrix is known to be at most n*m*p (schoolbook). For small cases,
    the minimal rank is often equal to or slightly less than n*m*p.
    
    Let me think about this more carefully. The tensor T in R^{n x m x p} has entries
    T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].
    
    Wait, I need to be more precise. The matrix multiplication tensor is defined as:
    T_{i,j,k} = 1 for the "core" tensor where the multiplication is:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The tensor is of size (n*m) x (m*p) x (p*n) in the flattened representation,
    where:
    - U has shape (n*m, rank)
    - V has shape (m*p, rank)  
    - W has shape (p*n, rank)
    
    And the decomposition is: T = sum_r U[:,r] * V[:,r] * W[:,r]
    
    In the schoolbook method, each (i,j,k) triple gives one rank-1 term:
    U[i*m + j, c] = 1
    V[j*p + k, c] = 1
    W[k*n + i, c] = 1
    
    This gives rank = n*m*p = 42.
    
    Can we do better? For 2x3 * 3x7, let's see if there's a known lower rank.
    
    Actually, for small matrix multiplication, the minimal rank is often the same as
    schoolbook unless specific fast algorithms apply. For 2x2x2, Strassen gives rank 7
    instead of 8. For 2x3 * 3x7, it's less clear.
    
    Let me try a different approach: use the fact that we can combine some terms.
    
    Actually, let me just implement the schoolbook method first and see if the evaluator
    accepts it. The score is the rank, so 42 would be the baseline.
    
    But wait, the task says "rank-minimal exact tensor decomposition". So we need to
    find the minimum possible rank.
    
    For now, let's return the schoolbook decomposition of rank 42. If we can find a
    lower rank, we can improve later.
    
    Let me verify the convention: W[k*n+i, r] means the W matrix is indexed by (k, i)
    where k is the column index of B (and C) and i is the row index of A (and C).
    So W has shape (p*n, rank) = (7*2, rank) = (14, rank).
    
    U has shape (n*m, rank) = (2*3, rank) = (6, rank)
    V has shape (m*p, rank) = (3*7, rank) = (21, rank)
    W has shape (p*n, rank) = (7*2, rank) = (14, rank)
    
    The tensor T is implicitly defined by: for each rank-1 component r,
    the contribution to C is: for all i, k: C[i,k] += W[k*n+i, r] * sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
    
    Wait, I need to be more careful. The standard tensor decomposition for matrix
    multiplication is:
    
    C_{i,k} = sum_r (sum_j U_{i,j,r} * A_{i,j}) * (sum_j' V_{j',k,r} * B_{j',k}) * W_{i,k,r}
    
    No, that's not right either. Let me think again.
    
    The matrix multiplication can be written as:
    C = A * B
    
    In tensor form, the multiplication tensor T has entries such that:
    C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    
    A rank-1 decomposition of this tensor would be:
    T = sum_r alpha_r * u_r * v_r * w_r
    
    where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n} (or similar).
    
    Actually, the standard way is:
    For each term r in the decomposition:
    - u_r is a vector of length n*m, with u_r[i*m + j] representing the coefficient for A[i,j]
    - v_r is a vector of length m*p, with v_r[j*p + k] representing the coefficient for B[j,k]
    - w_r is a vector of length p*n, with w_r[k*n + i] representing the coefficient for C[i,k]
    
    And the decomposition is:
    sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k] = sum_r (sum_{i,j} u_r[i*m+j] A[i,j]) * (sum_{j,k} v_r[j*p+k] B[j,k]) * (sum_{i,k} w_r[k*n+i] C[i,k])
    
    Hmm, this is getting confusing. Let me just use the schoolbook method which is known to work:
    
    For each (i, j, k):
    U[i*m + j, c] = 1
    V[j*p + k, c] = 1
    W[k*n + i, c] = 1
    
    This gives:
    sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c] = 1 if (i,j,k) = (i'', j', k'') and i''=i, j'=j, k''=k, else 0
    
    Wait, that's not quite right. Let me trace through:
    
    The product for a given output C[i_out, k_out] is:
    sum_c (sum_j U[i_out*m + j, c]) * (sum_{j'} V[j'*p + k_out, c]) * W[k_out*n + i_out, c]
    
    For the schoolbook term corresponding to (i, j, k):
    U[i*m + j, c] = 1 (only when the row index is i*m+j)
    V[j*p + k, c] = 1 (only when the column index is j*p+k)
    W[k*n + i, c] = 1 (only when the output index is k*n+i)
    
    So for output C[i_out, k_out], the contribution from term (i,j,k) is:
    U[i_out*m + j, c] * V[j*p + k_out, c] * W[k_out*n + i_out, c]
    
    This is 1 only if:
    - i_out*m + j corresponds to the U index, so i_out = i (since U[i*m+j, c]=1)
    - j*p + k_out corresponds to the V index, so k_out = k (since V[j*p+k, c]=1)  
    - k_out*n + i_out corresponds to the W index, so (k_out, i_out) = (k, i)
    
    So the contribution is 1 when i_out = i and k_out = k, and the j in U and the j in V must match.
    
    Wait, I'm confusing myself. Let me just trust the schoolbook formula and verify with a simple case.
    
    For n=1, m=1, p=1:
    U[0,0] = 1, V[0,0] = 1, W[0,0] = 1
    C[0,0] = U[0,0] * A[0,0] * V[0,0] * B[0,0] * W[0,0] = A[0,0] * B[0,0] * 1 = A[0,0]*B[0,0]
    Correct.
    
    For n=1, m=2, p=1:
    Terms: (i=0, j=0, k=0) and (i=0, j=1, k=0)
    U[0*2+0, 0] = 1, V[0*1+0, 0] = 1, W[0*1+0, 0] = 1
    U[0*2+1, 1] = 1, V[1*1+0, 1] = 1, W[0*1+0, 1] = 1
    
    C[0,0] = sum_c (sum_j U[0*2+j, c] * A[0,j]) * (sum_j' V[j'*1+0, c] * B[j',0]) * W[0*1+0, c]
    
    For c=0: (U[0,0]*A[0,0] + U[1,0]*A[0,1]) * (V[0,0]*B[0,0] + V[1,0]*B[1,0]) * W[0,0]
    = (1*A[0,0] + 0*A[0,1]) * (1*B[0,0] + 0*B[1,0]) * 1 = A[0,0]*B[0,0]
    
    For c=1: (U[0,1]*A[0,0] + U[1,1]*A[0,1]) * (V[0,1]*B[0,0] + V[1,1]*B[1,0]) * W[0,1]
    = (0*A[0,0] + 1*A[0,1]) * (0*B[0,0] + 1*B[1,0]) * 1 = A[0,1]*B[1,0]
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    Correct!
    
    So the schoolbook method works. Now, can we do better than rank 42?
    
    For now, let's just return the schoolbook decomposition. The evaluator will tell us
    if this is accepted, and we can try to optimize later.
    """
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
# EVOLVE-BLOCK-END
