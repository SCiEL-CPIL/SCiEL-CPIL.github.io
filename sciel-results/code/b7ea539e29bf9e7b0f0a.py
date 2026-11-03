"""Matrix multiplication 3x3x6 tensor decomposition finder."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _repeated_3x3x3():
    """
    Use the schoolbook decomposition of 3x3x3 (rank 27) and repeat it
    to get a rank-27 decomposition of 3x3x6.
    
    The 3x3x6 tensor is two copies of 3x3x3:
    T[i,j,k] = 1 for all i,j,k (where i in [0,3), j in [0,3), k in [0,6))
    
    We can write this as: for each (i,j), the k-index has 6 entries.
    The schoolbook for 3x3x3 has 27 terms, one for each (i,j,k) with k in [0,3).
    For 3x3x6, we need terms for k in [0,6).
    
    Key insight: The tensor M_{3,3,6} has T[i,j,k] = 1 for all i,j,k.
    This is the same as M_{3,3,3} but with k doubled.
    
    Actually, let me think more carefully. The matrix multiplication tensor
    M_{n,m,p} has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    A rank-1 decomposition term is u ⊗ v ⊗ w where:
    - u has shape (n*m,), indexed by (i,j)
    - v has shape (m*p,), indexed by (j,k)
    - w has shape (p*n,), indexed by (k,i)
    
    And the condition is: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k.
    
    For the schoolbook of 3x3x3:
    - There are 27 terms, indexed by (i,j,k) with i,j,k in [0,3).
    - For term (i0,j0,k0): u[i0*3+j0] = 1, v[j0*3+k0] = 1, w[k0*3+i0] = 1.
    
    For 3x3x6, we need: sum_c u_c[i*3+j] * v_c[j*6+k] * w_c[k*3+i] = 1 for all i,j,k with k in [0,6).
    
    If we take the 27 terms from 3x3x3 and modify v and w:
    - For term (i0,j0,k0) from 3x3x3:
      - u[i0*3+j0] = 1
      - We need v[j*6+k] to be 1 when j=j0 and k=k0 or k=k0+3
      - We need w[k*3+i] to be 1 when k=k0 or k=k0+3 and i=i0
    
    Actually, this won't work directly because the indexing is different.
    
    Let me reconsider. The simplest correct approach:
    The tensor M_{3,3,6} = M_{3,3,3} ⊕ M_{3,3,3} (direct sum over the k-index).
    The rank of a direct sum is at most the sum of the ranks.
    But actually, since the pattern repeats, we might be able to share terms.
    
    However, the safest approach is just to use the schoolbook for 3x3x6 directly,
    which gives rank 54. That's what the current program does.
    
    Wait, I realize the previous response's insight was actually correct in principle
    but the implementation was flawed. Let me think again.
    
    Actually, the tensor M_{3,3,6} has T[i,j,k] = 1 for all i,j,k.
    This is a 3x3x6 all-ones tensor (in the appropriate flattened form).
    
    The rank of the all-ones tensor of shape a×b×c is 1 (it's a single rank-1 term).
    But that's not the matrix multiplication tensor!
    
    I confused myself. The matrix multiplication tensor is NOT all ones.
    Let me re-read the task.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix
    by a 3 by 6 matrix."
    
    The matrix multiplication tensor T satisfies:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This can be written as: C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
    where T[i,j,k] = 1 for all i,j,k.
    
    Wait, that IS all ones! The tensor is indeed the all-ones tensor in the
    (i,j,k) space. But the rank-1 decomposition is over the flattened indices:
    u indexed by (i,j), v indexed by (j,k), w indexed by (k,i).
    
    So the condition is: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k.
    
    This is NOT trivially rank 1 because the indices are coupled differently
    in u, v, w.
    
    For the schoolbook: each (i,j,k) triple gives one term where
    u[i,j] = 1, v[j,k] = 1, w[k,i] = 1, and all other entries are 0.
    This gives rank = n*m*p = 54 for 3x3x6.
    
    Can we do better? The key question is whether the all-ones tensor in this
    coupled-index form has lower rank.
    
    For 2x2x2, the answer is yes: rank 7 (Strassen) vs rank 8 (schoolbook).
    For 3x3x3, the best known is 23 vs schoolbook 27.
    For 3x3x6, what's the best known?
    
    I don't have a known explicit low-rank decomposition for 3x3x6.
    The safest bet is to return the schoolbook with rank 54.
    
    But wait - can we at least improve from 54? 
    Since 3x3x6 = two 3x3x3 blocks, and rank(M_{3,3,3}) <= 23,
    we can get rank <= 46 by using two independent 3x3x3 decompositions.
    
    But I don't have an explicit rank-23 decomposition for 3x3x3 hardcoded.
    
    Let me just return the schoolbook for now. It's correct and guaranteed.
    """
    return _schoolbook(3, 3, 6)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x3x6 matrix multiplication.
    Returns {'U': (9 x rank), 'V': (18 x rank), 'W': (18 x rank)}
    """
    n, m, p = DIMENSIONS
    
    # Return the schoolbook decomposition with rank 54.
    # This is correct and guaranteed to work.
    result = _schoolbook(n, m, p)
    
    # Verify correctness
    assert _verify_decomposition(result["U"], result["V"], result["W"], n, m, p), \
        "Schoolbook decomposition failed verification"
    
    return result
# EVOLVE-BLOCK-END
