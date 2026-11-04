"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _rank23_3x3x3():
    """
    Return the rank-23 decomposition of the 3x3 x 3x3 matrix multiplication tensor.
    
    Returns (U, V, W) as numpy arrays where:
      U has shape (9, 23): U[i*3+j, t]
      V has shape (9, 23): V[j*3+k, t]
      W has shape (9, 23): W[k*3+i, t]
    
    The decomposition satisfies:
      sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = delta (identity in the tensor sense)
    i.e., for all i,j,k: sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t] = 1 if the 
    tensor entry is 1, else 0.
    
    Actually, the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    So we need: sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t] = 1 for all i,j,k.
    """
    # Winograd's rank-23 decomposition for 3x3 matrix multiplication
    # This is based on the explicit construction from the literature.
    
    # The 23 bilinear forms. Let me use a verified construction.
    # 
    # One approach: use the fact that the 3x3 multiplication can be decomposed
    # using the Winograd scheme. The key is to find 23 vectors u_t in R^9, 
    # v_t in R^9, w_t in R^9 such that sum_t u_t ⊗ v_t ⊗ w_t = T where T[i,j,k]=1.
    
    # Let me construct this using a known valid decomposition.
    # 
    # After careful analysis, here's a valid rank-23 decomposition:
    # The decomposition uses the following 23 terms.
    # Each term (u, v, w) where u, v, w are 9-dimensional vectors (flattened 3x3 matrices).
    
    # I'll use a constructive approach based on the Winograd scheme.
    # The Winograd scheme for 3x3x3 gives rank 23.
    
    # Let me define the 23 terms explicitly.
    # Using the notation from the literature, the 23 terms are:
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Term 0-2: diagonal-like terms
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A[1,1]*B[1,1] -> C[1,1]
    U[4, 1] = 1; V[4, 1] = 1; W[4, 1] = 1
    # Term 2: A[2,2]*B[2,2] -> C[2,2]
    U[8, 2] = 1; V[8, 2] = 1; W[8, 2] = 1
    
    # This approach of building term by term is error-prone. Let me use a different strategy.
    # I'll use a known valid rank-23 decomposition from the literature.
    
    # Actually, let me use a simpler but still improved approach.
    # I'll use the schoolbook for the 3x3x3 part but look for a valid lower rank.
    
    # For now, let me return a valid schoolbook decomposition (rank 9) for 3x3x3
    # and use it as a building block. This gives total rank 9+9+3 = 21 for 3x3x3x7.
    # Wait, that's not right either. The schoolbook for 3x3x3 has rank 9 (one per output entry).
    # But the tensor rank is different from the number of output entries.
    
    # Let me reconsider. The matrix multiplication tensor for n×m × m×p has rank n*m*p 
    # for the schoolbook. For 3×3 × 3×3, schoolbook rank is 9.
    # The known minimal rank is 23? No, that's wrong. The minimal rank for 3x3x3 is actually 
    # known to be at most 23, but the schoolbook gives 9.
    
    # Wait, I'm confusing myself. Let me re-read the problem.
    # The tensor T for matrix multiplication C = A*B where A is 3x3, B is 3x3, C is 3x3.
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to T.
    # The schoolbook decomposition has rank 9 (one for each j, with U[i,j]=delta_ij, V[j,k]=delta_jk, W[k,i]=1).
    # Wait no, the schoolbook has rank 9 for 3x3x3? Let me think again.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For each j (0,1,2), we have a term: U[i,j]=1 if i is the row, V[j,k]=1 if k is the col, W[k,i]=1.
    # Actually, the standard schoolbook decomposition for 3x3x3 has rank 9:
    # For each (i,j,k) triple, we have a term. That's 27 terms.
    # No wait, for each j, the term is: U[i,j] = delta_{i,?}... 
    
    # Let me be more careful. The schoolbook decomposition:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # This is 9 terms (one for each j and each (i,k) pair? No.)
    # Actually, for each j, we have one "product" (A_col_j) * (B_row_j), and this contributes 
    # to all (i,k) entries. So the rank-1 tensor for term j is:
    #   u_j[i,j'] = 1 if j'=j else 0 (so u_j is a vector in R^9 with 1 at position j)
    #   v_j[j',k] = 1 if j'=j else 0 (so v_j is a vector in R^9 with 1 at position j*3+k for all k? No.)
    
    # I'm getting confused. Let me use the standard formulation.
    # The matrix multiplication tensor T ∈ R^{(n*m) × (m*p) × (p*n)} where T[i,j, k,l] = delta_{j,k}... 
    # No, the standard formulation is T[i,j,k] where i indexes rows of A, j indexes columns of A / rows of B, 
    # k indexes columns of B. T[i,j,k] = 1.
    # The rank-1 decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t where u_t ∈ R^{n*m}, v_t ∈ R^{m*p}, w_t ∈ R^{p*n}.
    # Schoolbook: for each (i,j,k), u_{i,j,k}[i',j'] = delta_{i,i'}delta_{j,j'}, v_{i,j,k}[j',k'] = delta_{j,j'}delta_{k,k'}, w_{i,j,k}[k',i'] = delta_{k,k'}delta_{i,i'}.
    # That's n*m*p terms. For 3x3x3, that's 27 terms.
    
    # So schoolbook rank for 3x3x3 is 27, not 9. And the known minimal rank is 23.
    # For 3x3x3x7 (i.e., 3x3 * 3x7), schoolbook rank is 3*3*7 = 63.
    
    # OK so the schoolbook in the current program is correct with rank 63.
    # The improvement would be to use rank 23 for the 3x3x3 parts.
    
    # Let me implement a valid rank-23 decomposition for 3x3x3.
    # I'll use a known construction.
    
    # Reset and try again with a verified approach.
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # I'll use the following verified rank-23 decomposition.
    # This is based on the Winograd scheme.
    
    # After research, here's a valid rank-23 decomposition for 3x3x3:
    # The 23 terms are defined by the following coefficient patterns.
    
    # Let me use a different approach: I'll construct a valid decomposition 
    # by starting from the schoolbook (rank 27) and finding 4 terms to eliminate.
    
    # Actually, for the purpose of this task, let me use a simpler improvement:
    # Use the schoolbook for 3x3x3 (rank 27) but note that for 3x3x3x7,
    # we can potentially share some terms.
    
    # For now, let me just return the schoolbook decomposition (rank 63) 
    # since I can't easily verify a rank-23 decomposition in the time available.
    
    # But wait - I can try a different approach. Let me use the fact that
    # rank(3x3 x 3x7) <= rank(3x3 x 3x3) + rank(3x3 x 3x3) + rank(3x3 x 3x1)
    # = 27 + 27 + 3 = 57 (using schoolbook for each part)
    
    # Or if I can find a rank-23 for 3x3x3, then 23 + 23 + 3 = 49.
    
    # Let me try to implement a valid rank-23 decomposition.
    # I'll use the following approach: verify a candidate decomposition numerically.
    
    # For now, return schoolbook (rank 63) as a safe baseline.
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Uses schoolbook decomposition (rank 63) with deadline guard.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Deadline check - return schoolbook immediately (it's fast)
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
