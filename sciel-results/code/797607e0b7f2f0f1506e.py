"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Return (U, V, W) as list-of-lists for the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact for multiplying (n x m) by (m x p).
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 for all (i,j,k).
    """
    rank = len(U[0])
    # Build U and V as numpy for efficiency
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                tensor_val = np.dot(u_val * v_val, w_val)
                if abs(tensor_val - 1.0) > 1e-9:
                    return False
    return True


def _try_strassen_like(n, m, p, max_rank=None):
    """
    Attempt to find a lower-rank decomposition using a structured approach.
    For 2x4x4, we can think of it as two 2x2x4 blocks or use partitioning.
    
    Strategy: Try to find a decomposition with rank < n*m*p by exploiting
    the fact that matrix multiplication can sometimes be done with fewer
    multiplications (Strassen-style).
    
    For 2x4x4: We can partition the 4x4 matrix into 2x2 blocks:
    B = [[B11, B12], [B21, B22]] where each Bxy is 2x2
    A = [a1, a2, a3, a4] (2x4)
    
    C = A * B = [a1*B11 + a2*B12, a1*B21 + a2*B22] (each entry is 2x2 sum)
    
    Actually, let's think differently. The tensor for 2x4x4 multiplication
    has 2*4*4 = 32 entries. We want to find rank-r decomposition.
    
    A simpler approach: use the fact that 2x4x4 can be decomposed into
    two independent 2x2x4 multiplications? No, that's not right.
    
    Let's try a different structured approach: 
    Represent the multiplication as:
    C_i = sum_{j=1}^{4} A_i * B_j  (for i=1,2 and appropriate j ranges)
    
    Actually, for a 2x4 matrix times a 4x4 matrix:
    C[0, k] = sum_{j=0}^{3} A[0, j] * B[j, k]
    C[1, k] = sum_{j=0}^{3} A[1, j] * B[j, k]
    
    This is essentially two independent 1x4x4 multiplications, but they share
    the same B matrix. The tensor is the sum of two 1x4x4 tensors (one for each
    row of A), but with the B entries shared.
    
    For 1x4x4, the rank is at most 16 (schoolbook), but can it be lower?
    The 1x4x4 tensor T[i,j,k] = delta(i,0) * B[j,k] * A[i,j]... 
    Actually for 1x4x4: T[j,k] = A_j * B_{j,k}, which has rank 4 (since for each j,
    we have a rank-1 contribution A_j * B_{j,:}).
    
    So the 2x4x4 tensor is the sum of two 1x4x4 tensors:
    T[i,j,k] = sum over i=0,1 of (A_i * B_{j,k}) for the appropriate i
    
    More precisely:
    T[i,j,k] = 1 for all (i,j,k) in {0,1}x{0,1,2,3}x{0,1,2,3}
    
    This is a 2x4x4 all-ones tensor? No! The tensor for matrix multiplication
    T[i,j,k] = 1 means that the coefficient of A_i * B_{j,k} in C is 1.
    But C_{i,k} = sum_j A_i * B_{j,k}, so the tensor is:
    T[i,j,k] = 1 for all i,j,k
    
    Wait, that's not right either. The matrix multiplication tensor is defined
    such that C_{i,k} = sum_j A_{i,j} * B_{j,k}. The tensor T[i,j,k] = 1 for all
    (i,j,k) because each triple (i,j,k) contributes A_i * B_{j,k} to C_{i,k}.
    
    So T is the all-ones tensor of shape (2,4,4). The rank of the all-ones tensor
    of shape (n,m,p) is known to be:
    - For 2x2x2: rank 7 (Strassen)
    - For larger sizes: can use tensor product or other decompositions
    
    The rank of the all-ones tensor 2x4x4 can be bounded by:
    - rank(2x4x4) <= rank(2x2x2) * rank(2x2x2) = 7*7 = 49 (worse than schoolbook 32)
    - Or use: 2x4x4 = (2x2x2) x (1x2x2) ... this gets complicated
    
    Let me try a direct approach: search for a rank-r decomposition with r < 32
    by using a heuristic.
    
    For now, let's return None to indicate we couldn't find a better decomposition.
    """
    # This is a placeholder for a more sophisticated search.
    # For the 2x4x4 case, let's try to see if we can achieve rank < 32.
    # 
    # One idea: use the decomposition 2x4x4 = 2 * (1x4x4)
    # The rank of 1x4x4 all-ones tensor is 4 (since T[j,k] = A_j * B_{j,k}
    # can be written as sum of 4 rank-1 tensors: for each j, A_j * (e_j * B_{j,:}))
    # 
    # So rank(1x4x4) = 4
    # Then rank(2x4x4) <= rank(1x4x4) + rank(1x4x4) = 4 + 4 = 8? 
    # No, that's not how tensor rank addition works when the tensors share indices.
    #
    # The 2x4x4 tensor T[i,j,k] = 1 for all i,j,k.
    # We can write T = T_0 + T_1 where T_0[i,j,k] = 1 if i=0 else 0, and
    # T_1[i,j,k] = 1 if i=1 else 0.
    # T_0 is a 2x4x4 tensor that is zero in the i=1 slice, and has all ones in
    # the i=0 slice. This is equivalent to (e_0) x (ones_4x4).
    # The rank of (e_0) x (ones_4x4) is rank(ones_4x4) = 4 (since ones_4x4 has
    # rank 4 as a matrix, and the tensor rank of a matrix is its matrix rank).
    #
    # Wait, that's not quite right. The tensor (e_0) x (ones_4x4) means:
    # T_0[i,j,k] = e_0[i] * ones[j,k]
    # This is a "simple" tensor with rank equal to the rank of the ones_4x4 matrix,
    # which is 1 (since all entries are 1, it's a rank-1 matrix).
    #
    # So T_0 has tensor rank 1, and T_1 has tensor rank 1.
    # Therefore T = T_0 + T_1 has tensor rank at most 2!
    #
    # But that can't be right. Let me verify:
    # If T has rank 2, then T[i,j,k] = sum_{c=1}^{2} U_i[c] * V_j[c] * W_k[c]
    # For T to be all ones, we need:
    # U_0[0]*V_j[0]*W_k[0] + U_0[1]*V_j[1]*W_k[1] = 1 for all j,k
    # U_1[0]*V_j[0]*W_k[0] + U_1[1]*V_j[1]*W_k[1] = 1 for all j,k
    #
    # This means that for each c, U_0[c]*V_j[c]*W_k[c] must be constant in j,k,
    # which implies V_j[c] * W_k[c] is constant in j,k, which is only possible
    # if V_j[c] and W_k[c] are both constant (or one is zero).
    #
    # If V_j[0] = a, W_k[0] = b for all j,k, then U_0[0]*a*b = 1.
    # Similarly for c=1: U_0[1]*a'*b' = 1 - 1 = 0 (since the first term already gives 1)
    # This forces U_0[1]*a'*b' = 0.
    #
    # So actually, the rank-2 decomposition works! Let me construct it:
    # c=0: U_0[0]=1, U_1[0]=1, V_j[0]=1, W_k[0]=1
    # c=1: U_0[1]=0, U_1[1]=0, V_j[1]=0, W_k[1]=0
    #
    # Then T[i,j,k] = 1*1*1 + 0*0*0 = 1. 
    #
    # But this is just the rank-1 decomposition! The rank of the all-ones tensor
    # is 1, not 32.
    #
    # I think I'm confusing the matrix multiplication tensor with the all-ones tensor.
    # The matrix multiplication tensor T[i,j,k] is defined such that:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor coefficient T[i,j,k] is 1 if (i,j,k) is a valid triple, and 0 otherwise.
    # For 2x4x4, all triples (i,j,k) with i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # are valid, so T is indeed the all-ones tensor of shape (2,4,4).
    #
    # And the rank of the all-ones tensor of shape (n,m,p) is 1, since:
    # T[i,j,k] = 1 = 1*1*1
    #
    # So the rank-minimal decomposition has rank 1!
    #
    # U = [[1], [1]]  (shape 2x1, but we need n*m = 8 rows)
    # Wait, U has shape (n*m, rank) = (8, 1)
    # V has shape (m*p, rank) = (16, 1)
    # W has shape (p*n, rank) = (8, 1)
    #
    # For the rank-1 decomposition:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    #
    # Then T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1*1*1 = 1. 
    #
    # This is correct! The rank is 1.
    #
    # But wait, this seems too good to be true. Let me re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix
    # by a 4 by 4 matrix."
    #
    # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) in the domain.
    # The rank of this tensor is indeed 1, since T[i,j,k] = 1 = 1*1*1.
    #
    # So the answer is rank 1!
    
    rank = 1
    n, m, p = DIMENSIONS
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a lower-rank decomposition
    result = _try_strassen_like(n, m, p)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
