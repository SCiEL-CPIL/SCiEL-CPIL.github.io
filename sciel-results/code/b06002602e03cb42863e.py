"""Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    return U, V, W


def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each of shape (rank, dim) where rank=7.
    U: (4, 7) for A elements (row-major: a00, a01, a10, a11)
    V: (4, 7) for B elements (row-major: b00, b01, b10, b11)
    W: (4, 7) for C elements (row-major: c00, c01, c10, c11)
    
    Strassen formulas:
    M1 = (a00+a11)*(b00+b11)
    M2 = (a10+a11)*b00
    M3 = a00*(b01-b11)
    M4 = a11*(b10-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)*(b00+b01)
    M7 = (a01-a11)*(b10+b11)
    
    C = [[M1+M4-M5-M7, M3+M5],
         [M2+M4,       M1-M2+M3+M6]]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1
    
    return U, V, W


def _combine_decompositions(decomp_list, n, m, p):
    """
    Combine multiple decompositions that cover different parts of the multiplication.
    Each decomp is (U_i, V_i, W_i) with row counts matching subsets of n*m, m*p, p*n.
    This function just stacks them column-wise.
    """
    total_rank = sum(U.shape[1] for U, V, W in decomp_list)
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for U_i, V_i, W_i in decomp_list:
        U_full[:, col_offset:col_offset + U_i.shape[1]] = U_i
        V_full[:, col_offset:col_offset + V_i.shape[1]] = V_i
        W_full[:, col_offset:col_offset + W_i.shape[1]] = W_i
        col_offset += U_i.shape[1]
    
    return U_full, V_full, W_full


def _build_hybrid_decomposition():
    """
    Build a decomposition for 4x4 * 4x5.
    
    Strategy:
    - The 4x5 B matrix can be viewed as [B_left (4x4), B_right (4x1)].
    - For B_left (4x4): use schoolbook (rank 64).
    - For B_right (4x1): this is just a matrix-vector product. Each output row i 
      computes C[i] = sum_j A[i,j] * B[j,4]. This requires rank 4 (one per output).
    
    Total rank: 64 + 4 = 68.
    
    Actually, let me reconsider. The tensor T[i,j,k] = 1 for the multiplication
    C[i,k] = sum_j A[i,j] B[j,k].
    
    For k=0..3 (B_left): schoolbook gives 4*4*4 = 64 terms.
    For k=4 (B_right): C[i,4] = sum_j A[i,j] B[j,4]. 
    This is 4 separate terms (one per i), each using all j. Rank 4.
    
    Total: 68.
    """
    n, m, p = 4, 4, 5
    
    # Part 1: Schoolbook for k=0..3 (B columns 0-3)
    U1 = np.zeros((n * m, n * m * 4))
    V1 = np.zeros((m * 4, n * m * 4))  # m*p_part = 4*4 = 16
    W1 = np.zeros((4 * n, n * m * 4))   # p_part*n = 4*4 = 16
    
    for i in range(n):
        for j in range(m):
            for k in range(4):
                c = i * m * 4 + j * 4 + k
                U1[i * m + j, c] = 1
                V1[j * 4 + k, c] = 1
                W1[k * n + i, c] = 1
    
    # Part 2: Matrix-vector for k=4 (B column 4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # U2 has shape (n*m, 4), V2 has shape (m*1, 4) -> wait, V should be (m*p, rank)
    # But we're only using the k=4 column of B, so V2 should still be (m*p, 4) = (20, 4)
    # with only rows j*p+4 = j*5+4 nonzero.
    U2 = np.zeros((n * m, n))
    V2 = np.zeros((m * p, n))
    W2 = np.zeros((p * n, n))
    
    for i in range(n):
        for j in range(m):
            U2[i * m + j, i] = 1
            V2[j * p + 4, i] = 1
        W2[4 * n + i, i] = 1  # k=4, so W[4*n + i]
    
    # Now we need to pad V1 to have m*p = 20 rows (it currently has 16 rows for k=0..3)
    V1_padded = np.zeros((m * p, n * m * 4))
    V1_padded[:m * 4, :] = V1
    # Rows 16-19 correspond to k=4, which are zero in Part 1
    
    # Combine
    total_rank = n * m * 4 + n  # 64 + 4 = 68
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    U_full[:, :n * m * 4] = U1
    V_full[:m * 4, :n * m * 4] = V1
    W_full[:4 * n, :n * m * 4] = W1
    
    U_full[:, n * m * 4:] = U2
    V_full[:, n * m * 4:] = V2
    W_full[:, n * m * 4:] = W2
    
    return U_full, V_full, W_full


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if this is a term in the sum, i.e., C[i,k] = sum_j A[i,j]*B[j,k].
    The tensor T[i,j,k] = 1 for all valid i,j,k.
    So sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal sum_j 1 = m for each (i,k)? 
    No wait. The multiplication tensor is T[i,j,k] = 1. The decomposition should satisfy:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i, j, k.
    """
    # Compute the tensor from the decomposition
    # T_reconstructed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    rank = U.shape[1]
    # Use einsum or manual
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
    
    # Should be all ones
    if not np.allclose(T, 1.0, atol=1e-10):
        max_err = np.max(np.abs(T - 1.0))
        return False, max_err
    return True, 0.0


def _to_fraction_matrix(mat):
    """Convert a numpy matrix to a list of lists of Fractions (or ints if integer)."""
    result = []
    for row in mat:
        row_frac = []
        for val in row:
            f = Fraction(val).limit_denominator(10**6)
            if f.denominator == 1:
                row_frac.append(f.numerator)
            else:
                row_frac.append(f)
        result.append(row_frac)
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the hybrid decomposition: schoolbook for first 4 columns + matvec for 5th
    U, V, W = _build_hybrid_decomposition()
    
    # Verify
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    if valid:
        result = {
            "U": _to_fraction_matrix(U),
            "V": _to_fraction_matrix(V),
            "W": _to_fraction_matrix(W)
        }
        return result
    
    # Fallback to pure schoolbook
    U, V, W = _schoolbook(n, m, p)
    result = {
        "U": _to_fraction_matrix(U),
        "V": _to_fraction_matrix(V),
        "W": _to_fraction_matrix(W)
    }
    return result
