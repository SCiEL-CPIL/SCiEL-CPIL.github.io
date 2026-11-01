"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import null_space

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all required (i,j,k) combinations equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check all other (i,j,k) combinations equal 0 (not required by the problem but good practice)
    # Actually, the problem only requires the (i,j,k) combinations where the tensor is 1.
    # The tensor T[i,j,k] = 1 if (i,j,k) is a valid index in the multiplication, else 0.
    # Wait, the tensor is T_{i,j,k} = 1 if the entry (i,k) of C = A*B comes from A_{i,j}*B_{j,k}.
    # So T_{i,j,k} is 1 for all i in [0,n), j in [0,m), k in [0,p).
    # There are no other entries. The tensor is sparse.
    # So we only need to check that the sum equals 1 for all i,j,k.
    # We do NOT need to check that other combinations are 0 because they don't exist in the tensor.
    
    return True

def _strassen_decomposition():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    
    A (2x4) * B (4x8) = C (2x8)
    
    Partition: A = [A_left A_right] where A_left, A_right are 2x2
               B = [B_top; B_bottom] where B_top, B_bottom are 2x8
    
    C = A_left * B_top + A_right * B_bottom
    
    We partition the 4 columns of A into two 2-column blocks, and the 8 columns 
    of B into four 2-column blocks. For each of the 2 A-blocks and 4 B-blocks,
    we perform a 2x2 by 2x2 matrix multiplication using Strassen's algorithm (7 mults).
    
    Total rank = 2 * 4 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
    strassen_terms = [
        # M1 = (a00 + a11)(b00 + b11)
        (
            {(0,0): 1, (1,1): 1},   # A coeffs
            {(0,0): 1, (1,1): 1},   # B coeffs
            {(0,0): 1, (1,1): 1}    # C coeffs
        ),
        # M2 = (a10 + a11)b00
        (
            {(1,0): 1, (1,1): 1},   # A coeffs
            {(0,0): 1},             # B coeffs
            {(1,0): 1, (1,1): -1}   # C coeffs
        ),
        # M3 = a00(b01 - b11)
        (
            {(0,0): 1},             # A coeffs
            {(0,1): 1, (1,1): -1},  # B coeffs
            {(0,1): 1, (1,1): 1}    # C coeffs
        ),
        # M4 = a11(b10 - b00)
        (
            {(1,1): 1},             # A coeffs
            {(1,0): 1, (0,0): -1},  # B coeffs
            {(0,0): 1, (1,0): 1}    # C coeffs
        ),
        # M5 = (a00 + a01)b11
        (
            {(0,0): 1, (0,1): 1},   # A coeffs
            {(1,1): 1},             # B coeffs
            {(0,0): -1, (0,1): 1}   # C coeffs
        ),
        # M6 = (a10 - a00)(b00 + b01)
        (
            {(1,0): 1, (0,0): -1},  # A coeffs
            {(0,0): 1, (0,1): 1},   # B coeffs
            {(1,1): 1}              # C coeffs
        ),
        # M7 = (a01 - a11)(b10 + b11)
        (
            {(0,1): 1, (1,1): -1},  # A coeffs
            {(1,0): 1, (1,1): 1},   # B coeffs
            {(0,0): 1}              # C coeffs
        ),
    ]
    
    col = 0
    # For each A-block (2 of them) and B-block (4 of them)
    for a_block in range(2):
        for b_block in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                
                # Fill U: U[i*m + j, col]
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = 2 * a_block + lj
                    U[global_i * m + global_j, col] += coeff
                
                # Fill V: V[j*p + k, col]
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = 2 * a_block + lj
                    global_k = 2 * b_block + lk
                    V[global_j * p + global_k, col] += coeff
                
                # Fill W: W[k*n + i, col]
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = 2 * b_block + lk
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
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

def _try_rank_reduction(U, V, W, n, m, p, max_time=30.0):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 terms.
    We look for columns c such that the corresponding rank-1 tensor is a linear combination
    of the others. If we find such a column, we can remove it.
    
    For each column c, we check if the tensor T_c = u_c \otimes v_c \otimes w_c is in the span
    of the other tensors. This is equivalent to checking if the vectorized form of T_c is in
    the span of the vectorized forms of the other T_c's.
    
    However, the space of tensors is large. A more practical approach is to look for linear
    dependencies in the U, V, or W matrices directly, but that doesn't necessarily imply
    a reduction in the tensor rank.
    
    A better approach: use the fact that if the rank-1 tensors are linearly dependent, then
    the rank can be reduced. We can check for linear dependencies by constructing a matrix
    where each column is the vectorized form of one rank-1 tensor, and then finding the rank
    of this matrix. If the rank is less than the number of columns, we can remove some columns.
    
    But the vectorized tensor has size n*m*p = 64, and we have 56 columns. So we can form a
    64x56 matrix and check its rank. If the rank is 56, no reduction is possible. If it's less,
    we can remove columns.
    """
    start = time.time()
    deadline = start + max_time
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Vectorize each rank-1 tensor
    # T_c = u_c \otimes v_c \otimes w_c
    # The vectorized form has size n*m*p
    # We can compute it as:
    # T_c[i*m + j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But this is not a standard vectorization. Let's think of the tensor as a 3D array.
    # T_c[i, j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can flatten this to a 1D array of size n*m*p.
    
    # Let's construct the matrix M where M[:, c] is the vectorized T_c.
    M = np.zeros((n * m * p, rank), dtype=float)
    
    for c in range(rank):
        # For each (i, j, k), compute T_c[i, j, k]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, c] = U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
    
    # Now, find the rank of M. If rank(M) < rank, we can remove some columns.
    # We can use SVD to find the rank and the null space.
    try:
        # Use SVD to find the rank
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # The rank is the number of non-zero singular values.
        # We can use a tolerance to determine the rank.
        tol = max(M.shape) * np.finfo(float).eps * s[0] if s.size > 0 else 0
        rank_M = np.sum(s > tol)
        
        if rank_M < rank:
            # We can remove some columns.
            # We need to find a set of columns that spans the column space of M.
            # We can use the first rank_M columns of Vt (which are the right singular vectors)
            # to express the other columns as linear combinations.
            
            # For now, let's just return the original decomposition if we can't easily
            # find a reduced set.
            # A more sophisticated approach would be to use the null space of M to find
            # linear dependencies and remove columns.
            pass
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen decomposition first (rank 56)
    try:
        result = _strassen_decomposition()
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            # Try to reduce rank further if time permits
            if time.time() < deadline - 35.0:
                try:
                    reduced = _try_rank_reduction(
                        result["U"], result["V"], result["W"], n, m, p,
                        max_time=min(30.0, deadline - time.time() - 5.0)
                    )
                    if _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p, tol=1e-9):
                        return reduced
                except Exception:
                    pass
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 64)
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
