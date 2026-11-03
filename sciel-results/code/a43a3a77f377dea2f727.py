import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Try structured decomposition using Strassen for 2x2x2 blocks
    best_result = _structured_decomposition(n, m, p)
    best_rank = _get_rank(best_result)
    
    # If structured approach gives improvement, try to verify and potentially improve
    if best_rank < schoolbook_rank:
        # Verify the decomposition
        if _verify_decomposition(best_result['U'], best_result['V'], best_result['W'], n, m, p, best_rank):
            return best_result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

def _get_rank(result):
    """Get the rank from a decomposition result."""
    if not result or 'U' not in result:
        return 999
    return len(result['U'][0]) if result['U'] else 999

def _schoolbook_decomposition(n, m, p):
    """Create the standard schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
    And cross terms should be 0.
    """
    # Check main terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1.0) > 1e-6:
                    return False
                
                # Check cross terms for this i, k
                for j_prime in range(m):
                    if j_prime == j:
                        continue
                    total_cross = 0.0
                    for c in range(rank):
                        total_cross += U[i * m + j_prime][c] * V[j_prime * p + k][c] * W[k * n + i][c]
                    if abs(total_cross) > 1e-6:
                        return False
    return True

def _structured_decomposition(n, m, p):
    """
    Use a structured decomposition that applies Strassen's algorithm to 2x2x2 sub-blocks.
    
    For 3x5x6:
    - Partition into blocks where possible
    - Apply Strassen (rank 7) to 2x2x2 blocks
    - Use schoolbook for remaining 1x1 or 1x2 blocks
    
    Strategy:
    - n=3: split into 2+1
    - m=5: split into 2+2+1  
    - p=6: split into 2+2+2
    
    This gives us a block structure where we can apply Strassen to the 2x2x2 blocks.
    """
    # Define block sizes
    n_blocks = [2, 1]  # 3 = 2 + 1
    m_blocks = [2, 2, 1]  # 5 = 2 + 2 + 1
    p_blocks = [2, 2, 2]  # 6 = 2 + 2 + 2
    
    # Calculate cumulative offsets
    n_offsets = [0]
    for size in n_blocks:
        n_offsets.append(n_offsets[-1] + size)
    m_offsets = [0]
    for size in m_blocks:
        m_offsets.append(m_offsets[-1] + size)
    p_offsets = [0]
    for size in p_blocks:
        p_offsets.append(p_offsets[-1] + size)
    
    # Build the decomposition by combining block decompositions
    # Each block (ni, mj, pk) contributes a sub-decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # We'll build U, V, W as lists of lists, one row per (i,j), (j,k), (k,i)
    # Start with empty
    U_rows = {i * m + j: [] for i in range(n) for j in range(m)}
    V_rows = {j * p + k: [] for j in range(m) for k in range(p)}
    W_rows = {k * n + i: [] for k in range(p) for i in range(n)}
    
    # For each block combination, add the appropriate decomposition
    for ni_idx, ni_size in enumerate(n_blocks):
        for mj_idx, mj_size in enumerate(m_blocks):
            for pk_idx, pk_size in enumerate(p_blocks):
                # Determine if this block can use Strassen (all sizes must be 2)
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    # Use Strassen decomposition for this 2x2x2 block
                    _add_strassen_block(U_rows, V_rows, W_rows, 
                                       n_offsets[ni_idx], m_offsets[mj_idx], p_offsets[pk_idx],
                                       n, m, p)
                else:
                    # Use schoolbook for this block
                    _add_schoolbook_block(U_rows, V_rows, W_rows,
                                        n_offsets[ni_idx], m_offsets[mj_idx], p_offsets[pk_idx],
                                        ni_size, mj_size, pk_size,
                                        n, m, p)
    
    # Determine the rank (maximum length of any row)
    rank = 0
    for row in U_rows.values():
        rank = max(rank, len(row))
    for row in V_rows.values():
        rank = max(rank, len(row))
    for row in W_rows.values():
        rank = max(rank, len(row))
    
    # Pad all rows to have the same length (rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for c, val in enumerate(U_rows[i * m + j]):
                U[i * m + j][c] = val
    
    for j in range(m):
        for k in range(p):
            for c, val in enumerate(V_rows[j * p + k]):
                V[j * p + k][c] = val
    
    for k in range(p):
        for i in range(n):
            for c, val in enumerate(W_rows[k * n + i]):
                W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}

def _add_strassen_block(U_rows, V_rows, W_rows, i_offset, j_offset, k_offset, n, m, p):
    """
    Add a Strassen decomposition for a 2x2x2 block starting at (i_offset, j_offset, k_offset).
    
    Strassen's algorithm computes C = A * B where A is 2x2, B is 2x2, C is 2x2.
    It uses 7 multiplications instead of 8.
    
    The 7 products are:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    And C is:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # For tensor decomposition, we need to represent each product as a rank-1 term
    # Each Mx corresponds to one rank-1 term in the tensor
    
    # The Strassen decomposition has 7 rank-1 terms
    # For each term, we need to define U, V, W contributions
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    # This contributes to C00, C11
    # U: affects A00, A11 rows
    # V: affects B00, B11 cols  
    # W: affects C00, C11
    
    # We'll add 7 terms, each with coefficients in U, V, W
    
    # For simplicity, let's define the Strassen coefficients
    # Each term c has:
    #   U[i,j] coefficient for each (i,j) in the 2x2 A block
    #   V[j,k] coefficient for each (j,k) in the 2x2 B block
    #   W[k,i] coefficient for each (k,i) in the 2x2 C block
    
    # Strassen coefficients (7 terms):
    # Term 0: M1
    #   U: A00=1, A11=1
    #   V: B00=1, B11=1
    #   W: C00=1, C11=1
    # Term 1: M2
    #   U: A10=1, A11=1
    #   V: B00=1
    #   W: C10=1
    # Term 2: M3
    #   U: A00=1
    #   V: B01=1, B11=-1
    #   W: C01=1
    # Term 3: M4
    #   U: A11=1
    #   V: B10=1, B00=-1
    #   W: C00=1, C10=1
    # Term 4: M5
    #   U: A00=1, A01=1
    #   V: B11=1
    #   W: C00=-1, C01=1
    # Term 5: M6
    #   U: A10=1, A00=-1
    #   V: B00=1, B01=1
    #   W: C11=1
    # Term 6: M7
    #   U: A01=1, A11=-1
    #   V: B10=1, B11=1
    #   W: C00=1, C11=-1
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict mapping (local_i, local_j) -> coeff
        # V_coeffs: dict mapping (local_j, local_k) -> coeff
        # W_coeffs: dict mapping (local_k, local_i) -> coeff
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        },
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(1,0): 1}
        },
        {
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(0,1): 1}
        },
        {
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(0,0): 1, (1,0): 1}
        },
        {
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,0): -1, (0,1): 1}
        },
        {
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (0,1): 1},
            'W': {(1,1): 1}
        },
        {
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): -1}
        }
    ]
    
    for term in strassen_terms:
        # Add this term to U, V, W
        # We need to find the next available column index
        # For simplicity, we'll append to each row
        
        # Get current max length
        max_len = 0
        for row in U_rows.values():
            max_len = max(max_len, len(row))
        for row in V_rows.values():
            max_len = max(max_len, len(row))
        for row in W_rows.values():
            max_len = max(max_len, len(row))
        
        col_idx = max_len
        
        # Initialize all rows with 0 at this column
        for i in range(n):
            for j in range(m):
                U_rows[i * m + j].append(0)
        for j in range(m):
            for k in range(p):
                V_rows[j * p + k].append(0)
        for k in range(p):
            for i in range(n):
                W_rows[k * n + i].append(0)
        
        # Set the coefficients
        for (li, lj), coeff in term['U'].items():
            i_global = i_offset + li
            j_global = j_offset + lj
            U_rows[i_global * m + j_global][col_idx] = coeff
        
        for (lj, lk), coeff in term['V'].items():
            j_global = j_offset + lj
            k_global = k_offset + lk
            V_rows[j_global * p + k_global][col_idx] = coeff
        
        for (lk, li), coeff in term['W'].items():
            k_global = k_offset + lk
            i_global = i_offset + li
            W_rows[k_global * n + i_global][col_idx] = coeff

def _add_schoolbook_block(U_rows, V_rows, W_rows, i_offset, j_offset, k_offset, 
                          ni_size, mj_size, pk_size, n, m, p):
    """
    Add a schoolbook decomposition for a block of size ni_size x mj_size x pk_size.
    """
    for li in range(ni_size):
        for lj in range(mj_size):
            for lk in range(pk_size):
                # Add one rank-1 term
                # Get current max length
                max_len = 0
                for row in U_rows.values():
                    max_len = max(max_len, len(row))
                for row in V_rows.values():
                    max_len = max(max_len, len(row))
                for row in W_rows.values():
                    max_len = max(max_len, len(row))
                
                col_idx = max_len
                
                # Initialize all rows with 0 at this column
                for i in range(n):
                    for j in range(m):
                        U_rows[i * m + j].append(0)
                for j in range(m):
                    for k in range(p):
                        V_rows[j * p + k].append(0)
                for k in range(p):
                    for i in range(n):
                        W_rows[k * n + i].append(0)
                
                # Set the coefficients
                i_global = i_offset + li
                j_global = j_offset + lj
                k_global = k_offset + lk
                
                U_rows[i_global * m + j_global][col_idx] = 1
                V_rows[j_global * p + k_global][col_idx] = 1
                W_rows[k_global * n + i_global][col_idx] = 1

# EVOLVE-BLOCK-END
