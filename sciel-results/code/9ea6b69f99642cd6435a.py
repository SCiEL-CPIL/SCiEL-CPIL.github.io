import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2_factors():
    """
    Returns the 7 Strassen factors for 2x2x2 matrix multiplication.
    Each factor is (u_coeffs, v_coeffs, w_coeffs) where each is a list of 4 entries
    corresponding to positions [0,1,2,3] = [A11,A12,A21,A22] etc.
    
    Strassen's algorithm:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    factors = [
        # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (A21+A22)B11 -> C21 += 1, C22 -= 1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        # M3 = A11(B21-B22) -> C12 += 1, C22 += 1
        ([1,0,0,0], [0,1,-1,0], [0,1,0,1]),
        # M4 = A22(B12-B11) -> C11 += 1, C21 += 1
        ([0,0,0,1], [1,-1,0,0], [1,0,1,0]),
        # M5 = (A11+A12)B22 -> C11 -= 1, C12 += 1
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # M6 = (A21-A11)(B11+B12) -> C22 += 1
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M7 = (A12-A22)(B21+B22) -> C11 += 1
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    return factors

def _construct_optimized():
    """
    Constructs a decomposition using block partitioning:
    n=3 split into 2+1
    m=5 split into 2+2+1
    p=6 split into 2+2+2
    
    This yields a total rank of 84.
    """
    n, m, p = DIMENSIONS
    
    # Define partitions
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Calculate offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
        
    # Pre-calculate total rank
    total_rank = 0
    for i_idx, n_size in enumerate(n_splits):
        for j_idx, m_size in enumerate(m_splits):
            for k_idx, p_size in enumerate(p_splits):
                if n_size == 2 and m_size == 2 and p_size == 2:
                    total_rank += 7
                else:
                    total_rank += n_size * m_size * p_size
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    strassen_factors = _strassen_2x2x2_factors()
    
    for i_idx, n_size in enumerate(n_splits):
        i_start = n_offsets[i_idx]
        for j_idx, m_size in enumerate(m_splits):
            j_start = m_offsets[j_idx]
            for k_idx, p_size in enumerate(p_splits):
                k_start = p_offsets[k_idx]
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    # Apply Strassen
                    for u_coeffs, v_coeffs, w_coeffs in strassen_factors:
                        # Map local indices to global
                        # Local A index: i_local (0-1), j_local (0-1) -> flat i*m+j
                        # Local B index: j_local (0-1), k_local (0-1) -> flat j*p+k
                        # Local C index: k_local (0-1), i_local (0-1) -> flat k*n+i
                        
                        for li in range(2):
                            for lj in range(2):
                                local_a_idx = li * 2 + lj
                                val = u_coeffs[local_a_idx]
                                if val != 0:
                                    global_i = i_start + li
                                    global_j = j_start + lj
                                    U[global_i * m + global_j][rank_offset] += val
                                    
                        for lj in range(2):
                            for lk in range(2):
                                local_b_idx = lj * 2 + lk
                                val = v_coeffs[local_b_idx]
                                if val != 0:
                                    global_j = j_start + lj
                                    global_k = k_start + lk
                                    V[global_j * p + global_k][rank_offset] += val
                                    
                        for lk in range(2):
                            for li in range(2):
                                local_c_idx = lk * 2 + li
                                val = w_coeffs[local_c_idx]
                                if val != 0:
                                    global_k = k_start + lk
                                    global_i = i_start + li
                                    W[global_k * n + global_i][rank_offset] += val
                                    
                        rank_offset += 1
                else:
                    # Schoolbook
                    for li in range(n_size):
                        for lj in range(m_size):
                            for lk in range(p_size):
                                global_i = i_start + li
                                global_j = j_start + lj
                                global_k = k_start + lk
                                U[global_i * m + global_j][rank_offset] = 1
                                V[global_j * p + global_k][rank_offset] = 1
                                W[global_k * n + global_i][rank_offset] = 1
                                rank_offset += 1
                                
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    """
    Constructs the standard schoolbook decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with index k*n+i
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term corresponds to A[i,j] * B[j,k] contributing to C[i,k]
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Entry point to return the factor matrices.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    
    # Use the optimized block decomposition (rank 84)
    result = _construct_optimized()
    
    # Safety check
    if time.time() - start_time > 100:
         return _construct_schoolbook()
         
    return result

# EVOLVE-BLOCK-END
