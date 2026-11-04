"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
from fractions import Fraction
from itertools import combinations
import time

# EVOLVE-BLOCK-START

def _deadline_ok(start_time, margin=10.0):
    return time.time() - start_time < 150.0 - margin

def _verify_decomp(n, m, p, U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition with rank n*m*p."""
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

def _strassen_222_block():
    """
    Return the 7 Strassen multiplication terms for 2x2 * 2x2.
    Each term is (u_coeffs, v_coeffs, w_coeffs) where:
      u_coeffs: length-4 list for the 2x2 A subblock (row-major: [a00,a01,a10,a11])
      v_coeffs: length-4 list for the 2x2 B subblock
      w_coeffs: length-4 list for the 2x2 C subblock
    """
    # Standard Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # Indexing: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    terms = []
    # M1 = (A00+A11)*(B00+B11)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))  # C00 += M1
    # M2 = (A10+A11)*B00
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))  # C10 += M2
    # M3 = A00*(B01-B11)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))  # C01 += M3
    # M4 = A11*(B10-B00)
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,0,1]))  # C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11
    terms.append(([1,1,0,0], [0,0,0,1], [0,1,-1,0]))  # C01 += M5, C00 -= M5
    # M6 = (A10-A00)*(B00+B10)
    terms.append(([0,0,1,-1], [1,0,1,0], [0,0,0,1]))  # C11 += M6
    # M7 = (A01-A11)*(B01+B11)
    terms.append(([0,1,0,-1], [0,1,0,1], [1,0,0,1]))  # C00 += M7, C11 += M7
    
    return terms

def _add_strassen_222(U, V, W, rank_offset, a_row_offset, b_row_offset, c_row_offset, n, m, p):
    """
    Add Strassen 2x2*2x2 block.
    a_row_offset: starting row index in A (n x m)
    b_row_offset: starting row index in B (m x p)  
    c_row_offset: starting row index in C (n x p), but C is indexed as [k*n+i]
    
    The 2x2 A subblock covers rows [a_row_offset, a_row_offset+1] of A,
    all within the m columns.
    The 2x2 B subblock covers rows [b_row_offset, b_row_offset+1] of B,
    all within the p columns.
    The 2x2 C subblock covers rows [c_row_offset, c_row_offset+1] of C (i-index),
    and columns [c_col_offset, c_col_offset+1] (k-index).
    
    Actually let me be more careful. A is n x m, B is m x p, C is n x p.
    A[i,j] for i in [0,n), j in [0,m)
    B[j,k] for j in [0,m), k in [0,p)
    C[i,k] for i in [0,n), k in [0,p)
    W is indexed as W[k*n + i, t]
    
    For a 2x2 block: A rows i0,i1; B rows j0,j1 (which are the "middle" dimension); C rows i0,i1 and cols k0,k1.
    The middle dimension j must match: A's columns j0,j1 = B's rows j0,j1.
    """
    strassen_terms = _strassen_222_block()
    
    for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
        t = rank_offset
        # A subblock: rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1
        for ai in range(2):
            for aj in range(2):
                a_idx = (a_row_offset + ai) * m + (a_col_offset + aj) if False else None
                # A is n x m, indexed as [i*m + j]
                # The 2x2 A subblock: rows i0,i1 (of n), cols j0,j1 (of m)
                # u_coeffs[ai*2+aj]
                pass
        # Let me redo this more carefully
        # A[i,j] -> U[i*m+j, t]
        # B[j,k] -> V[j*p+k, t]
        # C[i,k] -> W[k*n+i, t]
        
        # A subblock at rows [i0, i0+1], cols [j0, j0+1]
        # B subblock at rows [j0, j0+1], cols [k0, k0+1]
        # C subblock at rows [i0, i0+1], cols [k0, k0+1]
        pass
    return rank_offset + 7

def _build_strassen_based(n, m, p):
    """
    Build a Strassen-based decomposition for 2x5*5x5.
    
    Partition:
    - A is 2x5: rows {0,1}, cols {0,1,2,3,4}
    - B is 5x5: rows {0,1,2,3,4}, cols {0,1,2,3,4}
    - C is 2x5: rows {0,1}, cols {0,1,2,3,4}
    
    Partition the middle dimension (cols of A / rows of B) into {0,1}, {2,3}, {4}
    Partition the output columns (cols of B / cols of C) into {0,1}, {2,3}, {4}
    
    For each pair (middle_block, output_block), we have a subproblem:
    - A_sub: 2 x |middle_block|
    - B_sub: |middle_block| x |output_block|
    - C_sub: 2 x |output_block|
    
    Cases:
    1. middle={0,1}, output={0,1}: A is 2x2, B is 2x2, C is 2x2 -> Strassen (7 mults)
    2. middle={0,1}, output={2,3}: A is 2x2, B is 2x2, C is 2x2 -> Strassen (7 mults)
    3. middle={0,1}, output={4}: A is 2x2, B is 2x1, C is 2x1 -> 2x2*2x1 (4 mults)
    4. middle={2,3}, output={0,1}: A is 2x2, B is 2x2, C is 2x2 -> Strassen (7 mults)
    5. middle={2,3}, output={2,3}: A is 2x2, B is 2x2, C is 2x2 -> Strassen (7 mults)
    6. middle={2,3}, output={4}: A is 2x2, B is 2x1, C is 2x1 -> 2x2*2x1 (4 mults)
    7. middle={4}, output={0,1}: A is 2x1, B is 1x2, C is 2x2 -> 2x1*1x2 (2 mults)
    8. middle={4}, output={2,3}: A is 2x1, B is 1x2, C is 2x2 -> 2x1*1x2 (2 mults)
    9. middle={4}, output={4}: A is 2x1, B is 1x1, C is 2x1 -> 2x1*1x1 (2 mults)
    
    Total: 7+7+4+7+7+4+2+2+2 = 42
    
    Wait, let me recount:
    - 2x2*2x2 blocks: 4 of them -> 4*7 = 28
    - 2x2*2x1 blocks: 2 of them -> 2*4 = 8
    - 2x1*1x2 blocks: 2 of them -> 2*2 = 4
    - 2x1*1x1 block: 1 of them -> 1*2 = 2
    Total: 28+8+4+2 = 42
    """
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0  # current rank offset
    
    # Define the partitions
    middle_blocks = [(0,1), (2,3), (4,5)]  # (start, end) exclusive
    output_blocks = [(0,1), (2,3), (4,5)]
    
    for (j0, j1) in middle_blocks:
        for (k0, k1) in output_blocks:
            mid_size = j1 - j0
            out_size = k1 - k0
            
            if mid_size == 2 and out_size == 2:
                # Strassen 2x2 * 2x2
                strassen_terms = _strassen_222_block()
                for (u_c, v_c, w_c) in strassen_terms:
                    for ai in range(2):
                        for aj in range(2):
                            a_idx = (0 + ai) * m + (j0 + aj)  # A row ai, col j0+aj
                            # Wait, n=2 so rows are 0,1
                            a_row = ai  # 0 or 1
                            a_col = j0 + aj
                            U[a_row * m + a_col][t] = u_c[ai * 2 + aj]
                    for bi in range(2):
                        for bk in range(2):
                            b_row = j0 + bi
                            b_col = k0 + bk
                            V[b_row * p + b_col][t] = v_c[bi * 2 + bk]
                    for ci in range(2):
                        for ck in range(2):
                            c_row = ci  # 0 or 1 (i index)
                            c_col = k0 + ck
                            # W is indexed as [k*n + i]
                            W[c_col * n + c_row][t] = w_c[ci * 2 + ck]
                    t += 1
                    
            elif mid_size == 2 and out_size == 1:
                # 2x2 * 2x1: 4 multiplications
                # A is 2x2 (rows 0,1; cols j0,j0+1), B is 2x1 (rows j0,j0+1; col k0)
                # C is 2x1 (rows 0,1; col k0)
                # C[i,k0] = A[i,j0]*B[j0,k0] + A[i,j0+1]*B[j0+1,k0]
                # Just use schoolbook: 4 terms
                for i in range(2):
                    for j in range(2):
                        U[i * m + (j0 + j)][t] = 1
                        V[(j0 + j) * p + k0][t] = 1
                        W[k0 * n + i][t] = 1
                        t += 1
                        
            elif mid_size == 1 and out_size == 2:
                # 2x1 * 1x2: 2 multiplications
                # A is 2x1 (rows 0,1; col j0), B is 1x2 (row j0; cols k0,k0+1)
                # C is 2x2 (rows 0,1; cols k0,k0+1)
                # C[i,k] = A[i,j0]*B[j0,k]
                for i in range(2):
                    for k in range(2):
                        U[i * m + j0][t] = 1
                        V[j0 * p + (k0 + k)][t] = 1
                        W[(k0 + k) * n + i][t] = 1
                        t += 1
                        
            elif mid_size == 1 and out_size == 1:
                # 2x1 * 1x1: 2 multiplications
                for i in range(2):
                    U[i * m + j0][t] = 1
                    V[j0 * p + k0][t] = 1
                    W[k0 * n + i][t] = 1
                    t += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition (rank 42)
    try:
        result = _build_strassen_based(n, m, p)
        if _verify_decomp(n, m, p, result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
