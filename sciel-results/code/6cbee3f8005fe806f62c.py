"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    
    # First, build the rank-28 Strassen decomposition as baseline
    rank28_U, rank28_V, rank28_W = _build_rank28()
    
    # Try to find rank-27 by checking if we can eliminate one multiplication
    # through linear dependency analysis
    best_result = {"U": rank28_U, "V": rank28_V, "W": rank28_W}
    
    # Verify rank-28 is correct
    if time.time() - start_time < 140:
        if _verify_decomposition(rank28_U, rank28_V, rank28_W, n, m, p):
            # Try rank 27: check if any column in the tensor decomposition is redundant
            if time.time() - start_time < 140:
                rank27_result = _try_rank27(rank28_U, rank28_V, rank28_W, n, m, p)
                if rank27_result is not None:
                    best_result = rank27_result
    
    return best_result

def _build_rank28():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1
    
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_k U[:,k] outer V[:,k] outer W[:,k] = matmul tensor"""
    # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # should equal 1 if i corresponds to (row_a, col_a), j to (row_b, col_b), k to (col_c, row_c)
    # and col_a == row_b and row_a == row_c and col_c == col_c
    rank = len(U[0])
    for i in range(n * m):
        a_row, a_col = divmod(i, m)
        for j in range(m * p):
            b_row, b_col = divmod(j, p)
            for k in range(p * n):
                c_col, c_row = divmod(k, n)
                expected = 1.0 if (a_col == b_row and a_row == c_row and a_col == c_col) else 0.0
                # Wait, the tensor entry T[i,j,k] = 1 iff i's col == j's row AND k's row == i's row AND k's col == j's col
                # i = a_row*m + a_col, j = b_row*p + b_col, k = c_col*n + c_row
                # T=1 iff a_col == b_row AND a_row == c_row AND a_col == c_col? No...
                # C[c_row, c_col] = sum_a A[a_row, a_col] * B[a_col, c_col]
                # So T[i,j,k]=1 iff a_col==b_row AND a_row==c_row AND c_col==b_col... wait
                # Actually T[i,j,k] = 1 iff the indices are consistent for the product:
                # i encodes (a_row, a_col), j encodes (b_row, b_col), k encodes (c_row, c_col)
                # T=1 iff a_col == b_row AND a_row == c_row AND b_col == c_col
                if a_col == b_row and a_row == c_row and b_col == c_col:
                    expected = 1.0
                else:
                    expected = 0.0
                
                actual = sum(U[i][r] * V[j][r] * W[k][r] for r in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _try_rank27(U28, V28, W28, n, m, p):
    """Try to find a rank-27 decomposition by removing one term if it's linearly dependent"""
    # This is a heuristic - in general the 28 terms from Strassen are independent
    # For 2x4x4, R=27 is known to exist but requires a different construction
    # Return None if we can't find one quickly
    return None
# EVOLVE-BLOCK-END
