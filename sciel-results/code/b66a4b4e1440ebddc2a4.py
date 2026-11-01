"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    
    # Attempt to find a rank-27 decomposition by exploiting linear dependencies
    # in the tensor structure. If it fails or takes too long, fall back to rank 28.
    
    # The multiplication tensor T has entries T[i, j, k] = 1 if i*m + j = k*n + i? 
    # No, T is defined by (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor indices: (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # Flattened: U index i*m+j, V index j*p+k, W index k*n+i.
    
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    
    # This is equivalent to finding a rank-r decomposition of the matrix M
    # obtained by matricizing the tensor.
    # Let's try to solve for rank 27 using linear algebra.
    
    # We can fix V to be a basis-like matrix or use a random V and solve for U and W?
    # No, this is a bilinear problem.
    
    # A common approach is to set up the equations and use a solver.
    # However, for rank minimization, we can try to see if the tensor lies in a
    # specific algebraic variety.
    
    # Given the time constraint and the difficulty of general rank minimization,
    # we will stick to the known rank-28 Strassen-based decomposition which is robust.
    # The "inspiration" to find rank 27 is theoretically sound but computationally
    # risky without a guaranteed solver.
    
    # Let's verify the rank 28 solution is correct and return it.
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        """
        Add a Strassen block for multiplying A[a_row_base:a_row_base+2, a_col_base:a_col_base+2]
        by B[b_row_base:b_row_base+2, b_col_base:b_col_base+2]
        contributing to C[c_row_base:c_row_base+2, c_col_base:c_col_base+2]
        
        U index for A element (i,j): i*m + j = i*4 + j
        V index for B element (i,j): i*p + j = i*4 + j
        W index for C element (i,j): j*n + i = j*2 + i  (trace convention: W[k*n+i, r] where k=col, i=row)
        """
        # A is 2x2 at rows a_row_base, a_row_base+1 and cols a_col_base, a_col_base+1
        # B is 2x2 at rows b_row_base, b_row_base+1 and cols b_col_base, b_col_base+1
        # C is 2x2 at rows c_row_base, c_row_base+1 and cols c_col_base, c_col_base+1
        
        # U indices for A elements
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        # V indices for B elements
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        # W indices for C elements (trace convention: W[k*n+i, r] where k=col, i=row)
        # C element at row i, col j has W index = j*n + i
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1
    
    # C_left = A_left * B_top_left + A_right * B_bottom_left
    # A_left: rows 0-1, cols 0-1 of A (2x4)
    # A_right: rows 0-1, cols 2-3 of A (2x4)
    # B_top_left: rows 0-1, cols 0-1 of B (4x4)
    # B_bottom_left: rows 2-3, cols 0-1 of B (4x4)
    # B_top_right: rows 0-1, cols 2-3 of B (4x4)
    # B_bottom_right: rows 2-3, cols 2-3 of B (4x4)
    # C_left: rows 0-1, cols 0-1 of C (2x4)
    # C_right: rows 0-1, cols 2-3 of C (2x4)
    
    # Block 1: A_left * B_top_left -> C_left
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_right * B_bottom_left -> C_left
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A_left * B_top_right -> C_right
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A_right * B_bottom_right -> C_right
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
