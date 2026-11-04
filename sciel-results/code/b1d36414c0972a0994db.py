import time
import numpy as np

def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' factor matrices.
    
    U: shape (16, rank) - n*m = 4*4 = 16
    V: shape (24, rank) - m*p = 4*6 = 24
    W: shape (24, rank) - p*n = 6*4 = 24
    
    The decomposition satisfies:
    For all i in [0,4), k in [0,6):
        C[i,k] = sum_j A[i,j] * B[j,k]
    where the tensor T[i,j,k] = delta_{i,k} is decomposed as:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = 4, 4, 6
    
    # We'll use a block-based Strassen approach.
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # Split B (4x6) into 2x3 blocks: B11, B12, B21, B22 (each 2x3)
    # C (4x6) into 2x3 blocks: C11, C12, C21, C22 (each 2x3)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For the 2x2 * 2x3 products, we can use a Strassen-like approach.
    # Actually, let's think about this more carefully.
    
    # A 2x2 * 2x3 product:
    # [a b] * [e f g] = [ae+bf, af+bg, ag+bh]
    # [c d]   [h i j]   [ce+df, cf+dg, cg+dh]
    
    # This has 6 outputs, each requiring 2 multiplications = 12 schoolbook.
    # Can we do better?
    
    # One approach: view 2x2*2x3 as two 2x2*2 products plus one 2x2*1 product.
    # But that doesn't obviously help.
    
    # Alternative: Use the fact that we can apply Strassen to the 2x2*2x2 part
    # and handle the extra column separately.
    
    # Let me try a different approach: use the known rank-7 Strassen for 2x2*2x2,
    # and handle the rectangular case.
    
    # Actually, let me just implement a clean Strassen-based decomposition for 4x4*4x4
    # and 4x4*4x2, then combine.
    
    # 4x4 * 4x6 = 4x4 * [4x4 | 4x2]
    # C = [C_left | C_right] where C_left = A*B_left (4x4) and C_right = A*B_right (4x2)
    
    # For 4x4*4x4: Use recursive Strassen. Rank = 7^2 = 49.
    # For 4x4*4x2: 
    #   Split A into 2x2 blocks, B_right (4x2) into 2x1 blocks.
    #   C_right = [C1; C2] where C1, C2 are 2x2.
    #   C1 = A11*B1 + A12*B3 (B1, B3 are 2x1)
    #   C2 = A21*B1 + A22*B3
    #   Each 2x2*2x1 product has rank 4 (schoolbook: 2*2*1=4).
    #   Total: 4 * 4 = 16.
    
    # Total rank: 49 + 16 = 65.
    
    # Wait, let me reconsider. For 4x4*4x2:
    # B_right is 4x2. Split into 2x2 blocks vertically: B_top (2x2) and B_bottom (2x2).
    # C_right = A * B_right = [A11 A12; A21 A22] * [B_top; B_bottom]
    #         = A11*B_top + A12*B_bottom (top 2x2)
    #         + A21*B_top + A22*B_bottom (bottom 2x2)
    # So 4 products of 2x2*2x2, each rank 7. Total: 4*7 = 28.
    
    # Total: 49 + 28 = 77.
    
    # Let me implement this carefully.
    
    rank = 77
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper functions for Strassen 2x2*2x2
    def strassen_2x2(a_rows, b_rows, c_rows, col):
        """
        a_rows: [a00, a01, a10, a11] - U row indices
        b_rows: [b00, b01, b10, b11] - V row indices  
        c_rows: [c00, c01, c10, c11] - W row indices
        Returns new col.
        """
        a00, a01, a10, a11 = a_rows
        b00, b01, b10, b11 = b_rows
        c00, c01, c10, c11 = c_rows
        
        # M1 = (a00+a11)(b00+b11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1; W[c01, col] += -1; W[c10, col] += -1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] += -1
        col += 1
        
        # M3 = a00(b10-b11)
        U[a00, col] += 1
        V[b10, col] += 1; V[b11, col] += -1
        W[c01, col] += 1; W[c11, col] += -1
        col += 1
        
        # M4 = a11(b01-b00)
        U[a11, col] += 1
        V[b01, col] += 1; V[b00, col] += -1
        W[c00, col] += 1; W[c11, col] += -1
        col += 1
        
        # M5 = (a00+a01)b11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c01, col] += -1; W[c11, col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10, col] += 1; U[a00, col] += -1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01, col] += 1; U[a11, col] += -1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
        return col
    
    # Index helpers
    def a_idx(i, j):
        """U row index for A[i,j]"""
        return i * m + j
    
    def v_idx(j, k):
        """V row index for B[j,k]"""
        return j * p + k
    
    def w_idx(k, i):
        """W row index for C[i,k] = W[k*n+i]"""
        return k * n + i
    
    col = 0
    
    # Part 1: 4x4 * 4x4 (B columns 0-3, C columns 0-3)
    # Strassen on 2x2 blocks
    # A blocks: A11=(0,0),(0,1),(1,0),(1,1); A12=(0,2),(0,3),(1,2),(1,3); etc.
    
    A11 = [a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1)]
    A12 = [a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3)]
    A21 = [a_idx(2,0), a_idx(2,1), a_idx(3,0), a_idx(3,1)]
    A22 = [a_idx(2,2), a_idx(2,3), a_idx(3,2), a_idx(3,3)]
    
    B11 = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    B12 = [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)]
    B21 = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)]
    B22 = [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)]
    
    C11 = [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)]
    C12 = [w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)]
    C21 = [w_idx(0,2), w_idx(1,2), w_idx(0,3), w_idx(1,3)]
    C22 = [w_idx(2,2), w_idx(3,2), w_idx(2,3), w_idx(3,3)]
    
    # Strassen for 4x4*4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Mi, we need to apply Strassen to a 2x2*2x2 where the "blocks"
    # may be sums/differences of the base 2x2 blocks.
    
    # This requires careful handling. Let me define a function that takes
    # linear combinations of blocks.
    
    def add_strassen_for_sum(a_coeffs, b_coeffs, c_coeffs, col):
        """
        a_coeffs: list of (block_list, coeff) where block_list is [a00,a01,a10,a11]
        b_coeffs: same for B
        c_coeffs: same for C
        """
        # This is complex. Let me just expand manually.
        pass
    
    # Given the complexity, let me use a simpler but correct approach:
    # Just use schoolbook for the full 4x4*4x6, which gives rank 96.
    # But that's the same as the seed.
    
    # Let me try yet another approach: use the fact that we can do 4x4*4x4 with rank 49
    # by directly enumerating the 49 terms.
    
    # Reset and use a different strategy: build the 4x4*4x4 Strassen decomposition
    # by explicitly computing what the 49 rank-1 terms are.
    
    # Actually, let me just go with a clean schoolbook implementation first,
    # then try to optimize.
    
    U = np.zeros((n*m, n*m*p), dtype=float)
    V = np.zeros((m*p, n*m*p), dtype=float)
    W = np.zeros((p*n, n*m*p), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = a_idx(i, j)
                v = v_idx(j, k)
                w = w_idx(k, i)
                U[u, r] = 1
                V[v, r] = 1
                W[w, r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
