"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def get_strassen_2x2():
    """
    Returns the 7 bilinear forms for 2x2 matrix multiplication using Strassen's algorithm.
    """
    # Indices for 2x2 matrix: 00, 01, 10, 11 -> 0, 1, 2, 3
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # P3
        ([0, 0, 0, 1], [0, 0, 1, -1]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1]), # P5
        ([0, 0, 1, -1], [1, 1, 0, 0]), # P6
        ([0, 1, 0, -1], [0, 0, 1, 1]), # P7
    ]
    
    recon_2x2 = [
        [1, 0, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, -1, 1, 0, 0, 0]
    ]
    
    return strassen_2x2, recon_2x2

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # We will compute 4x4 by 4x8 as two 4x4 by 4x4 multiplications.
    # Rank = 2 * 49 = 98.
    
    strassen_2x2, recon_2x2 = get_strassen_2x2()
    
    # Generate the 49 scalar products for one 4x4 by 4x4 multiplication.
    # A is 4x4, B is 4x4.
    
    # Block products for 4x4 x 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M_i is a 2x2 matrix product.
    # We expand each M_i using strassen_2x2.
    
    # Define L_i and R_i as 2x2 matrices of block references.
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11.
    
    # For each block product, we define the linear forms on the blocks.
    # L_i[k,l] is a list of (coeff, block_idx)
    
    L_defs = [
        # M1: A00 + A11
        [[(1, 0), (1, 3)], [(1, 0), (1, 3)], [(1, 0), (1, 3)], [(1, 0), (1, 3)]],
        # M2: A10 + A11
        [[(1, 2), (1, 3)], [(1, 2), (1, 3)], [(1, 2), (1, 3)], [(1, 2), (1, 3)]],
        # M3: A00
        [[(1, 0)], [(1, 0)], [(1, 0)], [(1, 0)]],
        # M4: A11
        [[(1, 3)], [(1, 3)], [(1, 3)], [(1, 3)]],
        # M5: A00 + A01
        [[(1, 0), (1, 1)], [(1, 0), (1, 1)], [(1, 0), (1, 1)], [(1, 0), (1, 1)]],
        # M6: A10 - A00
        [[(1, 2), (-1, 0)], [(1, 2), (-1, 0)], [(1, 2), (-1, 0)], [(1, 2), (-1, 0)]],
        # M7: A01 - A11
        [[(1, 1), (-1, 3)], [(1, 1), (-1, 3)], [(1, 1), (-1, 3)], [(1, 1), (-1, 3)]],
    ]
    
    R_defs = [
        # M1: B00 + B11
        [[(1, 0), (1, 3)], [(1, 0), (1, 3)], [(1, 0), (1, 3)], [(1, 0), (1, 3)]],
        # M2: B00
        [[(1, 0)], [(1, 0)], [(1, 0)], [(1, 0)]],
        # M3: B01 - B11
        [[(1, 1), (-1, 3)], [(1, 1), (-1, 3)], [(1, 1), (-1, 3)], [(1, 1), (-1, 3)]],
        # M4: B10 - B00
        [[(1, 2), (-1, 0)], [(1, 2), (-1, 0)], [(1, 2), (-1, 0)], [(1, 2), (-1, 0)]],
        # M5: B11
        [[(1, 3)], [(1, 3)], [(1, 3)], [(1, 3)]],
        # M6: B00 + B01
        [[(1, 0), (1, 1)], [(1, 0), (1, 1)], [(1, 0), (1, 1)], [(1, 0), (1, 1)]],
        # M7: B10 + B11
        [[(1, 2), (1, 3)], [(1, 2), (1, 3)], [(1, 2), (1, 3)], [(1, 2), (1, 3)]],
    ]
    
    # Now generate the 49 scalar products.
    # For each block product i (0..6) and each Strassen 2x2 product j (0..6):
    #   We have u_j (4-dim) and v_j (4-dim) from strassen_2x2[j].
    #   The left factor is L_i composed with u_j.
    #   The right factor is R_i composed with v_j.
    
    # L_i is a 2x2 matrix. Each entry L_i[k,l] is a linear form on the 4 entries of the blocks.
    # u_j is a 4-dim vector. u_j[k,l] is the coefficient for L_i[k,l].
    
    # The composed left factor is:
    # sum_{k,l} u_j[k,l] * L_i[k,l]
    # where L_i[k,l] is a list of (coeff, block_idx).
    
    # The result is a 16-dim vector.
    
    def compose_left(L_i, u_j):
        vec = [0.0] * 16
        for k in range(2):
            for l in range(2):
                coeff_u = u_j[k * 2 + l]
                if coeff_u != 0:
                    for coeff, block_idx in L_i[k][l]:
                        # L_i[k][l] contributes coeff to each entry of the block?
                        # No, L_i[k][l] is a 2x2 matrix.
                        # L_i[k][l] = sum_{b} c_{k,l,b} * A_block_b.
                        # Each A_block_b is a 2x2 matrix.
                        # So L_i[k][l] is a 2x2 matrix of linear forms.
                        # This is getting complicated.
                        pass
        return vec
    
    # This approach is flawed. Let's use a different representation.
    # Represent L_i as a 4x16 matrix where L_i[k,l, entry] is the coefficient of A[entry].
    # Actually, L_i is a 2x2 matrix. Each entry is a 2x2 matrix of coefficients.
    # So L_i is a 4x16 matrix? No, L_i is a 4x4 matrix of blocks, each block is 4 entries.
    # Total 16 entries.
    
    # Let's define L_i as a 16-dim vector for each of the 4 entries of the 2x2 matrix.
    # L_i[k,l] is a 16-dim vector.
    
    L_vecs = []
    R_vecs = []
    
    for i in range(7):
        L_vec = [[0.0] * 16 for _ in range(4)]
        R_vec = [[0.0] * 16 for _ in range(4)]
        
        for k in range(2):
            for l in range(2):
                idx = k * 2 + l
                for coeff, block_idx in L_defs[i][k][l]:
                    for entry in range(4):
                        L_vec[idx][block_idx * 4 + entry] = coeff
                for coeff, block_idx in R_defs[i][k][l]:
                    for entry in range(4):
                        R_vec[idx][block_idx * 4 + entry] = coeff
        
        L_vecs.append(L_vec)
        R_vecs.append(R_vec)
    
    # Now compose:
    # For block product i and Strassen product j:
    #   u_j is 4-dim.
    #   The left factor is sum_{idx} u_j[idx] * L_vecs[i][idx].
    
    U = []
    V = []
    W = []
    
    # We need to generate the 49 products for one 4x4 x 4x4.
    # Then we use two copies for the 4x4 x 4x8.
    
    # First, generate the 49 products for 4x4 x 4x4.
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    for i in range(7):
        for j in range(7):
            u_j, v_j = strassen_2x2[j]
            
            left = [0.0] * 16
            right = [0.0] * 16
            
            for idx in range(4):
                left = [left[k] + u_j[idx] * L_vecs[i][idx][k] for k in range(16)]
                right = [right[k] + v_j[idx] * R_vecs[i][idx][k] for k in range(16)]
            
            U_4x4.append(left)
            V_4x4.append(right)
            
            # W_4x4 is determined by the reconstruction.
            # c_ij = sum_{i,j} recon_4x4[i,j,k] * P_k
            # where k is the product index (i*7 + j).
            
            # We need to define recon_4x4.
            # recon_4x4 is a 4x4x49 tensor.
            # recon_4x4[i,j,k] = coefficient of product k in c_ij.
            
            # This is complex. Let's just output the U and V for now, and assume W is correct.
            # Actually, the evaluator checks the tensor decomposition, so W must be correct.
            
            # I will fall back to schoolbook to avoid bugs.
            
            pass
    
    # I will return the schoolbook rank 128.
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
