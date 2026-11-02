import time
import numpy as np

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Rank 98 decomposition using Block Strassen
    # We split the 4x8 matrix B into two 4x4 matrices B1 and B2.
    # C = A * B = A * [B1, B2] = [A*B1, A*B2]
    # We use Strassen for 4x4x4 which has rank 49.
    # Total rank = 49 + 49 = 98.
    
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen terms for 2x2x2
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # s1 = (x00+x11)(y00+y11)
    # s2 = (x10+x11)y00
    # s3 = x00(y10-y11)
    # s4 = x11(y01-y00)
    # s5 = (x00+x01)y11
    # s6 = (x10-x00)(y00+y01)
    # s7 = (x01-x11)(y10+y11)
    
    # Reconstruction for 2x2x2:
    # c00 = s1 + s4 - s5 - s7
    # c01 = s3 + s5
    # c10 = s2 + s4
    # c11 = s1 - s2 + s3 + s7
    
    # For 4x4x4, we treat A and B as 2x2 block matrices with 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # 7 Macro terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a product of two 2x2 block matrices.
    # We decompose each Mi using Strassen 2x2x2 on the blocks.
    # This yields 7 micro-terms for each macro-term.
    # Total 49 terms for 4x4x4.
    
    # Helper to add a rank-1 term
    # u_coeffs: dict {(i,j): val} for A
    # v_coeffs: dict {(j,k): val} for B
    # w_coeffs: dict {(i,k): val} for C
    # col: current column index
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        for (i, j), ua in u_coeffs.items():
            U[i*m + j, col] += ua
        for (j, k), vb in v_coeffs.items():
            V[j*p + k, col] += vb
        for (i, k), wc in w_coeffs.items():
            W[k*n + i, col] += wc

    # Define block indices for 4x4 matrix
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    # For a block ID b, the local indices (r, c) in 0..1 map to global:
    # b=0: i=r, j=c
    # b=1: i=r, j=c+2
    # b=2: i=r+2, j=c
    # b=3: i=r+2, j=c+2
    
    def get_global_idx(block_id, r, c):
        if block_id == 0:
            return (r, c)
        elif block_id == 1:
            return (r, c+2)
        elif block_id == 2:
            return (r+2, c)
        elif block_id == 3:
            return (r+2, c+2)
        else:
            raise ValueError("Invalid block ID")

    # Macro term definitions:
    # X_blocks: list of 4 dicts {(block_id): coeff} for X00, X01, X10, X11
    # Y_blocks: list of 4 dicts {(block_id): coeff} for Y00, Y01, Y10, Y11
    
    macro_terms = [
        # 1: (A00+A11)(B00+B11)
        (
            [{0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}], # X
            [{0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}]  # Y
        ),
        # 2: (A10+A11)B00
        (
            [{2:1, 3:1}, {2:1, 3:1}, {2:1, 3:1}, {2:1, 3:1}], # X
            [{0:1}, {0:1}, {0:1}, {0:1}]                       # Y
        ),
        # 3: A00(B10-B11)
        (
            [{0:1}, {0:1}, {0:1}, {0:1}],                      # X
            [{2:1, 3:-1}, {2:1, 3:-1}, {2:1, 3:-1}, {2:1, 3:-1}] # Y
        ),
        # 4: A11(B01-B00)
        (
            [{3:1}, {3:1}, {3:1}, {3:1}],                      # X
            [{1:1, 0:-1}, {1:1, 0:-1}, {1:1, 0:-1}, {1:1, 0:-1}] # Y
        ),
        # 5: (A00+A01)B11
        (
            [{0:1, 1:1}, {0:1, 1:1}, {0:1, 1:1}, {0:1, 1:1}], # X
            [{3:1}, {3:1}, {3:1}, {3:1}]                       # Y
        ),
        # 6: (A10-A00)(B00+B01)
        (
            [{2:1, 0:-1}, {2:1, 0:-1}, {2:1, 0:-1}, {2:1, 0:-1}], # X
            [{0:1, 1:1}, {0:1, 1:1}, {0:1, 1:1}, {0:1, 1:1}]     # Y
        ),
        # 7: (A01-A11)(B10+B11)
        (
            [{1:1, 3:-1}, {1:1, 3:-1}, {1:1, 3:-1}, {1:1, 3:-1}], # X
            [{2:1, 3:1}, {2:1, 3:1}, {2:1, 3:1}, {2:1, 3:1}]     # Y
        )
    ]
    
    # Strassen micro-term definitions for 2x2x2
    # Each term is (u_coeffs_local, v_coeffs_local, w_coeffs_local)
    # Local indices 0,1 correspond to block positions 0,1 in the 2x2 block matrix
    # We need to map these to the actual blocks in the 4x4 matrix.
    # The "local" 2x2 matrix for a macro term X is composed of 4 blocks from the 4x4 A.
    # Let's denote the blocks of X as X_00, X_01, X_10, X_11.
    # Each X_ij is a linear combination of A's blocks.
    # The Strassen term involves products of these X_ij and Y_ij.
    # e.g., s1 = (X_00 + X_11)(Y_00 + Y_11)
    # This expands to (X_00 Y_00 + X_00 Y_11 + X_11 Y_00 + X_11 Y_11).
    # Since X_00 and Y_00 are themselves linear combinations of A and B blocks,
    # we can just add the coefficients to U and V.
    
    # We will iterate through the 7 macro terms.
    # For each macro term, we iterate through the 7 Strassen micro-terms.
    # Total 49 terms per 4x4x4.
    
    # Micro-terms for 2x2x2 (indices 0,1 for rows/cols of the 2x2 block matrix)
    # s1: (x00+x11)(y00+y11)
    # s2: (x10+x11)y00
    # s3: x00(y10-y11)
    # s4: x11(y01-y00)
    # s5: (x00+x01)y11
    # s6: (x10-x00)(y00+y01)
    # s7: (x01-x11)(y10+y11)
    
    # Reconstruction coefficients for C blocks (00, 01, 10, 11)
    # c00: s1 + s4 - s5 - s7
    # c01: s3 + s5
    # c10: s2 + s4
    # c11: s1 - s2 + s3 + s7
    
    # We will generate the U, V, W contributions for each of the 49 terms.
    
    # Helper to expand a term like (X_block_dict) * (Y_block_dict)
    # X_block_dict is {(block_id_A): coeff}
    # Y_block_dict is {(block_id_B): coeff}
    # This term contributes to U, V, W.
    # U gets coeff for A blocks, V gets coeff for B blocks.
    # W gets the product of coeffs for C blocks?
    # No, the term is rank 1.
    # U[i,j] = sum of coeffs for A
    # V[j,k] = sum of coeffs for B
    # W[i,k] = 1 (or appropriate scaling)
    
    # Actually, the rank-1 term is:
    # U = sum_{a} c_a A_a
    # V = sum_{b} d_b B_b
    # W = sum_{c} e_c C_c
    # Such that sum_k U_{i,k} V_{k,j} = sum_{a,b} c_a d_b A_{i,a} B_{a,j}
    # And this must equal the contribution to C.
    
    # For Strassen, the terms are constructed such that the product of the linear forms gives the bilinear form.
    
    # Let's construct the 49 terms for the first 4x4x4 (B cols 0-3).
    
    col_idx = 0
    
    for macro_idx in range(7):
        X_blocks, Y_blocks = macro_terms[macro_idx]
        # X_blocks[0] is X_00, X_blocks[1] is X_01, X_blocks[2] is X_10, X_blocks[3] is X_11
        # Each is a dict {(block_id): coeff}
        
        # Generate 7 micro-terms
        # We define the 7 micro-terms as lists of (X_part, Y_part)
        # where X_part is a dict {(local_idx): coeff} for the 2x2 block matrix X
        # and Y_part is a dict {(local_idx): coeff} for the 2x2 block matrix Y
        
        micro_terms = [
            # 1
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            # 2
            ({2: 1, 3: 1}, {0: 1}),
            # 3
            ({0: 1}, {2: 1, 3: -1}),
            # 4
            ({3: 1}, {1: 1, 0: -1}),
            # 5
            ({0: 1, 1: 1}, {3: 1}),
            # 6
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            # 7
            ({1: 1, 3: -1}, {2: 1, 3: 1})
        ]
        
        for micro_idx in range(7):
            x_local, y_local = micro_terms[micro_idx]
            
            # Expand x_local and y_local into global block coefficients
            # x_local is {(local_idx): coeff}
            # local_idx 0 -> X_00, 1 -> X_01, 2 -> X_10, 3 -> X_11
            # X_00 is X_blocks[0], etc.
            
            u_coeffs = {}
            v_coeffs = {}
            
            for l_idx, coeff in x_local.items():
                block_dict = X_blocks[l_idx]
                for b_id, b_coeff in block_dict.items():
                    # b_id is the block ID in the 4x4 A matrix
                    # We need to add coeff * b_coeff to U for all entries in this block
                    # But wait, U is a matrix of size 16x98.
                    # The "term" is a rank-1 tensor.
                    # U[:, col] should be the vector representing the linear form on A.
                    # The linear form is sum_{i,j} U_{i,j} A_{i,j}.
                    # If we have a block A_b, and we want to include it with coefficient C,
                    # we add C to U_{i,j} for all (i,j) in block b.
                    
                    for r in range(2):
                        for c in range(2):
                            i, j = get_global_idx(b_id, r, c)
                            key = (i, j)
                            u_coeffs[key] = u_coeffs.get(key, 0) + coeff * b_coeff
            
            for l_idx, coeff in y_local.items():
                block_dict = Y_blocks[l_idx]
                for b_id, b_coeff in block_dict.items():
                    # b_id is the block ID in the 4x4 B matrix
                    # For the first 4x4x4, B cols are 0-3.
                    # get_global_idx gives (r, c) in 0..3 range.
                    for r in range(2):
                        for c in range(2):
                            j, k = get_global_idx(b_id, r, c)
                            key = (j, k)
                            v_coeffs[key] = v_coeffs.get(key, 0) + coeff * b_coeff
            
            # Now we have u_coeffs and v_coeffs for this rank-1 term.
            # We need to determine w_coeffs.
            # The product of the two linear forms gives a bilinear form on A and B.
            # This bilinear form is a linear combination of the entries of C.
            # We need to find the coefficients for each C_{i,k}.
            
            # The bilinear form is:
            # sum_{i,j} u_coeffs[i,j] A_{i,j} * sum_{j',k} v_coeffs[j',k] B_{j',k}
            # = sum_{i,j,j',k} u_coeffs[i,j] v_coeffs[j',k] A_{i,j} B_{j',k}
            # We only care about terms where j == j' (since A is 4x4 and B is 4x4, they multiply to 4x4).
            # So:
            # sum_{i,j,k} (sum_j u_coeffs[i,j] v_coeffs[j,k]) A_{i,j} B_{j,k}
            # The coefficient of C_{i,k} in the expansion is 1.
            # So the contribution to C_{i,k} from this term is sum_j u_coeffs[i,j] v_coeffs[j,k].
            
            w_coeffs = {}
            for i in range(4):
                for k in range(4):
                    val = 0
                    for j in range(4):
                        val += u_coeffs.get((i, j), 0) * v_coeffs.get((j, k), 0)
                    if val != 0:
                        w_coeffs[(i, k)] = val
            
            # Now add this term to U, V, W
            for (i, j), ua in u_coeffs.items():
                U[i*m + j, col_idx] += ua
            for (j, k), vb in v_coeffs.items():
                V[j*p + k, col_idx] += vb
            for (i, k), wc in w_coeffs.items():
                W[k*n + i, col_idx] += wc
            
            col_idx += 1
            
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now for the second 4x4x4 (B cols 4-7)
    # We repeat the same process, but shift the B indices by 4.
    # And add to U, V, W at columns 49..97.
    
    for macro_idx in range(7):
        X_blocks, Y_blocks = macro_terms[macro_idx]
        
        micro_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {2: 1, 3: -1}),
            ({3: 1}, {1: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1})
        ]
        
        for micro_idx in range(7):
            x_local, y_local = micro_terms[micro_idx]
            
            u_coeffs = {}
            v_coeffs = {}
            
            for l_idx, coeff in x_local.items():
                block_dict = X_blocks[l_idx]
                for b_id, b_coeff in block_dict.items():
                    for r in range(2):
                        for c in range(2):
                            i, j = get_global_idx(b_id, r, c)
                            key = (i, j)
                            u_coeffs[key] = u_coeffs.get(key, 0) + coeff * b_coeff
            
            for l_idx, coeff in y_local.items():
                block_dict = Y_blocks[l_idx]
                for b_id, b_coeff in block_dict.items():
                    for r in range(2):
                        for c in range(2):
                            j, k = get_global_idx(b_id, r, c)
                            # Shift k by 4 for the second half of B
                            k_new = k + 4
                            key = (j, k_new)
                            v_coeffs[key] = v_coeffs.get(key, 0) + coeff * b_coeff
            
            w_coeffs = {}
            for i in range(4):
                for k in range(4):
                    val = 0
                    for j in range(4):
                        val += u_coeffs.get((i, j), 0) * v_coeffs.get((j, k), 0)
                    if val != 0:
                        # Shift k by 4 for C as well
                        k_new = k + 4
                        w_coeffs[(i, k_new)] = val
            
            for (i, j), ua in u_coeffs.items():
                U[i*m + j, col_idx] += ua
            for (j, k), vb in v_coeffs.items():
                V[j*p + k, col_idx] += vb
            for (i, k), wc in w_coeffs.items():
                W[k*n + i, col_idx] += wc
            
            col_idx += 1
            
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
