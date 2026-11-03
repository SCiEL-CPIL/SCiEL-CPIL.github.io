"""Matrix multiplication 4x4x8 - Strassen-based rank reduction."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 -> 4x8.
    
    Strategy: 
    1. Use recursive Strassen to get a rank-49 decomposition for 4x4 * 4x4 -> 4x4.
    2. Extend to 4x4 * 4x8 by applying the same A-side factors to two B-halves.
    Total rank: 49 * 2 = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build rank-49 decomposition for 4x4 * 4x4 -> 4x4 using recursive Strassen
    # A is 4x4, B is 4x4, C is 4x4
    # U: (16, 49), V: (16, 49), W: (16, 49)
    
    # Recursive Strassen construction
    # Level 1: Split 4x4 into 2x2 blocks of 2x2 matrices
    # Use Strassen's 7 products at the block level
    # Each block product is a 2x2 * 2x2 -> 2x2 multiplication
    # Level 2: Use Strassen's 7 products for each 2x2 * 2x2 -> 2x2
    
    # Strassen products for 2x2 * 2x2 -> 2x2 (rank 7)
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x2 -> 2x2, we can directly enumerate the rank-7 decomposition
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # where U2[i*2+j, c] corresponds to A[i,j] in product c
    #       V2[j*2+k, c] corresponds to B[j,k] in product c  
    #       W2[k*2+i, c] corresponds to C[i,k] in product c
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A00 + A11)(B00 + B11) -> C00(+1), C11(+1)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10 + A11)B00 -> C10(+1)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1
    
    # M3: A00(B10 - B11) -> C01(+1)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1
    
    # M4: A11(B01 - B00) -> C00(+1), C10(+1)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5: (A00 + A01)B11 -> C00(-1), C01(+1)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M6: (A10 - A00)(B00 + B01) -> C11(+1)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7: (A01 - A11)(B10 + B11) -> C00(-1)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1
    
    # Now build 4x4 * 4x4 -> 4x4 using Strassen at block level
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]] (each 2x2)
    # C = [[C00, C01], [C10, C11]] (each 2x2)
    
    # Block-level Strassen products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Each Mi is a 2x2 matrix product, decomposed with rank 7
    # Total rank: 7 * 7 = 49
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block index mapping:
    # A blocks: 0=A00(rows 0-1, cols 0-1), 1=A01(rows 0-1, cols 2-3)
    #           2=A10(rows 2-3, cols 0-1), 3=A11(rows 2-3, cols 2-3)
    # B blocks: 0=B00(rows 0-1, cols 0-1), 1=B01(rows 0-1, cols 2-3)
    #           2=B10(rows 2-3, cols 0-1), 3=B11(rows 2-3, cols 2-3)
    # C blocks: 0=C00(rows 0-1, cols 0-1), 1=C01(rows 0-1, cols 2-3)
    #           2=C10(rows 2-3, cols 0-1), 3=C11(rows 2-3, cols 2-3)
    
    # For block b, the local indices within the 2x2 block map to global indices:
    # Block 0: rows 0-1, cols 0-1 -> local (r,c) -> global (r, c)
    # Block 1: rows 0-1, cols 2-3 -> local (r,c) -> global (r, c+2)
    # Block 2: rows 2-3, cols 0-1 -> local (r,c) -> global (r+2, c)
    # Block 3: rows 2-3, cols 2-3 -> local (r,c) -> global (r+2, c+2)
    
    def block_offset(block_idx, local_row, local_col, n_side=2):
        """Convert local block coordinates to global coordinates."""
        if block_idx == 0:
            return local_row, local_col
        elif block_idx == 1:
            return local_row, local_col + 2
        elif block_idx == 2:
            return local_row + 2, local_col
        elif block_idx == 3:
            return local_row + 2, local_col + 2
    
    # Define the 7 Strassen products at the block level
    # Each product is a list of (A_block, coeff) and (B_block, coeff)
    strassen_products = [
        # (A_terms, B_terms) where each term is (block_idx, coeff)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)]),             # M2
        ([(0, 1)], [(2, 1), (3, -1)]),            # M3
        ([(3, 1)], [(1, 1), (0, -1)]),            # M4
        ([(0, 1), (1, 1)], [(3, 1)]),             # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),    # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),    # M7
    ]
    
    # C contributions: which C blocks each Mi contributes to, and with what coefficient
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_contributions = {
        0: [(0, 1), (3, 1), (4, -1), (6, -1)],   # C00
        1: [(2, 1), (4, 1)],                       # C01
        2: [(1, 1), (3, 1)],                       # C10
        3: [(0, 1), (1, -1), (2, 1), (5, 1)],     # C11
    }
    
    # Build the 49-rank decomposition
    col = 0
    for prod_idx in range(7):
        A_terms, B_terms = strassen_products[prod_idx]
        
        # For each of the 7 inner Strassen products
        for inner_idx in range(7):
            # U: combine A blocks with coefficients, then apply inner U2
            for local_r in range(4):
                local_c = local_r % 2  # local column within 2x2 block
                local_r2 = local_r // 2  # local row within 2x2 block
                
                # Accumulate contribution from each A block
                u_val = 0.0
                for a_block, a_coeff in A_terms:
                    gr, gc = block_offset(a_block, local_r2, local_c)
                    global_idx = gr * 4 + gc
                    u_val += a_coeff * U2[local_r, inner_idx]
                
                U4[global_idx, col] = u_val
            
            # V: combine B blocks with coefficients, then apply inner V2
            for local_r in range(4):
                local_c = local_r % 2
                local_r2 = local_r // 2
                
                v_val = 0.0
                for b_block, b_coeff in B_terms:
                    gr, gc = block_offset(b_block, local_r2, local_c)
                    global_idx = gr * 4 + gc
                    v_val += b_coeff * V2[local_r, inner_idx]
                
                V4[global_idx, col] = v_val
            
            # W: For each C block that this product contributes to
            for c_block, c_coeff in c_contributions[0]:
                pass  # This approach won't work directly; need to handle C contributions properly
            
            col += 1
    
    # The above approach has issues with W. Let me redo this more carefully.
    # 
    # Actually, let me think about this differently.
    # 
    # The tensor decomposition for 4x4 * 4x4 -> 4x4 with rank 49:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # 
    # In Strassen, we compute:
    # M1 = (A00+A11)(B00+B11), etc.
    # Each Mi is a 2x2 matrix, computed as Mi = sum_{inner} u_inner * A' * v_inner * B' * w_inner
    # where A', B' are linear combinations of the 2x2 blocks.
    # 
    # Then C blocks are linear combinations of the Mi.
    # 
    # So for the full 4x4 multiplication:
    # C[i,k] = sum_{prod} (sum_{c_block containing (i,k)} c_coeff) * Mi[i_local, k_local]
    #        = sum_{prod} sum_{inner} (sum_{c_block} c_coeff) * u_inner * A'[i_local, j_local] * v_inner * B'[j_local, k_local] * w_inner
    # 
    # This is getting complex. Let me just directly construct the rank-98 decomposition
    # for 4x4 * 4x8 using a simpler approach.
    
    # Alternative: Use schoolbook for the 4x4 * 4x8 directly but with Strassen for the 4x4 * 4x4 part.
    # 
    # Actually, let me just build the rank-98 decomposition by taking two copies of a rank-49 
    # 4x4*4x4 decomposition and applying them to the two halves of B and C.
    
    # For simplicity and correctness, let me construct the decomposition directly.
    # 
    # The key insight: 4x4 * 4x8 can be seen as 4x4 * 4x4 for two separate B matrices.
    # If we have a rank-r decomposition for 4x4 * 4x4 -> 4x4, then we can get a rank-2r
    # decomposition for 4x4 * 44x8 by using the same U factors and duplicating V and W.
    
    # Let me construct a rank-49 4x4*4x4 decomposition properly.
    
    def _build_strassen_4x4():
        """Build rank-49 decomposition for 4x4 * 4x4 -> 4x4."""
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Block-level Strassen with inner Strassen
        # 7 outer products, each decomposed into 7 inner products = 49 total
        
        # For each outer product p (0..6) and inner product q (0..6):
        # The A factor is a linear combination of 2x2 blocks
        # The B factor is a linear combination of 2x2 blocks  
        # The C factor combines contributions to the appropriate C blocks
        
        # Let me define the structure more explicitly.
        
        # Outer Strassen products (block level):
        # p=0: M1 = (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
        # p=1: M2 = (A10+A11)B00 -> contributes to C10(+1)
        # p=2: M3 = A00(B10-B11) -> contributes to C01(+1)
        # p=3: M4 = A11(B01-B00) -> contributes to C00(+1), C10(+1)
        # p=4: M5 = (A00+A01)B11 -> contributes to C00(-1), C01(+1)
        # p=5: M6 = (A10-A00)(B00+B01) -> contributes to C11(+1)
        # p=6: M7 = (A01-A11)(B10+B11) -> contributes to C00(-1)
        
        outer_A = [
            [(0, 1), (3, 1)],  # p=0
            [(2, 1), (3, 1)],  # p=1
            [(0, 1)],           # p=2
            [(3, 1)],           # p=3
            [(0, 1), (1, 1)],  # p=4
            [(2, 1), (0, -1)], # p=5
            [(1, 1), (3, -1)], # p=6
        ]
        outer_B = [
            [(0, 1), (3, 1)],  # p=0
            [(0, 1)],           # p=1
            [(2, 1), (3, -1)], # p=2
            [(1, 1), (0, -1)], # p=3
            [(3, 1)],           # p=4
            [(0, 1), (1, 1)],  # p=5
            [(2, 1), (3, 1)],  # p=6
        ]
        outer_C = [
            [(0, 1), (3, 1)],  # p=0: C00(+1), C11(+1)
            [(2, 1)],           # p=1: C10(+1)
            [(1, 1)],           # p=2: C01(+1)
            [(0, 1), (2, 1)],  # p=3: C00(+1), C10(+1)
            [(0, -1), (1, 1)], # p=4: C00(-1), C01(+1)
            [(3, 1)],           # p=5: C11(+1)
            [(0, -1)],          # p=6: C00(-1)
        ]
        
        # Inner Strassen (2x2 * 2x2 -> 2x2)
        inner_U = U2
        inner_V = V2
        inner_W = W2
        
        col = 0
        for p in range(7):
            for q in range(7):
                # U: for each global index, sum over A blocks
                for gr in range(4):
                    for gc in range(4):
                        u_val = 0.0
                        for a_block, a_coeff in outer_A[p]:
                            # Determine local position within block
                            if a_block == 0:
                                lr, lc = gr, gc
                            elif a_block == 1:
                                lr, lc = gr, gc - 2
                            elif a_block == 2:
                                lr, lc = gr - 2, gc
                            else:  # a_block == 3
                                lr, lc = gr - 2, gc - 2
                            
                            if 0 <= lr < 2 and 0 <= lc < 2:
                                u_val += a_coeff * inner_U[lr * 2 + lc, q]
                        
                        U[gr * 4 + gc, col] = u_val
                
                # V: for each global index, sum over B blocks
                for gr in range(4):
                    for gc in range(4):
                        v_val = 0.0
                        for b_block, b_coeff in outer_B[p]:
                            if b_block == 0:
                                lr, lc = gr, gc
                            elif b_block == 1:
                                lr, lc = gr, gc - 2
                            elif b_block == 2:
                                lr, lc = gr - 2, gc
                            else:  # b_block == 3
                                lr, lc = gr - 2, gc - 2
                            
                            if 0 <= lr < 2 and 0 <= lc < 2:
                                v_val += b_coeff * inner_V[lr * 2 + lc, q]
                        
                        V[gr * 4 + gc, col] = v_val
                
                # W: for each global index, sum over C blocks
                for gr in range(4):
                    for gc in range(4):
                        w_val = 0.0
                        for c_block, c_coeff in outer_C[p]:
                            if c_block == 0:
                                lr, lc = gr, gc
                            elif c_block == 1:
                                lr, lc = gr, gc - 2
                            elif c_block == 2:
                                lr, lc = gr - 2, gc
                            else:  # c_block == 3
                                lr, lc = gr - 2, gc - 2
                            
                            if 0 <= lr < 2 and 0 <= lc < 2:
                                w_val += c_coeff * inner_W[lc * 2 + lr, q]
                        
                        W[gr * 4 + gc, col] = w_val
                
                col += 1
        
        return U, V, W
    
    # Build the 4x4*4x4 rank-49 decomposition
    U4, V4, W4 = _build_strassen_4x4()
    
    # Now extend to 4x4 * 4x8 -> 4x8
    # B is 4x8, split into B1 (4x4) and B2 (4x4)
    # C is 4x8, split into C1 (4x4) and C2 (4x4)
    # C1 = A * B1, C2 = A * B2
    # Use the same U4 for both, and V4/W4 for each half
    
    rank_total = 98
    U = np.zeros((16, rank_total))
    V = np.zeros((32, rank_total))
    W = np.zeros((32, rank_total))
    
    # First 49 terms: for B1 and C1
    for c in range(49):
        U[:, c] = U4[:, c]
        V[:16, c] = V4[:, c]  # B1 part
        W[:16, c] = W4[:, c]  # C1 part
    
    # Next 49 terms: for B2 and C2
    for c in range(49):
        U[:, 49 + c] = U4[:, c]
        V[16:32, 49 + c] = V4[:, c]  # B2 part
        W[16:32, 49 + c] = W4[:, c]  # C2 part
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

