import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i,j,k] = 1 for all i,j,k (the matrix multiplication tensor for 0-1 A,B gives C[i,k]=sum_j A[i,j]B[j,k])
    # Actually the matrix multiplication tensor T[i,j,k] has T[i,j,k]=1 for all i in [n], j in [m], k in [p]
    # because C[i,k] = sum_j A[i,j]*B[j,k], so the tensor is all-ones in the appropriate sense.
    # Let's verify: for random A (n x m) and B (m x p), C = A @ B
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]
    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    # Check with specific values
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    # Returns 7 triples of 2x2 matrices (U_r, V_r, W_r) for 2x2x2 Strassen
    # C = A*B where A, B, C are 2x2
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor indices: i in {0,1} (row of A / row of C), j in {0,1} (col of A / row of B), k in {0,1} (col of B / col of C)
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U_list = []
    V_list = []
    W_list = []
    
    # M1 = (a11+a22)(b11+b22)
    U1 = np.array([[1, 0], [0, 1]], dtype=float)  # U[i*2+j, r]: i=0,j=0 -> row0; i=0,j=1 -> row1; i=1,j=0 -> row2; i=1,j=1 -> row3
    # U maps (i,j) -> coefficient for A[i,j]
    # For M1: A factor is (a11+a22) = A[:,0] + A[:,1]... no.
    # A[i,j] coefficient: for i=0,j=0: 1; i=0,j=1: 0; i=1,j=0: 0; i=1,j=1: 1
    # So U[:,r] = [1, 0, 0, 1] (flattened i*m+j with m=2: index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    U1 = np.array([1, 0, 0, 1], dtype=float)
    # B factor: (b11+b22) -> b11 is B[0,0], b22 is B[1,1]
    # V[j*p+k, r] with p=2: index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    V1 = np.array([1, 0, 0, 1], dtype=float)
    # C factor: C[0,0] += M1, C[1,1] += M1
    # W[k*n+i, r] with n=2: index 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
    W1 = np.array([1, 0, 0, 1], dtype=float)
    U_list.append(U1); V_list.append(V1); W_list.append(W1)
    
    # M2 = (a21+a22)(b11)
    U2 = np.array([0, 0, 1, 1], dtype=float)  # a21: (1,0)->idx2, a22: (1,1)->idx3
    V2 = np.array([1, 0, 0, 0], dtype=float)  # b11: (0,0)->idx0
    # C[1,0] += M2
    W2 = np.array([0, 0, 1, 0], dtype=float)  # (k=1,i=1)->idx3? No: W[k*n+i], n=2. (k=1,i=1) = 1*2+1=3
    # Wait: C[1,0] means row 1, col 0 of C. i=1 (row of C), k=0 (col of C). W[k*n+i] = W[0*2+1] = W[1]
    W2 = np.array([0, 1, 0, 0], dtype=float)
    U_list.append(U2); V_list.append(V2); W_list.append(W2)
    
    # M3 = (a11)(b21+b22)
    U3 = np.array([1, 0, 0, 0], dtype=float)  # a11: (0,0)->idx0
    V3 = np.array([0, 0, 1, 1], dtype=float)  # b21: (1,0)->idx2, b22: (1,1)->idx3
    # C[0,1] += M3: i=0, k=1. W[k*n+i] = W[1*2+0] = W[2]
    W3 = np.array([0, 0, 1, 0], dtype=float)
    U_list.append(U3); V_list.append(V3); W_list.append(W3)
    
    # M4 = (a22)(b12-b22)
    U4 = np.array([0, 0, 0, 1], dtype=float)  # a22: (1,1)->idx3
    V4 = np.array([0, 1, 0, -1], dtype=float)  # b12: (0,1)->idx1, -b22: (1,1)->idx3
    # C[1,1] += M4: i=1, k=1. W[k*n+i] = W[1*2+1] = W[3]
    W4 = np.array([0, 0, 0, 1], dtype=float)
    U_list.append(U4); V_list.append(V4); W_list.append(W4)
    
    # M5 = (a11+a12)(b22)
    U5 = np.array([1, 1, 0, 0], dtype=float)  # a11: idx0, a12: (0,1)->idx1
    V5 = np.array([0, 0, 0, 1], dtype=float)  # b22: (1,1)->idx3
    # C[0,1] += M5: i=0, k=1. W[2]
    W5 = np.array([0, 0, 1, 0], dtype=float)
    U_list.append(U5); V_list.append(V5); W_list.append(W5)
    
    # M6 = (a21-a11)(b11+b12)
    U6 = np.array([-1, 0, 1, 0], dtype=float)  # -a11: idx0, a21: idx2
    V6 = np.array([1, 1, 0, 0], dtype=float)  # b11: idx0, b12: idx1
    # C[1,0] += M6: i=1, k=0. W[1]
    W6 = np.array([0, 1, 0, 0], dtype=float)
    U_list.append(U6); V_list.append(V6); W_list.append(W6)
    
    # M7 = (a12-a22)(b21-b11)
    U7 = np.array([0, 1, 0, -1], dtype=float)  # a12: idx1, -a22: idx3
    V7 = np.array([-1, 0, 1, 0], dtype=float)  # -b11: idx0, b21: idx2
    # C[0,0] += M7: i=0, k=0. W[0]
    W7 = np.array([1, 0, 0, 0], dtype=float)
    U_list.append(U7); V_list.append(V7); W_list.append(W7)
    
    return U_list, V_list, W_list

def _build_4x4x4_strassen():
    # Build 49 terms for 4x4x4 using nested Strassen
    # 4x4 blocks: A = [[A11,A12],[A21,A22]] where each block is 2x2
    # Outer Strassen: 7 terms, each involving 2x2 matrix products
    # For each outer term, the "scalar" multiplication is actually 2x2x2, done via inner Strassen (7 terms)
    # Total: 7 * 7 = 49 terms
    
    U_outer, V_outer, W_outer = _strassen_2x2()
    U_inner, V_inner, W_inner = _strassen_2x2()
    
    all_U = []
    all_V = []
    all_W = []
    
    for o in range(7):
        # U_outer[o] is a 4-vector: coefficients for A[0,0], A[0,1], A[1,0], A[1,1] (each 2x2)
        # Flatten: for a 4x4 A, index = block_row * 2 + row_in_block, block_col * 2 + col_in_block
        # U[i*4+j, r] for 4x4: i in 0..3, j in 0..3
        # Block (br, bc) of A covers rows br*2..br*2+1, cols bc*2..bc*2+1
        # U_outer[o] = [c00, c01, c10, c11] where c_rc is coeff for A_block[rc]
        
        for inner_idx in range(4):
            # inner_idx: 0=(br=0,bc=0), 1=(br=0,bc=1), 2=(br=1,bc=0), 3=(br=1,bc=1)
            br = inner_idx // 2
            bc = inner_idx % 2
            coeff = U_outer[o][inner_idx]
            if coeff == 0:
                continue
            
            for ii in range(7):
                # For this inner Strassen term, we need to embed 2x2x2 into 4x4x4
                # U_inner[ii] is 4-vector for 2x2 A block: indices (0,0),(0,1),(1,0),(1,1) within block
                # In 4x4: row = br*2 + inner_row, col = bc*2 + inner_col
                U_term = np.zeros(16)
                for ir in range(2):
                    for ic in range(2):
                        inner_idx_val = ir * 2 + ic  # within 2x2 block
                        val = U_inner[ii][inner_idx_val]
                        if val != 0:
                            r4 = br * 2 + ir
                            c4 = bc * 2 + ic
                            U_term[r4 * 4 + c4] = coeff * val
                
                # V: B block (br2, bc2) where br2 corresponds to block_row of B (= block_col of A side? No)
                # In outer Strassen: M = A_factor * B_factor
                # A_factor is 2x2 matrix, B_factor is 2x2 matrix
                # The "j" index in outer is: for A, j indexes columns; for B, j indexes rows
                # In 4x4: A is 4x4, B is 4x4
                # A block (br, bc) -> A rows br*2.., cols bc*2..
                # B block: the "j" in B[j,k] corresponds to A's column j. So B's row j.
                # In outer Strassen for 4x4: A[br, bc] * B[bc, br2] where br2 is B's block column... 
                # Actually: C[br, bc2] gets contribution from A[br, bc] * B[bc, bc2]
                # So for outer term o, A factor uses block column bc, B factor uses block row bc
                # B block (bc, br2) where br2 ranges over {0,1}
                
                # Wait, I need to be more careful. Let me re-derive.
                # Outer: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
                # C = A*B, C[br,bc2] = sum_bc A[br,bc]*B[bc,bc2]
                # Strassen outer terms:
                # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
                # M2 = (A10+A11)(B00) -> contributes to C10
                # M3 = (A00)(B10+B11) -> contributes to C01
                # M4 = (A11)(B01-B11) -> contributes to C11
                # M5 = (A00+A01)(B11) -> contributes to C01
                # M6 = (A10-A00)(B00+B01) -> contributes to C10
                # M7 = (A01-A11)(B10-B00) -> contributes to C00
                
                # So for outer term o, the A block used has (br, bc) and B block has (bc, bc2)
                # From U_outer[o], we can figure out which A blocks are involved
                # From V_outer[o], we can figure out which B blocks are involved
                # But U_outer[o] has 4 entries for 4 A blocks, V_outer[o] has 4 entries for 4 B blocks
                # We need to pair them: for each (A_block, B_block) pair with nonzero product
                
                # Let me just use the structure directly
                pass
    
    # This nested approach is getting complicated. Let me just do it more directly.
    # For now, let's just return the schoolbook and use a simpler improvement.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook
    U, V, W = _make_schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p  # 100
    
    # Try a structured approach: 
    # 4x5x5 = 4x4x4 + 4x4x1 + 4x1x4 + 4x1x1
    # Use Strassen for 4x4x4 (if we can build it), schoolbook for the rest
    # 4x4x4 Strassen: 49 terms (nested 2x2 Strassen)
    # 4x4x1: 16 terms, 4x1x4: 16 terms, 4x1x1: 4 terms
    # Total: 49 + 16 + 16 + 4 = 85
    
    # Let me build the 4x4x4 Strassen properly
    U2, V2, W2 = _strassen_2x2()
    
    # For 4x4x4:
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], each 2x2
    # C[br,bc] = sum_bc2 A[br,bc2] * B[bc2,bc]
    
    # Outer Strassen gives 7 products of 2x2 matrices
    # Each 2x2 matrix product is decomposed via inner Strassen (7 terms)
    # Total 49 rank terms for 4x4x4
    
    U444 = []
    V444 = []
    W444 = []
    
    # Outer Strassen terms for 4x4 (blocks are 2x2)
    # For each outer term, identify which A-block and B-block it involves
    # M1 = (A00+A11)(B00+B11): A blocks (0,0),(1,1); B blocks (0,0),(1,1)
    # M2 = (A10+A11)(B00): A blocks (1,0),(1,1); B blocks (0,0)
    # M3 = (A00)(B10+B11): A blocks (0,0); B blocks (1,0),(1,1)
    # M4 = (A11)(B01-B11): A blocks (1,1); B blocks (0,1),(1,1)
    # M5 = (A00+A01)(B11): A blocks (0,0),(0,1); B blocks (1,1)
    # M6 = (A10-A00)(B00+B01): A blocks (1,0),(0,0); B blocks (0,0),(0,1)
    # M7 = (A01-A11)(B10-B00): A blocks (0,1),(1,1); B blocks (1,0),(0,0)
    
    outer_terms = [
        # (a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs)
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),   # M1
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0)], [1]),                           # M2
        ([(0,0)], [1], [(1,0), (1,1)], [1, 1], [(0,1)], [1]),                           # M3
        ([(1,1)], [1], [(0,1), (1,1)], [1, -1], [(1,1)], [1]),                          # M4
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,1)], [1]),                           # M5
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,0)], [1]),                # M6
        ([(0,1), (1,1)], [1, -1], [(1,0), (0,0)], [1, -1], [(0,0)], [1]),               # M7
    ]
    
    for a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs in outer_terms:
        for a_bi, a_bc in a_blocks:
            a_coeff = a_coeffs[a_blocks.index((a_bi, a_bc))]
            for b_bi, b_bc in b_blocks:
                b_coeff = b_coeffs[b_blocks.index((b_bi, b_bc))]
                if a_coeff == 0 or b_coeff == 0:
                    continue
                # A block (a_bi, a_bc): rows a_bi*2..a_bi*2+1, cols a_bc*2..a_bc*2+1
                # B block (b_bi, b_bc): rows b_bi*2..b_bi*2+1, cols b_bc*2..b_bc*2+1
                # Note: for matrix mult, A's col must match B's row.
                # A block col = a_bc, B block row = b_bi. They must be equal!
                # But in Strassen, the pairing is implicit. Let me check:
                # M1: (A00+A11)(B00+B11). A00 is block (0,0), col 0. B00 is block (0,0), row 0. Match.
                # A11 is block (1,1), col 1. B11 is block (1,1), row 1. Match.
                # So we need a_bc == b_bi. Let me filter.
                if a_bc != b_bi:
                    continue
                
                for ii in range(7):
                    # Inner Strassen term ii for the 2x2x2 product
                    U_inner_val = U2[ii]  # 4-vector for 2x2 A block
                    V_inner_val = V2[ii]  # 4-vector for 2x2 B block
                    W_inner_val = W2[ii]  # 4-vector for 2x2 C block
                    
                    U_term = np.zeros(16)
                    for ir in range(2):
                        for ic in range(2):
                            idx2 = ir * 2 + ic
                            val = U_inner_val[idx2]
                            if val != 0:
                                r4 = a_bi * 2 + ir
                                c4 = a_bc * 2 + ic
                                U_term[r4 * 4 + c4] = a_coeff * b_coeff * val
                    
                    V_term = np.zeros(16)
                    for ir in range(2):
                        for ic in range(2):
                            idx2 = ir * 2 + ic
                            val = V_inner_val[idx2]
                            if val != 0:
                                r4 = b_bi * 2 + ir
                                c4 = b_bc * 2 + ic
                                V_term[r4 * 4 + c4] = val  # b_coeff already applied in U? No, separate
                    
                    # Actually, the product is (a_coeff * A_block) * (b_coeff * B_block)
                    # = a_coeff * b_coeff * (A_block * B_block)
                    # So we can put a_coeff in U and b_coeff in V, or split them
                    # Let me put a_coeff * b_coeff in U and 1 in V, or distribute
                    # For cleanliness: U has a_coeff * val, V has b_coeff * val
                    U_term = np.zeros(16)
                    for ir in range(2):
                        for ic in range(2):
                            idx2 = ir * 2 + ic
                            val = U_inner_val[idx2]
                            if val != 0:
                                r4 = a_bi * 2 + ir
                                c4 = a_bc * 2 + ic
                                U_term[r4 * 4 + c4] = a_coeff * val
                    
                    V_term = np.zeros(16)
                    for ir in range(2):
                        for ic in range(2):
                            idx2 = ir * 2 + ic
                            val = V_inner_val[idx2]
                            if val != 0:
                                r4 = b_bi * 2 + ir
                                c4 = b_bc * 2 + ic
                                V_term[r4 * 4 + c4] = b_coeff * val
                    
                    W_term = np.zeros(16)
                    # C blocks from c_blocks
                    for ci, (c_bi, c_bc) in enumerate(c_blocks):
                        c_coeff = c_coeffs[ci]
                        # W_inner_val for 2x2 C: index (k, i) where i is C row, k is C col
                        # In 4x4: row = c_bi*2 + ir, col = c_bc*2 + ic
                        # W[k*n+i] with n=4: index = k*4 + i
                        for ir in range(2):
                            for ic in range(2):
                                idx2 = ic * 2 + ir  # k=ic, i=ir for 2x2
                                val = W_inner_val[idx2]
                                if val != 0:
                                    r4 = c_bc * 2 + ic  # k in 4x4
                                    c4 = c_bi * 2 + ir  # i in 4x4
                                    W_term[r4 * 4 + c4] = c_coeff * val
                    
                    U444.append(U_term)
                    V444.append(V_term)
                    W444.append(W_term)
    
    # Now build 4x5x5 from 4x4x4 + cross terms
    # A is 4x5, B is 5x5, C is 4x5
    # Split: A = [A_left (4x4), A_right (4x1)], B = [[B_topleft (4x4), B_topright (4x1)], [B_bottomleft (1x4), B_bottomright (1x1)]]
    # C = A*B = A_left @ B_topleft + A_right @ B_bottomleft + A_left @ B_topright... 
    # Wait: C[i,k] = sum_j A[i,j]*B[j,k]
    # j ranges 0..4. Split j into 0..3 (left) and 4 (right)
    # C[i,k] = sum_{j=0..3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    # = (A_left @ B_left)[i,k] + A[i,4]*B[4,k]
    # where B_left is B[0..3, :] (4x5)
    # So: C = A_left @ B_left + A_right_col * B_bottom_row
    # A_left is 4x4, B_left is 4x5
    # A_left @ B_left: 4x4 @ 4x5
    # Split B_left = [B_LL (4x4), B_LR (4x1)]
    # A_left @ B_left = A_left @ B_LL + A_left @ B_LR
    # A_left @ B_LL: 4x4x4 -> 49 terms
    # A_left @ B_LR: 4x4 @ 4x1 -> schoolbook 16 terms
    # A_right_col * B_bottom_row: 4x1 * 1x5 -> outer product, 4 terms
    
    # Total: 49 + 16 + 4 = 69 terms!
    
    # Let me verify this decomposition:
    # C[i,k] = sum_{j=0..3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    # = sum_{j=0..3} A[i,j] * (B[j,k]) + A[i,4]*B[4,k]
    # For the 4x4x4 part: covers j=0..3, k=0..3
    # For the 4x4x1 part: covers j=0..3, k=4
    # For the 4x1x1 part: covers j=4, k=0..4... wait, k ranges 0..4 for 4x1x5?
    # A_right is 4x1 (column j=4), B_bottom is 1x5 (row j=4)
    # Outer product: (4x1) * (1x5) = 4x5, rank 4
    # So total: 49 + 16 + 4 = 69
    
    # Build the full 4x5x5 decomposition
    U_full = []
    V_full = []
    W_full = []
    
    # Part 1: 4x4x4 (j=0..3, k=0..3)
    # U index: i*5 + j, V index: j*5 + k, W index: k*4 + i
    # A is 4x5: i in 0..3, j in 0..4
    # B is 5x5: j in 0..4, k in 0..4
    # C is 4x5: i in 0..3, k in 0..4
    
    for u444, v444, w444 in zip(U444, V444, W444):
        U_term = np.zeros(20)  # 4*5
        V_term = np.zeros(25)  # 5*5
        W_term = np.zeros(20)  # 5*4
        
        # u444 is 16-vector: index r4*4+c4 where r4 in 0..3, c4 in 0..3
        for idx in range(16):
            val = u444[idx]
            if val != 0:
                r4 = idx // 4
                c4 = idx % 4
                U_term[r4 * 5 + c4] = val  # i=r4, j=c4
        
        for idx in range(16):
            val = v444[idx]
            if val != 0:
                r4 = idx // 4
                c4 = idx % 4
                V_term[r4 * 5 + c4] = val  # j=r4, k=c4
        
        for idx in range(16):
            val = w444[idx]
            if val != 0:
                r4 = idx // 4  # this is k
                c4 = idx % 4  # this is i
                W_term[r4 * 4 + c4] = val  # k=r4, i=c4
        
        U_full.append(U_term)
        V_full.append(V_term)
        W_full.append(W_term)
    
    # Part 2: 4x4x1 (j=0..3, k=4)
    for i in range(4):
        for j in range(4):
            U_term = np.zeros(20)
            V_term = np.zeros(25)
            W_term = np.zeros(20)
            U_term[i * 5 + j] = 1
            V_term[j * 5 + 4] = 1
            W_term[4 * 4 + i] = 1  # k=4, i=i
            U_full.append(U_term)
            V_full.append(V_term)
            W_full.append(W_term)
    
    # Part 3: 4x1x5 (j=4, k=0..4)
    for i in range(4):
        for k in range(5):
            U_term = np.zeros(20)
            V_term = np.zeros(25)
            W_term = np.zeros(20)
            U_term[i * 5 + 4] = 1
            V_term[4 * 5 + k] = 1
            W_term[k * 4 + i] = 1
            U_full.append(U_term)
            V_full.append(V_term)
            W_full.append(W_term)
    
    rank = len(U_full)
    U_arr = np.array(U_full)
    V_arr = np.array(V_full)
    W_arr = np.array(W_full)
    
    if _verify(U_arr, V_arr, W_arr, n, m, p):
        best_U, best_V, best_W = U_arr, V_arr, W_arr
        best_rank = rank
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
