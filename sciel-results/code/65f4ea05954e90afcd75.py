import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 matrices
    # U2[l] = coeffs for X (left matrix)
    # V2[l] = coeffs for Y (right matrix)
    # W2[l] = coeffs for Z (product matrix)
    # Indices for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U2 = [
        [1, 0, 0, 1], # S1: x00 + x11
        [0, 0, 1, 1], # S2: x10 + x11
        [1, 0, 0, 0], # S3: x00
        [0, 0, 0, 1], # S4: x11
        [1, 1, 0, 0], # S5: x00 + x01
        [0, 0, 1, -1],# S6: x10 - x00
        [0, 1, 0, -1] # S7: x01 - x11
    ]
    
    V2 = [
        [1, 0, 0, 1], # S1: y00 + y11
        [1, 0, 0, 0], # S2: y00
        [0, 1, 0, -1],# S3: y01 - y11
        [0, 0, 1, -1],# S4: y10 - y00
        [0, 0, 0, 1], # S5: y11
        [1, 1, 0, 0], # S6: y00 + y01
        [0, 0, 1, 1]  # S7: y10 + y11
    ]
    
    W2 = [
        [1, 0, 0, 1], # S1: z00 += 1, z11 += 1
        [0, 0, 1, 0], # S2: z10 += 1
        [0, 1, 0, 0], # S3: z01 += 1
        [1, 0, 1, 0], # S4: z00 += 1, z10 += 1
        [-1, 1, 0, 0],# S5: z00 -= 1, z01 += 1
        [0, 0, 0, 1], # S6: z11 += 1
        [1, 0, 0, 0]  # S7: z00 += 1
    ]
    
    # Gamma: Coefficients for top-level blocks
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    # Indices: r (0,1), c (0,1)
    # Gamma[rc_idx, K]
    Gamma = np.zeros((4, 7), dtype=int)
    # r=0, c=0 (idx 0)
    Gamma[0, 0] = 1
    Gamma[0, 3] = 1
    Gamma[0, 4] = -1
    Gamma[0, 6] = 1
    # r=0, c=1 (idx 1)
    Gamma[1, 2] = 1
    Gamma[1, 4] = 1
    # r=1, c=0 (idx 2)
    Gamma[2, 1] = 1
    Gamma[2, 3] = 1
    # r=1, c=1 (idx 3)
    Gamma[3, 0] = 1
    Gamma[3, 1] = -1
    Gamma[3, 2] = 1
    Gamma[3, 5] = 1
    
    # Define LA and LB for each K
    # LA[K] is 4x16, maps A to X_K (2x2 matrix flattened)
    # LB[K] is 4x16, maps B to Y_K (2x2 matrix flattened)
    
    # Block indices in flat 4x4 (row-major)
    # A00: 0, 1, 4, 5
    # A01: 2, 3, 6, 7
    # A10: 8, 9, 12, 13
    # A11: 10, 11, 14, 15
    
    def get_block_indices(block_r, block_c):
        rows = [2*block_r, 2*block_r + 1]
        cols = [2*block_c, 2*block_c + 1]
        idxs = []
        for r in rows:
            for c in cols:
                idxs.append(r*4 + c)
        return idxs
    
    B00 = get_block_indices(0, 0)
    B01 = get_block_indices(0, 1)
    B10 = get_block_indices(1, 0)
    B11 = get_block_indices(1, 1)
    
    # Top-level Strassen definitions
    # K=0: X=A00+A11, Y=B00+B11
    # K=1: X=A10+A11, Y=B00
    # K=2: X=A00,     Y=B01-B11
    # K=3: X=A11,     Y=B10-B00
    # K=4: X=A00+A01, Y=B11
    # K=5: X=A10-A00, Y=B00+B01
    # K=6: X=A01-A11, Y=B10+B11
    
    # We need to construct LA[K] such that X_K = LA[K] @ vec(A)
    # X_K is 2x2, flattened to 4.
    # X_K[0] = sum of top-left entries of constituent blocks
    # X_K[1] = sum of top-right entries
    # X_K[2] = sum of bottom-left entries
    # X_K[3] = sum of bottom-right entries
    
    LA = []
    LB = []
    
    def build_mapping(blocks, signs):
        # blocks: list of (block_idx, sign)
        # block_idx: 0=B00, 1=B01, 2=B10, 3=B11
        mat = np.zeros((4, 16), dtype=int)
        block_indices = [B00, B01, B10, B11]
        # Entries in 2x2 block: 0=TL, 1=TR, 2=BL, 3=BR
        for b_idx, sign in blocks:
            idxs = block_indices[b_idx]
            # idxs[0]=TL, idxs[1]=TR, idxs[2]=BL, idxs[3]=BR
            mat[0, idxs[0]] = sign
            mat[1, idxs[1]] = sign
            mat[2, idxs[2]] = sign
            mat[3, idxs[3]] = sign
        return mat

    # K=0: A00(1) + A11(1)
    LA.append(build_mapping([(0, 1), (3, 1)]))
    LB.append(build_mapping([(0, 1), (3, 1)]))
    
    # K=1: A10(1) + A11(1); B00(1)
    LA.append(build_mapping([(2, 1), (3, 1)]))
    LB.append(build_mapping([(0, 1)]))
    
    # K=2: A00(1); B01(1) - B11(1)
    LA.append(build_mapping([(0, 1)]))
    LB.append(build_mapping([(1, 1), (3, -1)]))
    
    # K=3: A11(1); B10(1) - B00(1)
    LA.append(build_mapping([(3, 1)]))
    LB.append(build_mapping([(2, 1), (0, -1)]))
    
    # K=4: A00(1) + A01(1); B11(1)
    LA.append(build_mapping([(0, 1), (1, 1)]))
    LB.append(build_mapping([(3, 1)]))
    
    # K=5: A10(1) - A00(1); B00(1) + B01(1)
    LA.append(build_mapping([(2, 1), (0, -1)]))
    LB.append(build_mapping([(0, 1), (1, 1)]))
    
    # K=6: A01(1) - A11(1); B10(1) + B11(1)
    LA.append(build_mapping([(1, 1), (3, -1)]))
    LB.append(build_mapping([(2, 1), (3, 1)]))
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for K in range(7):
        for L in range(7):
            col_idx = K * 7 + L
            
            # U[:, col] = LA[K].T @ U2[L]
            U[:, col_idx] = np.dot(LA[K].T, U2[L])
            # V[:, col] = LB[K].T @ V2[L]
            V[:, col_idx] = np.dot(LB[K].T, V2[L])
            
            # W rows
            for w_idx in range(16):
                # w_idx corresponds to C[i, k]
                # k = w_idx // 4, i = w_idx % 4
                k_out = w_idx // 4
                i_out = w_idx % 4
                
                # Block indices
                r = i_out // 2
                c = k_out // 2
                rc_idx = r * 2 + c
                
                # Local indices
                u_loc = i_out % 2
                v_loc = k_out % 2
                w2_idx = u_loc * 2 + v_loc
                
                coeff = Gamma[rc_idx, K] * W2[L][w2_idx]
                W[w_idx, col_idx] = coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
