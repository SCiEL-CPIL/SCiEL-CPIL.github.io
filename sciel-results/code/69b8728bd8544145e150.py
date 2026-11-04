"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition search."""
import numpy as np
import itertools
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (n*p, rank)
    # W[k*n + i, r] corresponds to output C[i, k]
    
    # Block-Winograd approach:
    # Partition: i in {0,1} (2 rows) and i=2 (1 row)
    # j in {0,1} and {2,3} (two 2-col blocks)
    # k in {0,1}, {2,3}, {4,5}, {6,7} (four 2-col blocks)
    #
    # For 2x2x2 blocks (i∈{0,1}, j-block, k-block): use rank-7 Winograd
    # For 1x2x2 blocks (i=2, j-block, k-block): use rank-4 schoolbook
    #
    # Total: 8 blocks of 2x2x2 (rank 7 each = 56) + 8 blocks of 1x2x2 (rank 4 each = 32) = 88
    
    # Winograd 2x2x2 rank-7 decomposition:
    # For A(2x2) * B(2x2) = C(2x2):
    # C[0,0] = m1 + m3 - m6
    # C[0,1] = m5 + m7 - m3  -- let me derive this properly
    #
    # Standard rank-7 decomposition for 2x2x2 matmul tensor:
    # The 7 rank-1 tensors are defined by (u, v, w) triples where:
    #
    # Let me use the well-known decomposition. For the 2x2x2 matmul tensor,
    # one rank-7 decomposition has these (U, V, W) entries for the 7 terms:
    #
    # Term 0: U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1]  (but this needs verification)
    #
    # Actually, let me construct it from the bilinear algorithm:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00+b10)
    # m3 = a00(b00-b10)
    # m4 = a11(b11-b01)
    # m5 = (a00+a01)(b01+b11)
    # m6 = a10(b10-b00)
    # m7 = (a00+a01)b10  -- not sure
    #
    # Let me just use a verified decomposition. The 2x2x2 matmul tensor rank is 7.
    # I'll use the following known decomposition:
    #
    # For indices: A has indices (i,j) with i,j in {0,1}, so A index = i*2+j
    # B has indices (j,k) with j,k in {0,1}, so B index = j*2+k
    # C has indices (i,k) with i,k in {0,1}, so C index = i*2+k (but we use k*n+i for global)
    #
    # Rank-7 decomposition (one standard form):
    # r=0: U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1]
    # r=1: U=[0,0,1,1], V=[1,1,0,0], W=[-1,1,1,-1]
    # r=2: U=[1,0,0,0], V=[1,0,-1,0], W=[1,0,0,0]
    # r=3: U=[0,0,0,1], V=[0,0,0,1], W=[0,0,0,1]
    # r=4: U=[0,1,0,0], V=[0,1,0,0], W=[0,1,0,0]
    # r=5: U=[1,0,0,0], V=[0,0,0,-1], W=[0,1,0,0]
    # r=6: U=[0,1,0,0], V=[0,0,1,0], W=[0,0,1,0]
    #
    # Let me verify this. Actually, I'll just test it numerically in the code.
    
    # Let me use a different approach: I'll hardcode a verified rank-7 decomposition
    # for 2x2x2 and verify it in code.
    
    # Known rank-7 decomposition for 2x2x2 matmul (verified):
    # Using the formulation from Strassen's work:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00+b10)  
    # m3 = a00(b00-b10)
    # m4 = a11(b11-b01)
    # m5 = (a00+a01)(b01+b11)
    # m6 = a10(b10-b00)
    # m7 = (a00+a01)b10
    #
    # C00 = m1 + m3 - m6
    # C01 = m5 + m7 - m3  -- need to verify
    # C10 = m2 - m3 + m6  -- need to verify  
    # C11 = m1 - m2 + m4
    #
    # Let me just use a numerically verified decomposition.
    
    # I'll construct the U, V, W matrices for the 2x2x2 block and verify.
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    # Let me use this verified decomposition:
    winograd_2x2x2_U = [
        [1, 0, 1, 0, 1, 0, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, 0],  # a11
    ]
    winograd_2x2x2_V = [
        [1, 1, 1, 1, 1, 0, 0],  # b00
        [0, 0, 0, -1, 1, 0, 0], # b01
        [0, 1, -1, 0, 0, 1, 0], # b10
        [1, 0, 0, 1, 0, 0, 1],  # b11
    ]
    winograd_2x2x2_W = [
        [1, -1, 1, 0, 0, 0, 0],  # c00
        [0, 1, 0, 0, 1, 1, 0],   # c01
        [0, -1, 0, 0, 0, 1, 0],  # c10
        [1, 0, 0, 1, 0, 0, 1],   # c11
    ]
    
    # Verify this decomposition
    def verify_winograd():
        import numpy as np
        U = np.array(winograd_2x2x2_U, dtype=float)
        V = np.array(winograd_2x2x2_V, dtype=float)
        W = np.array(winograd_2x2x2_W, dtype=float)
        # Check: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all i,j,k
        # U has shape (4, 7) where index = i*2+j
        # V has shape (4, 7) where index = j*2+k
        # W has shape (4, 7) where index = i*2+k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = i * 2 + j
                    v_idx = j * 2 + k
                    w_idx = i * 2 + k
                    val = sum(U[u_idx, r] * V[v_idx, r] * W[w_idx, r] for r in range(7))
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # If verification fails, fall back to schoolbook for 2x2x2 (rank 8)
    use_winograd = verify_winograd()
    
    if not use_winograd:
        # Use schoolbook for 2x2x2 blocks
        winograd_2x2x2_U = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]]
        winograd_2x2x2_V = [[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0]]
        winograd_2x2x2_W = [[1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0]]
        # Actually just use rank 8 schoolbook
        winograd_2x2x2_U = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]]
        winograd_2x2x2_V = [[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0]]
        winograd_2x2x2_W = [[1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0]]
        # This is getting complicated. Let me just use schoolbook for all blocks.
        use_winograd = False
    
    # Build the full decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    
    # i-blocks: [(0, [0,1]), (1, [2])]  -- block_id, list of i indices
    i_blocks = [(0, [0, 1]), (1, [2])]
    # j-blocks: [(0, [0,1]), (1, [2,3])]
    j_blocks = [(0, [0, 1]), (1, [2, 3])]
    # k-blocks: [(0, [0,1]), (1, [2,3]), (2, [4,5]), (3, [6,7])]
    k_blocks = [(0, [0, 1]), (1, [2, 3]), (2, [4, 5]), (3, [6, 7])]
    
    for ib, i_list in i_blocks:
        for jb, j_list in j_blocks:
            for kb, k_list in k_blocks:
                if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    # 2x2x2 block: use Winograd if available, else schoolbook
                    if use_winograd:
                        # Add 7 terms
                        for r in range(7):
                            u_col = [0.0] * (n * m)
                            v_col = [0.0] * (m * p)
                            w_col = [0.0] * (n * p)
                            for ii, i_val in enumerate(i_list):
                                for jj, j_val in enumerate(j_list):
                                    a_idx = i_val * m + j_val
                                    u_col[a_idx] = winograd_2x2x2_U[ii * 2 + jj][r]
                            for jj, j_val in enumerate(j_list):
                                for kk, k_val in enumerate(k_list):
                                    b_idx = j_val * p + k_val
                                    v_col[b_idx] = winograd_2x2x2_V[jj * 2 + kk][r]
                            for ii, i_val in enumerate(i_list):
                                for kk, k_val in enumerate(k_list):
                                    c_idx = k_val * n + i_val
                                    w_col[c_idx] = winograd_2x2x2_W[ii * 2 + kk][r]
                            U_cols.append(u_col)
                            V_cols.append(v_col)
                            W_cols.append(w_col)
                    else:
                        # Schoolbook: 8 terms
                        for ii, i_val in enumerate(i_list):
                            for jj, j_val in enumerate(j_list):
                                for kk, k_val in enumerate(k_list):
                                    u_col = [0.0] * (n * m)
                                    v_col = [0.0] * (m * p)
                                    w_col = [0.0] * (n * p)
                                    u_col[i_val * m + j_val] = 1.0
                                    v_col[j_val * p + k_val] = 1.0
                                    w_col[k_val * n + i_val] = 1.0
                                    U_cols.append(u_col)
                                    V_cols.append(v_col)
                                    W_cols.append(w_col)
                else:
                    # 1x2x2 block (i_list has 1 element): schoolbook, 4 terms
                    for jj, j_val in enumerate(j_list):
                        for kk, k_val in enumerate(k_list):
                            i_val = i_list[0]
                            u_col = [0.0] * (n * m)
                            v_col = [0.0] * (m * p)
                            w_col = [0.0] * (n * p)
                            u_col[i_val * m + j_val] = 1.0
                            v_col[j_val * p + k_val] = 1.0
                            w_col[k_val * n + i_val] = 1.0
                            U_cols.append(u_col)
                            V_cols.append(v_col)
                            W_cols.append(w_col)
    
    rank = len(U_cols)
    U = [col[:] for col in zip(*U_cols)] if U_cols else []
    V = [col[:] for col in zip(*V_cols)] if V_cols else []
    W = [col[:] for col in zip(*W_cols)] if W_cols else []
    
    # Convert to integer/half-integer if possible
    def to_half_int(val):
        # Check if val is integer or half-integer
        if abs(val - round(val)) < 1e-10:
            return int(round(val))
        elif abs(val * 2 - round(val * 2)) < 1e-10:
            return round(val * 2) / 2
        else:
            return val
    
    U = [[to_half_int(v) for v in row] for row in U]
    V = [[to_half_int(v) for v in row] for row in V]
    W = [[to_half_int(v) for v in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
