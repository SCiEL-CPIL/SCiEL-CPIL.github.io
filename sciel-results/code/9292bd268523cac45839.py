"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _build_strassen_2x2():
    """Return U, V, W factors for 2x2x2 Strassen (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _build_4x4_strassen():
    """Construct 4x4x4 factors using 4 instances of 2x2 Strassen."""
    # 4x4 matrix indices: 0..15
    # Block 00: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 01: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 2:4]
    # Block 10: A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    # Block 11: A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    
    U_4 = [[0]*28 for _ in range(16)]
    V_4 = [[0]*28 for _ in range(16)]
    W_4 = [[0]*28 for _ in range(16)]
    
    U_2, V_2, W_2 = _build_strassen_2x2()
    
    # Mapping for 2x2 blocks in 4x4
    # Block (br, bc) where br, bc in {0, 1}
    # A indices: (br*2 + r, bc*2 + c)
    # B indices: (br*2 + r, bc*2 + c) -> Wait, B is row-major.
    # B block (br, bc) is B[br*2 + r, bc*2 + c]
    # C block (br, bc) is C[br*2 + r, bc*2 + c]
    
    # We need to map the 2x2 Strassen indices to 4x4 indices.
    # Strassen U index: r*2 + c for A.
    # In 4x4, A index is (br*2 + r) * 4 + (bc*2 + c).
    
    blocks = [
        (0, 0), # Top-Left
        (0, 1), # Top-Right (A col 2:4, B row 0:2) -> A[0:2, 2:4] * B[0:2, 2:4]? 
        # Wait. C[0:2, 2:4] = A[0:2, 0:2]B[0:2, 2:4] + A[0:2, 2:4]B[2:4, 2:4].
        # My previous decomposition was:
        # C_00 = A_00 B_00 + A_01 B_10
        # C_01 = A_00 B_01 + A_01 B_11
        # C_10 = A_10 B_00 + A_11 B_10
        # C_11 = A_10 B_01 + A_11 B_11
        
        # So we need 4 products:
        # 1. A_00 * B_00
        # 2. A_01 * B_10
        # 3. A_00 * B_01
        # 4. A_01 * B_11
        # 5. A_10 * B_00
        # 6. A_11 * B_10
        # 7. A_10 * B_01
        # 8. A_11 * B_11
        
        # Each product is 2x2 * 2x2.
        # We use Strassen for each.
    ]
    
    # Product 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,4,5. B indices: 0,1,4,5. C indices: 0,1,4,5.
    # Strassen indices: 0,1,2,3.
    # Map:
    # A_strassen[r, c] -> A_4[r*4 + c]
    # B_strassen[r, c] -> B_4[r*4 + c]
    # C_strassen[r, c] -> C_4[r*4 + c]
    
    def add_block(U_src, V_src, W_src, a_offset_r, a_offset_c, b_offset_r, b_offset_c, c_offset_r, c_offset_c, rank_offset):
        for r in range(2):
            for c in range(2):
                u_idx = r*4 + c
                v_idx = r*4 + c
                w_idx = r*4 + c
                
                u_4_idx = (a_offset_r + r) * 4 + (a_offset_c + c)
                v_4_idx = (b_offset_r + r) * 4 + (b_offset_c + c)
                w_4_idx = (c_offset_r + r) * 4 + (c_offset_c + c)
                
                for k in range(7):
                    U_4[u_4_idx][rank_offset + k] += U_src[r][c][k]
                    V_4[v_4_idx][rank_offset + k] += V_src[r][c][k]
                    W_4[w_4_idx][rank_offset + k] += W_src[r][c][k]

    # 1. A_00 * B_00
    add_block(U_2, V_2, W_2, 0, 0, 0, 0, 0, 0, 0)
    
    # 2. A_01 * B_10
    # A_01 is A[0:2, 2:4]. B_10 is B[2:4, 0:2]. C_01 is C[0:2, 2:4].
    # A offset: r=0, c=2. B offset: r=2, c=0. C offset: r=0, c=2.
    add_block(U_2, V_2, W_2, 0, 2, 2, 0, 0, 2, 7)
    
    # 3. A_00 * B_01
    # A_00 is A[0:2, 0:2]. B_01 is B[0:2, 2:4]. C_01 is C[0:2, 2:4].
    # A offset: r=0, c=0. B offset: r=0, c=2. C offset: r=0, c=2.
    add_block(U_2, V_2, W_2, 0, 0, 0, 2, 0, 2, 14)
    
    # 4. A_01 * B_11
    # A_01 is A[0:2, 2:4]. B_11 is B[2:4, 2:4]. C_11 is C[2:4, 2:4].
    # A offset: r=0, c=2. B offset: r=2, c=2. C offset: r=2, c=2.
    add_block(U_2, V_2, W_2, 0, 2, 2, 2, 2, 2, 21)
    
    # Wait, I missed the bottom row of C.
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    # 5. A_10 * B_00
    # A_10 is A[2:4, 0:2]. B_00 is B[0:2, 0:2]. C_10 is C[2:4, 0:2].
    add_block(U_2, V_2, W_2, 2, 0, 0, 0, 2, 0, 28) # This exceeds 28?
    # Rank is 28. Indices 0..27.
    # I used 0-6, 7-13, 14-20, 21-27.
    # I need 8 blocks. 8 * 7 = 56 rank.
    # My previous estimate of 28 was for 4 products.
    # But there are 8 products in the 4x4 block decomposition.
    # So rank is 56 for the 4x4 block?
    
    # Let's re-calculate.
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    # 8 products. Each rank 7. Total 56.
    
    # So U_4, V_4, W_4 should have 56 columns.
    
    # Let's restart the 4x4 construction with rank 56.
    
    return None # Placeholder

def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try 89 rank solution
    # 1. 4x4 block: 8 Strassen instances (rank 56)
    # 2. Cross terms:
    #    A_12 B_21 (16)
    #    A_11 B_12 (16)
    #    A_12 B_22 (4)
    #    A_21 B_11 (16)
    #    A_22 B_21 (4)
    #    A_21 B_12 (4)
    #    A_22 B_22 (1)
    # Total: 56 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 117.
    # Still better than 125.
    
    U_5 = [[0]*117 for _ in range(25)]
    V_5 = [[0]*117 for _ in range(25)]
    W_5 = [[0]*117 for _ in range(25)]
    
    U_2, V_2, W_2 = _build_strassen_2x2()
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c, offset):
        for r in range(2):
            for c in range(2):
                u_idx = (a_r + r) * 5 + (a_c + c)
                v_idx = (b_r + r) * 5 + (b_c + c)
                w_idx = (c_r + r) * 5 + (c_c + c)
                for k in range(7):
                    U_5[u_idx][offset + k] += U_2[r][c][k]
                    V_5[v_idx][offset + k] += V_2[r][c][k]
                    W_5[w_idx][offset + k] += W_2[r][c][k]

    # 4x4 Block (0..3)
    # 8 blocks
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)   # A00 B00 -> C00
    add_strassen_block(0, 2, 2, 0, 0, 0, 7)   # A01 B10 -> C00
    add_strassen_block(0, 0, 0, 2, 0, 2, 14)  # A00 B01 -> C01
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)  # A01 B11 -> C01
    add_strassen_block(2, 0, 0, 0, 2, 0, 28)  # A10 B00 -> C10
    add_strassen_block(2, 2, 2, 0, 2, 0, 35)  # A11 B10 -> C10
    add_strassen_block(2, 0, 0, 2, 2, 2, 42)  # A10 B01 -> C11
    add_strassen_block(2, 2, 2, 2, 2, 2, 49)  # A11 B11 -> C11
    
    offset = 56
    
    # Cross terms
    # A_12 B_21: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # Outer product. 16 terms.
    for r in range(4):
        for c in range(4):
            U_5[r*5 + 4][offset] = 1
            V_5[4*5 + c][offset] = 1
            W_5[r*5 + c][offset] = 1
            offset += 1
            
    # A_11 B_12: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # Matrix vector. 16 terms.
    for r in range(4):
        for c in range(4):
            U_5[r*5 + c][offset] = 1
            V_5[c*5 + 4][offset] = 1
            W_5[r*5 + 4][offset] = 1
            offset += 1
            
    # A_12 B_22: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    # Vector scalar. 4 terms.
    for r in range(4):
        U_5[r*5 + 4][offset] = 1
        V_5[4*5 + 4][offset] = 1
        W_5[r*5 + 4][offset] = 1
        offset += 1
        
    # A_21 B_11: A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    # Scalar vector. 16 terms.
    for c in range(4):
        for r in range(4):
            U_5[4*5 + c][offset] = 1
            V_5[r*5 + c][offset] = 1
            W_5[4*5 + r][offset] = 1
            offset += 1
            
    # A_22 B_21: A[4, 4] * B[4, 0:4] -> C[4, 0:4]
    # Scalar vector. 4 terms.
    for c in range(4):
        U_5[4*5 + 4][offset] = 1
        V_5[4*5 + c][offset] = 1
        W_5[4*5 + c][offset] = 1
        offset += 1
        
    # A_21 B_12: A[4, 0:4] * B[0:4, 4] -> C[4, 4]
    # Dot product. 4 terms.
    for c in range(4):
        U_5[4*5 + c][offset] = 1
        V_5[c*5 + 4][offset] = 1
        W_5[4*5 + 4][offset] = 1
        offset += 1
        
    # A_22 B_22: A[4, 4] * B[4, 4] -> C[4, 4]
    # Scalar. 1 term.
    U_5[4*5 + 4][offset] = 1
    V_5[4*5 + 4][offset] = 1
    W_5[4*5 + 4][offset] = 1
    offset += 1
    
    # Verify
    if _verify(U_5, V_5, W_5, n, m, p, 117):
        best_rank = 117
        best_result = {"U": U_5, "V": V_5, "W": W_5}
    else:
        print("Verification failed for 117 rank solution")

    return best_result
# EVOLVE-BLOCK-END
