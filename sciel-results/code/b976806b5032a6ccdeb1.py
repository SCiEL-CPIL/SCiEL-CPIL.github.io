"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen decomposition for 2x2 blocks
    # For a 2x2 block multiplication C = A*B:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We'll use Strassen to decompose the 4x4 x 4x8 multiplication.
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B (4x8) into 2x4 blocks: B11, B12, B21, B22 (each 2x4)
    # But Strassen works on square blocks. So we need to handle 2x4 blocks.
    #
    # Alternative: Split B (4x8) into two 4x4 matrices: B_left (cols 0-3), B_right (cols 4-7)
    # Then do two separate 4x4 x 4x4 multiplications.
    # Each 4x4 x 4x4 can use Strassen on 2x2 blocks, giving rank 7 per 2x2, 
    # but for 4x4 we have 4 2x2 blocks, so rank = 4*7 = 28 per 4x4x4.
    # Total rank = 28 + 28 = 56.
    #
    # But we can do better by sharing A-side forms.
    #
    # Actually, let's think more carefully.
    #
    # A is 4x4, B is 4x8.
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B into 2x4 blocks: B = [[B1, B2], [B3, B4]] where B1,B2,B3,B4 are 2x4.
    #
    # C = A*B is 4x8, split into 2x4 blocks: C = [[C11, C12], [C21, C22]]
    # C11 = A11*B1 + A12*B3
    # C12 = A11*B2 + A12*B4
    # C21 = A21*B1 + A22*B3
    # C22 = A21*B2 + A22*B4
    #
    # For each 2x2 x 2x4 multiplication, we can use a variant of Strassen.
    # But 2x2 x 2x4 is not square, so standard Strassen doesn't apply directly.
    #
    # Let's use the fact that B is 4x8 = two 4x4 blocks.
    # B = [B_left | B_right] where B_left and B_right are both 4x4.
    # Then C = [A*B_left | A*B_right].
    #
    # For each 4x4 x 4x4 multiplication, use Strassen on 2x2 blocks.
    # A 4x4 x 4x4 using Strassen has rank 7 (per 2x2) * 4 = 28? No.
    # Actually, for 4x4 x 4x4, we split into 2x2 blocks (4 blocks), 
    # and use Strassen's 7 multiplications per 2x2 x 2x2, but the 2x2 blocks 
    # are not independent.
    #
    # Standard result: rank(4x4 x 4x4) <= 49 using fast matrix multiplication,
    # but with Strassen on 2x2 blocks, we get rank 28 (since each of the 4 
    # 2x2 blocks requires 7 multiplications, but they share linear forms).
    #
    # Actually, for 4x4 x 4x4, if we split into 2x2 blocks and apply Strassen 
    # to each pair of 2x2 blocks, we need 7 multiplications per "level", but 
    # the total rank is 7*4 = 28? No, it's more subtle.
    #
    # Let me just use a known low-rank decomposition.
    #
    # For 4x4 x 4x4, the schoolbook rank is 64.
    # With Strassen on 2x2 blocks, we can get rank 28.
    #
    # So for 4x4 x 4x8, we can get rank 28 + 28 = 56.
    #
    # Let me implement this.
    
    # First, let's build the Strassen decomposition for 2x2 x 2x2.
    # Then embed it into 4x4 x 4x4.
    # Then duplicate for 4x4 x 4x8.
    
    def strassen_2x2_decomposition():
        """Return U, V, W for 2x2 x 2x2 Strassen decomposition.
        U: (4, 7), V: (4, 7), W: (4, 7)
        U index: i*2+j, V index: j*2+k, W index: k*2+i (for 2x2)
        """
        rank = 7
        U = [[0.0]*rank for _ in range(4)]
        V = [[0.0]*rank for _ in range(4)]
        W = [[0.0]*rank for _ in range(4)]
        
        # P1 = (A11+A22)(B11+B22)
        # U: A11, A22 -> 1
        # V: B11, B22 -> 1
        # W: C11 (+1), C22 (+1)
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        W[0][0] = 1  # C11
        W[3][0] = 1  # C22
        
        # P2 = (A21+A22)B11
        # U: A21, A22 -> 1
        # V: B11 -> 1
        # W: C21 (+1), C22 (-1)
        U[2][1] = 1  # A21
        U[3][1] = 1  # A22
        V[0][1] = 1  # B11
        W[2][1] = 1  # C21
        W[3][1] = -1 # C22
        
        # P3 = A11(B21-B22)
        # U: A11 -> 1
        # V: B21 -> 1, B22 -> -1
        # W: C12 (+1), C22 (+1)
        U[0][2] = 1  # A11
        V[2][2] = 1  # B21
        V[3][2] = -1 # B22
        W[1][2] = 1  # C12
        W[3][2] = 1  # C22
        
        # P4 = A22(B12-B11)
        # U: A22 -> 1
        # V: B12 -> 1, B11 -> -1
        # W: C11 (+1), C21 (+1)
        U[3][3] = 1  # A22
        V[1][3] = 1  # B12
        V[0][3] = -1 # B11
        W[0][3] = 1  # C11
        W[2][3] = 1  # C21
        
        # P5 = (A11+A12)B22
        # U: A11, A12 -> 1
        # V: B22 -> 1
        # W: C11 (-1), C12 (+1)
        U[0][4] = 1  # A11
        U[1][4] = 1  # A12
        V[3][4] = 1  # B22
        W[0][4] = -1 # C11
        W[1][4] = 1  # C12
        
        # P6 = (A21-A11)(B11+B12)
        # U: A21 -> 1, A11 -> -1
        # V: B11 -> 1, B12 -> 1
        # W: C22 (+1)
        U[2][5] = 1  # A21
        U[0][5] = -1 # A11
        V[0][5] = 1  # B11
        V[1][5] = 1  # B12
        W[3][5] = 1  # C22
        
        # P7 = (A12-A22)(B21+B22)
        # U: A12 -> 1, A22 -> -1
        # V: B21 -> 1, B22 -> 1
        # W: C11 (+1)
        U[1][6] = 1  # A12
        U[3][6] = -1 # A22
        V[2][6] = 1  # B21
        V[3][6] = 1  # B22
        W[0][6] = 1  # C11
        
        return U, V, W
    
    def embed_2x2_in_4x4(U2, V2, W2, block_i, block_j, block_k):
        """Embed a 2x2 Strassen decomposition into a 4x4 x 4x4 decomposition.
        block_i, block_j, block_k specify which 2x2 block of A, B, C.
        Returns contributions to U (4*4=16 rows), V (4*4=16 rows), W (4*4=16 rows).
        """
        # U index: i*4+j, where i is global row, j is global col of A
        # For 2x2 block at (block_i, block_j) in A:
        # i ranges from 2*block_i to 2*block_i+1
        # j ranges from 2*block_j to 2*block_j+1
        
        U_contrib = {}
        V_contrib = {}
        W_contrib = {}
        
        for u_idx in range(4):  # 2x2 block indices
            a_i = u_idx // 2  # 0 or 1
            a_j = u_idx % 2   # 0 or 1
            global_i = 2*block_i + a_i
            global_j = 2*block_j + a_j
            u_global = global_i * 4 + global_j
            
            for col in range(7):
                if U2[u_idx][col] != 0:
                    U_contrib[(u_global, col)] = U2[u_idx][col]
        
        for v_idx in range(4):  # 2x2 block indices
            b_j = v_idx // 2  # 0 or 1 (row of B)
            b_k = v_idx % 2   # 0 or 1 (col of B)
            global_j = 2*block_j + b_j
            global_k = 2*block_k + b_k
            v_global = global_j * 4 + global_k  # B is 4x4, index j*4+k
            
            for col in range(7):
                if V2[v_idx][col] != 0:
                    V_contrib[(v_global, col)] = V2[v_idx][col]
        
        for w_idx in range(4):  # 2x2 block indices
            c_k = w_idx // 2  # 0 or 1 (row of C)
            c_i = w_idx % 2   # 0 or 1 (col of C)
            global_k = 2*block_k + c_k
            global_i = 2*block_i + c_i
            w_global = global_k * 4 + global_i  # C is 4x4, index k*4+i
            
            for col in range(7):
                if W2[w_idx][col] != 0:
                    W_contrib[(w_global, col)] = W2[w_idx][col]
        
        return U_contrib, V_contrib, W_contrib
    
    # Build 4x4 x 4x4 decomposition using Strassen on 2x2 blocks
    U2, V2, W2 = strassen_2x2_decomposition()
    
    U4 = {}
    V4 = {}
    W4 = {}
    
    for block_i in range(2):
        for block_j in range(2):
            for block_k in range(2):
                uc, vc, wc = embed_2x2_in_4x4(U2, V2, W2, block_i, block_j, block_k)
                for key, val in uc.items():
                    U4[key] = U4.get(key, 0) + val
                for key, val in vc.items():
                    V4[key] = V4.get(key, 0) + val
                for key, val in wc.items():
                    W4[key] = W4.get(key, 0) + val
    
    # Now build 4x4 x 4x8 decomposition.
    # B is 4x8, split into B_left (cols 0-3) and B_right (cols 4-7).
    # C is 4x8, split into C_left (cols 0-3) and C_right (cols 4-7).
    
    # For B_left (cols 0-3): block_k = 0
    # For B_right (cols 4-7): block_k = 1 (but shifted by 4 in global index)
    
    U_full = {}
    V_full = {}
    W_full = {}
    
    # Left half: B_left is 4x4 with cols 0-3
    for (u_global, col), val in U4.items():
        U_full[(u_global, col)] = U_full.get((u_global, col), 0) + val
    for (v_global, col), val in V4.items():
        # v_global = j*4 + k, where k is 0-3 for left half
        V_full[(v_global, col)] = V_full.get((v_global, col), 0) + val
    for (w_global, col), val in W4.items():
        # w_global = k*4 + i, where k is 0-3 for left half
        W_full[(w_global, col)] = W_full.get((w_global, col), 0) + val
    
    # Right half: B_right is 4x4 with cols 4-7
    # Shift B cols by 4, shift C cols by 4
    for (u_global, col), val in U4.items():
        # A is the same, so U is the same
        U_full[(u_global, col + 28)] = U_full.get((u_global, col + 28), 0) + val
    
    for (v_global, col), val in V4.items():
        # v_global = j*4 + k, where k is 0-3 in local coords
        # Global k = k + 4
        j = v_global // 4
        k = v_global % 4
        v_global_right = j * 8 + (k + 4)
        V_full[(v_global_right, col + 28)] = V_full.get((v_global_right, col + 28), 0) + val
    
    for (w_global, col), val in W4.items():
        # w_global = k*4 + i, where k is 0-3 in local coords
        # Global k = k + 4
        k = w_global // 4
        i = w_global % 4
        w_global_right = (k + 4) * 4 + i
        W_full[(w_global_right, col + 28)] = W_full.get((w_global_right, col + 28), 0) + val
    
    rank = 56
    U_mat = [[0.0]*rank for _ in range(n*m)]  # 16 x 56
    V_mat = [[0.0]*rank for _ in range(m*p)]  # 32 x 56
    W_mat = [[0.0]*rank for _ in range(p*n)]  # 32 x 56
    
    for (u_global, col), val in U_full.items():
        U_mat[u_global][col] = val
    for (v_global, col), val in V_full.items():
        V_mat[v_global][col] = val
    for (w_global, col), val in W_full.items():
        W_mat[w_global][col] = val
    
    if time.time() > deadline - 5:
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
