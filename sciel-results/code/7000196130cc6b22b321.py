import numpy as np
import time

DIMENSIONS = [4, 4, 9]
_DEADLINE = time.time() + 135

def _build_strassen_2x2():
    """
    Build rank-7 decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U: (i, j) -> i*2 + j
    V: (j, k) -> j*2 + k
    W: (k, i) -> k*2 + i
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A00 + A11) * (B00 + B11)
    # A00 is U[0,0], A11 is U[3,3]
    # B00 is V[0,0], B11 is V[3,3]
    # C00 gets M1, C11 gets M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1 # C00 (k=0, i=0)
    W[3, 0] = 1 # C11 (k=1, i=1)
    
    # M2 = (A10 + A11) * B00
    # A10 is U[2,0], A11 is U[3,3]
    # B00 is V[0,0]
    # C10 gets M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10 (k=0, i=1)
    
    # M3 = A00 * (B01 - B11)
    # A00 is U[0,0]
    # B01 is V[1,1], B11 is V[3,3]
    # C01 gets M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01 (k=1, i=0)
    
    # M4 = A11 * (B10 - B00)
    # A11 is U[3,3]
    # B10 is V[2,0], B00 is V[0,0]
    # C00 gets M4, C10 gets M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[2, 3] = 1 # C10
    
    # M5 = (A00 + A01) * B11
    # A00 is U[0,0], A01 is U[0,1]
    # B11 is V[3,3]
    # C00 gets -M5, C01 gets M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6 = (A10 - A00) * (B00 + B01)
    # A10 is U[2,0], A00 is U[0,0]
    # B00 is V[0,0], B01 is V[1,1]
    # C11 gets M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7 = (A01 - A11) * (B10 + B11)
    # A01 is U[0,1], A11 is U[3,3]
    # B10 is V[2,0], B11 is V[3,3]
    # C00 gets M7
    U[0, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00

    return U, V, W

def _build_strassen_4x4():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication.
    U shape (16, 49), V shape (16, 49), W shape (16, 49).
    """
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    U2, V2, W2 = _build_strassen_2x2()
    
    # The 7 Strassen multiplications M1..M7
    # Each involves specific 2x2 blocks of A and B.
    # We need to map the 2x2 indices to 4x4 indices.
    
    # Block mapping for A (4x4) composed of 2x2 blocks:
    # A11: i in [0,2), j in [0,2) -> local (0,0)
    # A12: i in [0,2), j in [2,4) -> local (0,1)
    # A21: i in [2,4), j in [0,2) -> local (1,0)
    # A22: i in [2,4), j in [2,4) -> local (1,1)
    
    # Similarly for B.
    
    # We iterate over the 7 multiplications.
    # For each multiplication r (0..6), we have a set of (i,j) pairs for U,
    # (j,k) pairs for V, and (k,i) pairs for W with coefficients.
    
    # Instead of hardcoding the 49 entries, we can reuse the 2x2 builder
    # by embedding it.
    
    # The 4x4 Strassen uses 7 "super-multiplications" of 2x2 matrices.
    # Each super-multiplication is a 2x2x2 problem.
    # So we can place 7 copies of the 2x2x2 decomposition.
    
    # We need to know which 2x2 block of A and B is used in each of the 7 steps.
    
    # M1: A11+A22, B11+B22
    # M2: A21+A22, B11
    # M3: A11, B12-B22
    # M4: A22, B21-B11
    # M5: A11+A12, B22
    # M6: A21-A11, B11+B12
    # M7: A12-A22, B21+B22
    
    # Let's define the blocks involved in each M_r.
    # A_blocks[r] is a list of (block_row, block_col, coeff)
    # B_blocks[r] is a list of (block_row, block_col, coeff)
    
    # Block indices: 00, 01, 10, 11 corresponding to (0,0), (0,1), (1,0), (1,1)
    
    A_blocks = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],           # M3
        [(1,1,1)],           # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    
    B_blocks = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],           # M2
        [(0,1,1), (1,1,-1)],# M3
        [(1,0,1), (0,0,-1)],# M4
        [(1,1,1)],           # M5
        [(0,0,1), (0,1,1)], # M6
        [(1,0,1), (1,1,1)]  # M7
    ]
    
    # W blocks (output C blocks):
    # M1 contributes to C11 and C22
    # M2 contributes to C21
    # M3 contributes to C12
    # M4 contributes to C11 and C21
    # M5 contributes to C11 and C12
    # M6 contributes to C22
    # M7 contributes to C11
    
    # We need to map the local 2x2 result to the global 4x4 result.
    # The local 2x2 result has indices (k_loc, i_loc).
    # The global C block is at (block_row, block_col).
    # The global indices are (i_glob, k_glob).
    # Wait, W is indexed by (k, i).
    # In the 2x2 case, W is (k, i).
    # In the 4x4 case, the output block is C[br, bc].
    # The local i maps to global i = br*2 + i_loc.
    # The local k maps to global k = bc*2 + k_loc.
    
    W_blocks = [
        [(0,0,1), (1,1,1)], # M1 -> C11, C22
        [(1,0,1)],           # M2 -> C21
        [(0,1,1)],           # M3 -> C12
        [(0,0,1), (1,0,1)], # M4 -> C11, C21
        [(0,0,-1), (0,1,1)],# M5 -> C11, C12
        [(1,1,1)],           # M6 -> C22
        [(0,0,1)]            # M7 -> C11
    ]
    
    for r in range(7):
        # Get the 2x2 decomposition for this multiplication
        u2, v2, w2 = U2, V2, W2
        
        # Fill U
        for local_i in range(2):
            for local_j in range(2):
                u_val = u2[local_i*2 + local_j, :] # This is wrong, u2 is (4,7)
                # u2 is already the full 2x2 U.
                # We need to distribute the contributions.
                
        # This approach is getting complicated. Let's just hardcode the 49 rank terms
        # or use a simpler construction.
        
        # Alternative: Just use schoolbook for 4x4x4? Rank 64.
        # 64 + 64 + 16 = 144. No gain.
        
        # Let's try to implement the embedding correctly.
        # For each of the 7 multiplications, we have a 2x2x2 tensor.
        # We need to "inflate" it to 4x4x4.
        
        pass

    # Due to complexity, let's fall back to a simpler improvement.
    # We can use Strassen for 2x2x2 and build 4x4x4 from 4 blocks of 2x2x2?
    # No, that's just schoolbook.
    
    # Let's return schoolbook for now to ensure correctness, 
    # but the 114 rank is a strong target.
    # I will implement the 4x4x4 Strassen carefully.
    
    U_final = np.zeros((16, 49))
    V_final = np.zeros((16, 49))
    W_final = np.zeros((16, 49))
    
    # For each of the 7 multiplications r:
    for r in range(7):
        # Get the 2x2x2 U, V, W
        u2, v2, w2 = U2, V2, W2
        
        # For U:
        # The A blocks involved are in A_blocks[r].
        # For each (br, bc, coeff) in A_blocks[r]:
        #   For each local (i, j) in 2x2:
        #     Global i = br*2 + i
        #     Global j = bc*2 + j
        #     U[global_i*4 + global_j, 7*r + local_rank] += coeff * u2[local_i*2+local_j, local_rank]
        
        for br, bc, coeff_a in A_blocks[r]:
            for local_i in range(2):
                for local_j in range(2):
                    global_i = br * 2 + local_i
                    global_j = bc * 2 + local_j
                    u_idx = global_i * 4 + global_j
                    u2_idx = local_i * 2 + local_j
                    U_final[u_idx, 7*r:] += coeff_a * u2[u2_idx, :]
                    
        # For V:
        for br, bc, coeff_b in B_blocks[r]:
            for local_j in range(2):
                for local_k in range(2):
                    global_j = br * 2 + local_j
                    global_k = bc * 2 + local_k
                    v_idx = global_j * 4 + global_k
                    v2_idx = local_j * 2 + local_k
                    V_final[v_idx, 7*r:] += coeff_b * v2[v2_idx, :]
                    
        # For W:
        for br, bc, coeff_c in W_blocks[r]:
            for local_k in range(2):
                for local_i in range(2):
                    global_k = bc * 2 + local_k
                    global_i = br * 2 + local_i
                    w_idx = global_k * 4 + global_i
                    w2_idx = local_k * 2 + local_i
                    W_final[w_idx, 7*r:] += coeff_c * w2[w2_idx, :]
                    
    return U_final, V_final, W_final

def _build_schoolbook_4x4x1():
    """
    Build rank-16 decomposition for 4x4x1 matrix multiplication.
    U shape (16, 16), V shape (4, 16), W shape (4, 16).
    """
    rank = 16
    U = np.zeros((16, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
            
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    
    # Block 1: 4x4x4 (cols 0-3)
    U1, V1, W1 = _build_strassen_4x4()
    # Block 2: 4x4x4 (cols 4-7)
    U2, V2, W2 = _build_strassen_4x4()
    # Block 3: 4x4x1 (col 8)
    U3, V3, W3 = _build_schoolbook_4x4x1()
    
    # Combine
    # U is same for all blocks (16, 49) or (16, 16)
    # V is (16, 49) for blocks 1,2 and (4, 16) for block 3?
    # Wait, V shape is (m*p, rank) = (4*9, rank) = (36, rank).
    # In Block 1, V is (4*4, 49) = (16, 49).
    # We need to embed this in (36, 49).
    
    R1 = 49
    R2 = 49
    R3 = 16
    Total_Rank = R1 + R2 + R3
    
    U_total = np.zeros((16, Total_Rank))
    V_total = np.zeros((36, Total_Rank))
    W_total = np.zeros((36, Total_Rank))
    
    # Block 1
    U_total[:, 0:R1] = U1
    # V1 is (16, 49). Embed in V_total[0:16, 0:R1]
    V_total[0:16, 0:R1] = V1
    # W1 is (16, 49). Embed in W_total[0:16, 0:R1]
    W_total[0:16, 0:R1] = W1
    
    # Block 2
    U_total[:, R1:R1+R2] = U2
    # V2 is (16, 49). Embed in V_total[16:32, R1:R1+R2]
    V_total[16:32, R1:R1+R2] = V2
    # W2 is (16, 49). Embed in W_total[16:32, R1:R1+R2]
    W_total[16:32, R1:R1+R2] = W2
    
    # Block 3
    U_total[:, R1+R2:R1+R2+R3] = U3
    # V3 is (4, 16). Embed in V_total[32:36, R1+R2:R1+R2+R3]
    V_total[32:36, R1+R2:R1+R2+R3] = V3
    # W3 is (4, 16). Embed in W_total[32:36, R1+R2:R1+R2+R3]
    W_total[32:36, R1+R2:R1+R2+R3] = W3
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

