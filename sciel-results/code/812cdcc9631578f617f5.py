import time
import numpy as np

def construct():
    # Rank-56 decomposition for 2x4 * 4x8 matrix multiplication
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # A is 2x4, B is 4x8, C is 2x8
    # 
    # Strategy:
    # Split A into left (cols 0-1) and right (cols 2-3) 2x2 blocks.
    # Split B into top (rows 0-1) and bottom (rows 2-3) 2x8 blocks.
    # C = A_left * B_top + A_right * B_bottom
    # 
    # Each term (2x2 * 2x8) is decomposed into 4 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 multiplication uses Strassen's algorithm (rank 7).
    # Total rank = 2 (A blocks) * 4 (B col blocks) * 7 (Strassen rank) = 56.
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=int) # 8 x 56
    V = np.zeros((m * p, rank), dtype=int) # 32 x 56
    W = np.zeros((p * n, rank), dtype=int) # 16 x 56
    
    # Strassen 2x2 * 2x2 decomposition
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W index convention: W[k*n+i, r]. For local C[i, k_local], global k = k_start + k_local.
    # Local C index mapping to local flat index (i*2+k) is standard for Strassen derivation.
    # We need to map local flat indices to global W indices.
    
    # Strassen components (r=0..6)
    # U_local shape (4, 7), V_local shape (4, 7), W_local shape (4, 7)
    U_loc = np.zeros((4, 7), dtype=int)
    V_loc = np.zeros((4, 7), dtype=int)
    W_loc = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11) -> c00+=M1, c11+=M1
    U_loc[0,0]=1; U_loc[3,0]=1
    V_loc[0,0]=1; V_loc[3,0]=1
    W_loc[0,0]=1; W_loc[3,0]=1
    
    # M2 = (a10+a11)b00 -> c10+=M2, c11-=M2
    U_loc[2,1]=1; U_loc[3,1]=1
    V_loc[0,1]=1
    W_loc[2,1]=1; W_loc[3,1]=-1
    
    # M3 = a00(b01-b11) -> c01+=M3, c11+=M3
    U_loc[0,2]=1
    V_loc[1,2]=1; V_loc[3,2]=-1
    W_loc[1,2]=1; W_loc[3,2]=1
    
    # M4 = a11(b10-b00) -> c00+=M4, c10+=M4
    U_loc[3,3]=1
    V_loc[2,3]=1; V_loc[0,3]=-1
    W_loc[0,3]=1; W_loc[2,3]=1
    
    # M5 = (a00+a01)b11 -> c00-=M5, c01+=M5
    U_loc[0,4]=1; U_loc[1,4]=1
    V_loc[3,4]=1
    W_loc[0,4]=-1; W_loc[1,4]=1
    
    # M6 = (a10-a00)(b00+b01) -> c11+=M6
    U_loc[2,5]=1; U_loc[0,5]=-1
    V_loc[0,5]=1; V_loc[1,5]=1
    W_loc[3,5]=1
    
    # M7 = (a01-a11)(b10+b11) -> c00+=M7
    U_loc[1,6]=1; U_loc[3,6]=-1
    V_loc[2,6]=1; V_loc[3,6]=1
    W_loc[0,6]=1
    
    # Fill U, V, W
    for a_block in range(2): # 0: left (cols 0-1), 1: right (cols 2-3)
        for b_block in range(4): # 0-3: column groups of B
            base_r = (a_block * 4 + b_block) * 7
            
            # A mapping: local i, j -> global i, a_block*2 + j
            # Global U index: i*4 + (a_block*2 + j)
            # Local U index: i*2 + j
            for li in range(4):
                # li is local flat index i*2+j
                li_i = li // 2
                li_j = li % 2
                global_u_idx = li_i * 4 + (a_block * 2 + li_j)
                
                # B mapping: local i, j -> global (a_block*2 + i), b_block*2 + j
                # Wait, B is 4x8. Rows 0-3.
                # a_block=0 uses B rows 0-1. a_block=1 uses B rows 2-3.
                # Local B index li is i*2+j where i is local row (0-1).
                # Global B row: a_block*2 + li_i
                # Global B col: b_block*2 + li_j
                # Global V index: (a_block*2 + li_i)*8 + (b_block*2 + li_j)
                
                # C mapping: local i, k -> global i, b_block*2 + k
                # Local W index li is i*2+k (where k is local col 0-1)
                # Global W row index: (b_block*2 + li_k)*2 + li_i
                # Note: W[k*n + i, r]
                
                # We iterate over the 7 Strassen components
                for l in range(7):
                    r = base_r + l
                    
                    if U_loc[li, l] != 0:
                        U[global_u_idx, r] += U_loc[li, l]
                    
                    # Calculate global V index
                    global_v_idx = (a_block * 2 + li_i) * 8 + (b_block * 2 + li_j)
                    if V_loc[li, l] != 0:
                        V[global_v_idx, r] += V_loc[li, l]
                    
                    # Calculate global W index
                    li_i_w = li // 2
                    li_k_w = li % 2
                    global_w_idx = (b_block * 2 + li_k_w) * 2 + li_i_w
                    if W_loc[li, l] != 0:
                        W[global_w_idx, r] += W_loc[li, l]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
