import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Total rank calculation
    # 6 blocks of (3, 2, 2)
    # Each (3, 2, 2) is split into (1, 2, 2) [Rank 1] and (2, 2, 2) [Rank 7]
    # Total rank per block = 8
    # Total rank = 6 * 8 = 48
    total_rank = 48
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen factors for 2x2x2
    # U_s: 2x7, V_s: 2x7, W_s: 2x7
    # Local indices: i in 0,1; j in 0,1; k in 0,1
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 1, 0, 1, 0, 1, 0]  # A10
    ])
    V_s = np.array([
        [1, 1, 0, 1, 0, 1, 1], # B00
        [0, 0, 1, 0, 0, 1, 0]  # B10
    ])
    # Wait, V indices are j*p+k. For 2x2, j=0,1; k=0,1.
    # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)?
    # No, V shape is (m*p, rank). For 2x2, m=2, p=2. Rows 0..3.
    # Row 0: j=0, k=0.
    # Row 1: j=0, k=1.
    # Row 2: j=1, k=0.
    # Row 3: j=1, k=1.
    # My previous U_s, V_s, W_s were for flattened indices.
    # Let's re-verify Strassen factors for 2x2x2 with flattened indices.
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # A indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U (A factors): 4x7
    U_s_flat = np.array([
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 1, 0, 0, 0, 1, 0], # A10
        [0, 0, 0, 0, 1, 0, 1], # A01
        [1, 1, 0, 1, 0, 0, 0]  # A11
    ])
    
    # V (B factors): 4x7
    V_s_flat = np.array([
        [1, 1, 0, 1, 0, 1, 1], # B00
        [1, 0, -1, -1, 0, 0, 1], # B01
        [0, 0, 1, 0, 0, 1, 0], # B10
        [1, 0, 0, 0, 1, 0, 1]  # B11
    ])
    
    # W (C factors): 4x7
    W_s_flat = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 0, 0, 1, 1, 0],  # C01
        [0, 1, 1, 0, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ])
    
    # Rank 1 for 1x2x2
    # A: 1x2. B: 2x2. C: 1x2.
    # U: 1x1. V: 2x1. W: 2x1.
    # But we need to map to flattened indices.
    # A index: 0.
    # B indices: 0,1 (for j=0), 2,3 (for j=1)?
    # Wait, for 1x2x2, m=2, p=2.
    # B is 2x2. Flattened indices 0..3.
    # V is 4x1.
    # V[0]=1, V[1]=1, V[2]=1, V[3]=1.
    # U is 1x1. U[0]=1.
    # W is 2x1 (p*n = 2*1 = 2).
    # W[0]=1, W[1]=1.
    
    U_122 = np.array([[1]])
    V_122 = np.array([[1], [1], [1], [1]])
    W_122 = np.array([[1], [1]])
    
    # Assemble
    rank_offset = 0
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            # Block 1: 1x2x2 (i=0)
            # U rows: i*m+j. i=0. j in [j_start, j_start+1].
            # U[0*m + j_start + 0, rank_offset] = 1
            # U[0*m + j_start + 1, rank_offset] = 1
            # Wait, U_122 is 1x1. It applies to A_0.
            # A_0 is at index 0 in the local 1x2 matrix?
            # Local A is 1x2. Flattened index 0.
            # Global A index: 0*m + j_global.
            # So U[0*m + j_start, rank_offset] += 1
            # U[0*m + j_start+1, rank_offset] += 1
            
            # V rows: j*p+k.
            # j in [j_start, j_start+1]. k in [k_start, k_start+1].
            # V[j_start*p + k_start, rank_offset] += 1
            # V[j_start*p + k_start+1, rank_offset] += 1
            # V[(j_start+1)*p + k_start, rank_offset] += 1
            # V[(j_start+1)*p + k_start+1, rank_offset] += 1
            
            # W rows: k*n+i.
            # k in [k_start, k_start+1]. i=0.
            # W[k_start*n + 0, rank_offset] += 1
            # W[(k_start+1)*n + 0, rank_offset] += 1
            
            U[0 * m + j_start, rank_offset] += 1
            U[0 * m + j_start + 1, rank_offset] += 1
            
            V[j_start * p + k_start, rank_offset] += 1
            V[j_start * p + k_start + 1, rank_offset] += 1
            V[(j_start + 1) * p + k_start, rank_offset] += 1
            V[(j_start + 1) * p + k_start + 1, rank_offset] += 1
            
            W[k_start * n + 0, rank_offset] += 1
            W[(k_start + 1) * n + 0, rank_offset] += 1
            
            rank_offset += 1
            
            # Block 2: 2x2x2 (i=1,2)
            # Local i: 0,1. Global i: 1,2.
            # Local j: 0,1. Global j: j_start, j_start+1.
            # Local k: 0,1. Global k: k_start, k_start+1.
            
            # U: 4x7.
            # Local A indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1).
            # Global A index: i_global * m + j_global.
            # i_global = 1 + i_local.
            # j_global = j_start + j_local.
            
            for r in range(7):
                # U_s_flat row 0 (local 0,0) -> global i=1, j=j_start
                U[1 * m + j_start, rank_offset + r] += U_s_flat[0, r]
                # U_s_flat row 1 (local 1,0) -> global i=2, j=j_start
                U[2 * m + j_start, rank_offset + r] += U_s_flat[1, r]
                # U_s_flat row 2 (local 0,1) -> global i=1, j=j_start+1
                U[1 * m + j_start + 1, rank_offset + r] += U_s_flat[2, r]
                # U_s_flat row 3 (local 1,1) -> global i=2, j=j_start+1
                U[2 * m + j_start + 1, rank_offset + r] += U_s_flat[3, r]
                
                # V_s_flat row 0 (local 0,0) -> global j=j_start, k=k_start
                V[j_start * p + k_start, rank_offset + r] += V_s_flat[0, r]
                # V_s_flat row 1 (local 0,1) -> global j=j_start, k=k_start+1
                V[j_start * p + k_start + 1, rank_offset + r] += V_s_flat[1, r]
                # V_s_flat row 2 (local 1,0) -> global j=j_start+1, k=k_start
                V[(j_start + 1) * p + k_start, rank_offset + r] += V_s_flat[2, r]
                # V_s_flat row 3 (local 1,1) -> global j=j_start+1, k=k_start+1
                V[(j_start + 1) * p + k_start + 1, rank_offset + r] += V_s_flat[3, r]
                
                # W_s_flat row 0 (local 0,0) -> global k=k_start, i=1
                W[k_start * n + 1, rank_offset + r] += W_s_flat[0, r]
                # W_s_flat row 1 (local 0,1) -> global k=k_start+1, i=1
                W[(k_start + 1) * n + 1, rank_offset + r] += W_s_flat[1, r]
                # W_s_flat row 2 (local 1,0) -> global k=k_start, i=2
                W[k_start * n + 2, rank_offset + r] += W_s_flat[2, r]
                # W_s_flat row 3 (local 1,1) -> global k=k_start+1, i=2
                W[(k_start + 1) * n + 2, rank_offset + r] += W_s_flat[3, r]
                
            rank_offset += 7
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
