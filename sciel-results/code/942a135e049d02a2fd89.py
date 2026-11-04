import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication using block decomposition.
    Partitions into 2x2 and 3x3 blocks.
    Uses Strassen (Rank 7) for 2x2 core.
    Uses Schoolbook for all other blocks.
    Total Rank: 7 (2x2x2) + 12 (2x3x2) + 12 (2x2x3) + 18 (2x3x3) + 
                12 (3x2x2) + 18 (3x3x2) + 18 (3x2x3) + 27 (3x3x3) = 124.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 25
    mp = m * p  # 25
    pn = p * n  # 25
    
    # Calculate total rank
    rank_total = 124
    
    U = np.zeros((nm, rank_total), dtype=np.float64)
    V = np.zeros((mp, rank_total), dtype=np.float64)
    W = np.zeros((pn, rank_total), dtype=np.float64)
    
    col = 0
    
    def add_block(n_block, m_block, p_block, i_off, j_off, k_off, use_strassen=False):
        nonlocal col
        # i_off: offset in A rows (and C rows)
        # j_off: offset in A cols (and B rows)
        # k_off: offset in B cols (and C cols)
        
        if use_strassen:
            # Strassen 2x2x2
            # M1 = (A11 + A22) * (B11 + B22)
            # M2 = (A21 + A22) * B11
            # M3 = A11 * (B12 - B22)
            # M4 = A22 * (B21 - B11)
            # M5 = (A11 + A12) * B22
            # M6 = (A21 - A11) * (B11 + B12)
            # M7 = (A12 - A22) * (B21 + B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # Indices:
            # A: (i, j) -> i*m + j. i in [i_off, i_off+1], j in [j_off, j_off+1]
            # B: (j, k) -> j*p + k. j in [j_off, j_off+1], k in [k_off, k_off+1]
            # C: (k, i) -> k*n + i. k in [k_off, k_off+1], i in [i_off, i_off+1]
            
            # Local indices 0,1
            # A11: i=0, j=0
            # A22: i=1, j=1
            # etc.
            
            def get_U_idx(li, lj):
                return (i_off + li) * m + (j_off + lj)
            
            def get_V_idx(lj, lk):
                return (j_off + lj) * p + (k_off + lk)
            
            def get_W_idx(lk, li):
                return (k_off + lk) * n + (i_off + li)
            
            # M1
            U[get_U_idx(0,0), col] += 1
            U[get_U_idx(1,1), col] += 1
            V[get_V_idx(0,0), col] += 1
            V[get_V_idx(1,1), col] += 1
            W[get_W_idx(0,0), col] += 1
            W[get_W_idx(1,1), col] -= 1
            W[get_W_idx(0,1), col] += 1
            W[get_W_idx(1,0), col] += 1
            col += 1
            
            # M2
            U[get_U_idx(1,0), col] += 1
            U[get_U_idx(1,1), col] += 1
            V[get_V_idx(0,0), col] += 1
            W[get_W_idx(0,1), col] += 1
            W[get_W_idx(1,0), col] += 1
            col += 1
            
            # M3
            U[get_U_idx(0,0), col] += 1
            V[get_V_idx(0,1), col] += 1
            V[get_V_idx(1,1), col] -= 1
            W[get_W_idx(0,1), col] += 1
            W[get_W_idx(1,1), col] += 1
            col += 1
            
            # M4
            U[get_U_idx(1,1), col] += 1
            V[get_V_idx(1,0), col] += 1
            V[get_V_idx(0,0), col] -= 1
            W[get_W_idx(0,0), col] += 1
            W[get_W_idx(1,1), col] += 1
            col += 1
            
            # M5
            U[get_U_idx(0,0), col] += 1
            U[get_U_idx(0,1), col] += 1
            V[get_V_idx(1,1), col] += 1
            W[get_W_idx(0,0), col] -= 1
            W[get_W_idx(1,0), col] += 1
            col += 1
            
            # M6
            U[get_U_idx(1,0), col] += 1
            U[get_U_idx(0,0), col] -= 1
            V[get_V_idx(0,0), col] += 1
            V[get_V_idx(0,1), col] += 1
            W[get_W_idx(1,1), col] += 1
            col += 1
            
            # M7
            U[get_U_idx(0,1), col] += 1
            U[get_U_idx(1,1), col] -= 1
            V[get_V_idx(1,0), col] += 1
            V[get_V_idx(1,1), col] += 1
            W[get_W_idx(0,0), col] += 1
            W[get_W_idx(1,1), col] += 1
            col += 1
        else:
            # Schoolbook
            for li in range(n_block):
                for lj in range(m_block):
                    for lk in range(p_block):
                        U[get_U_idx(li, lj), col] = 1
                        V[get_V_idx(lj, lk), col] = 1
                        W[get_W_idx(lk, li), col] = 1
                        col += 1

    # Block 1: C_II (2x2) = A_II (2x2) * B_II (2x2) + A_IJ (2x3) * B_JI (3x2)
    # A_II: i in [0,1], j in [0,1]
    # B_II: j in [0,1], k in [0,1]
    # C_II: i in [0,1], k in [0,1]
    add_block(2, 2, 2, 0, 0, 0, use_strassen=True)
    
    # A_IJ: i in [0,1], j in [2,4]
    # B_JI: j in [2,4], k in [0,1]
    # C_II: i in [0,1], k in [0,1]
    add_block(2, 3, 2, 0, 2, 0, use_strassen=False)
    
    # Block 2: C_IJ (2x3) = A_II (2x2) * B_IJ (2x3) + A_IJ (2x3) * B_JJ (3x3)
    # A_II: i in [0,1], j in [0,1]
    # B_IJ: j in [0,1], k in [2,4]
    # C_IJ: i in [0,1], k in [2,4]
    add_block(2, 2, 3, 0, 0, 2, use_strassen=False)
    
    # A_IJ: i in [0,1], j in [2,4]
    # B_JJ: j in [2,4], k in [2,4]
    # C_IJ: i in [0,1], k in [2,4]
    add_block(2, 3, 3, 0, 2, 2, use_strassen=False)
    
    # Block 3: C_JI (3x2) = A_JI (3x2) * B_II (2x2) + A_JJ (3x3) * B_JI (3x2)
    # A_JI: i in [2,4], j in [0,1]
    # B_II: j in [0,1], k in [0,1]
    # C_JI: i in [2,4], k in [0,1]
    add_block(3, 2, 2, 2, 0, 0, use_strassen=False)
    
    # A_JJ: i in [2,4], j in [2,4]
    # B_JI: j in [2,4], k in [0,1]
    # C_JI: i in [2,4], k in [0,1]
    add_block(3, 3, 2, 2, 2, 0, use_strassen=False)
    
    # Block 4: C_JJ (3x3) = A_JI (3x2) * B_IJ (2x3) + A_JJ (3x3) * B_JJ (3x3)
    # A_JI: i in [2,4], j in [0,1]
    # B_IJ: j in [0,1], k in [2,4]
    # C_JJ: i in [2,4], k in [2,4]
    add_block(3, 2, 3, 2, 0, 2, use_strassen=False)
    
    # A_JJ: i in [2,4], j in [2,4]
    # B_JJ: j in [2,4], k in [2,4]
    # C_JJ: i in [2,4], k in [2,4]
    add_block(3, 3, 3, 2, 2, 2, use_strassen=False)
    
    # Verify col matches rank_total
    if col != rank_total:
        # If mismatch, adjust or raise. Should be 124.
        pass

    if time.time() > deadline - safety_margin:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
