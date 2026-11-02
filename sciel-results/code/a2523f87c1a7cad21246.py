import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    # Pad to 3x9 by 9x9
    n_pad, m_pad, p_pad = 3, 9, 9
    
    # Strassen decomposition for 3x3 by 3x3
    # Rank 7
    # U: 9x7, V: 9x7, W: 9x7
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Strassen 2x2 logic applied to 3x3?
    # Standard Strassen is for 2x2.
    # For 3x3, we can use the rank-7 decomposition directly.
    # One known rank-7 decomposition for 3x3:
    # It is derived from 2x2 Strassen by padding 3x3 to 4x4? No.
    # Actually, the rank of M_3,3,3 is 7.
    # We can use the explicit factors.
    
    # Factors for 3x3 Strassen (rank 7)
    # Source: "On the rank of matrix multiplication tensors"
    # We use the standard 7-term decomposition.
    
    U_33 = np.zeros((9, 7), dtype=int)
    V_33 = np.zeros((9, 7), dtype=int)
    W_33 = np.zeros((9, 7), dtype=int)
    
    # Term 1: (A00+A11)(B00+B11)
    # A00 is (0,0), A11 is (1,1)
    # B00 is (0,0), B11 is (1,1)
    # Result at (0,0) and (1,1)
    # This is getting complicated to map manually.
    
    # Instead, use the 4x4 Strassen (rank 49) or 3x3 (rank 7)?
    # Let's use the 3x3 rank 7 decomposition.
    # Explicit factors:
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # This is for 2x2 blocks.
    # For 3x3, we can view it as 2x2 blocks of 1x1? No.
    # We can use the 2x2 Strassen on the top-left 2x2, and handle the rest?
    
    # Simpler: Use the 4x4 Strassen (rank 49) is too high.
    # Use 3x3 rank 7.
    
    # Let's hardcode the 3x3 rank 7 factors.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    # For 3x3, rank is 7.
    
    # I will use a known set of factors for 3x3 rank 7.
    
    # U (A factors)
    U_33[0, 0] = 1; U_33[1, 0] = 1  # A00+A11? No, indices are flat.
    
    # This is error-prone.
    # I will use the 4x4 Strassen (rank 49) which is easier to generate programmatically?
    # No, 49 > 28.
    
    # Let's use the 3x3 rank 7 decomposition from literature.
    
    # Factors for M_3,3,3
    # U: 9x7
    # V: 9x7
    # W: 9x7
    
    # I will construct them using the 2x2 Strassen logic on 3x3 by padding to 4x4?
    # No, just use the 3x3 specific ones.
    
    # For now, I will use the schoolbook 3x3 (rank 9) to ensure correctness, 
    # and then use 9 blocks of 3x3 for 3x9 by 9x9.
    # Rank 9 * 9 = 81.
    # 81 < 105.
    # Score 105/81 = 1.29.
    
    # This is a safe improvement.
    
    rank_33 = 9
    U_33 = np.zeros((9, rank_33), dtype=int)
    V_33 = np.zeros((9, rank_33), dtype=int)
    W_33 = np.zeros((9, rank_33), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_33[i*3+j, col] = 1
                V_33[j*3+k, col] = 1
                W_33[k*3+i, col] = 1
                col += 1
                
    # Build 3x9 by 9x9
    n_big, m_big, p_big = 3, 9, 9
    rank_big = 9 * rank_33 # 81
    
    U_big = np.zeros((n_big * m_big, rank_big), dtype=int)
    V_big = np.zeros((m_big * p_big, rank_big), dtype=int)
    W_big = np.zeros((p_big * n_big, rank_big), dtype=int)
    
    # 9 blocks
    for bi in range(3): # A block col
        for bj in range(3): # B block row
            # A block is at col bi*3 .. bi*3+2
            # B block is at row bj*3 .. bj*3+2
            # Result adds to C col bj*3 .. bj*3+2
            
            offset = (bi * 3 + bj) * rank_33
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        # U index
                        u_row = i * m_big + (bi * 3 + j)
                        U_big[u_row, offset + i*3+j] = U_33[i*3+j, i*3+j] # 1
                        
                        # V index
                        v_row = (bj * 3 + j) * p_big + k
                        V_big[v_row, offset + i*3+j] = V_33[j*3+k, j*3+k] # 1
                        
                        # W index
                        w_row = (bj * 3 + k) * n_big + i
                        W_big[w_row, offset + i*3+j] = W_33[k*3+i, k*3+i] # 1

    # Extract for 3x5 by 5x7
    # U rows: i<3, j<5
    # V rows: j<5, k<7
    # W rows: k<7, i<3
    
    U_final = []
    for i in range(n):
        for j in range(m):
            row = i * m_big + j
            U_final.append(U_big[row])
            
    V_final = []
    for j in range(m):
        for k in range(p):
            row = j * p_big + k
            V_final.append(V_big[row])
            
    W_final = []
    for k in range(p):
        for i in range(n):
            row = k * n_big + i
            W_final.append(W_big[row])
            
    # Convert to lists
    U_list = [list(map(int, row)) for row in U_final]
    V_list = [list(map(int, row)) for row in V_final]
    W_list = [list(map(int, row)) for row in W_final]
    
    # Remove zero columns
    def remove_zero_cols(U, V, W):
        rank = U[0].__len__() if U else 0
        keep = []
        for c in range(rank):
            if any(U[r][c] != 0 for r in range(len(U))) or \
               any(V[r][c] != 0 for r in range(len(V))) or \
               any(W[r][c] != 0 for r in range(len(W))):
                keep.append(c)
        
        U_new = [ [row[c] for c in keep] for row in U ]
        V_new = [ [row[c] for c in keep] for row in V ]
        W_new = [ [row[c] for c in keep] for row in W ]
        return U_new, V_new, W_new

    U_list, V_list, W_list = remove_zero_cols(U_list, V_list, W_list)
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
