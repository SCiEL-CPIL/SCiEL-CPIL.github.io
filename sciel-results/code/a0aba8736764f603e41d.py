import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U[i*2+j, c]
    V[i*2+j, c]
    W[i*2+j, c]
    """
    # Strassen formulas:
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
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1 # A11, A22
    V[0, 0] = 1; V[3, 0] = 1 # B11, B22
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1 # A21, A22
    V[0, 1] = 1 # B11
    W[2, 1] = 1 # C21
    
    # M3
    U[0, 2] = 1 # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12, -B22
    W[1, 2] = 1 # C12
    
    # M4
    U[3, 3] = 1 # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21, -B11
    W[0, 3] = 1; W[2, 3] = 1 # C11, C21
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1 # A11, A12
    V[3, 4] = 1 # B22
    W[0, 4] = -1; W[1, 4] = 1 # -C11, C12
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1 # B11, B12
    W[3, 5] = 1 # C22
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1 # B21, B22
    W[0, 6] = 1 # C11
    
    return U, V, W

def _schoolbook_1x2x2():
    """
    Returns U, V, W for 1x2x2 schoolbook decomposition.
    U shape: (2, 4)
    V shape: (2, 4)
    W shape: (2, 4)
    Indices:
    U[j, c] (A is 1x2, so index is j)
    V[k, c] (B is 2x2, so index is k? No, B is 2x2. Input to V is j*p+k? No.
    For 1x2x2: A is 1x2, B is 2x2.
    U rows: 1*2 = 2.
    V rows: 2*2 = 4.
    W rows: 2*1 = 2.
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    # C[k] = sum_j A[j] B[j,k]
    # 4 multiplications: A[0]B[0,0], A[0]B[0,1], A[1]B[1,0], A[1]B[1,1]
    
    # c=0: A[0] * B[0,0] -> C[0]
    U[0, 0] = 1
    V[0, 0] = 1 # B[0,0] is row 0, col 0. Index 0*2+0=0
    W[0, 0] = 1 # C[0] is row 0, col 0? W index is k*n+i. n=1, i=0. k=0. Index 0.
    
    # c=1: A[0] * B[0,1] -> C[1]
    U[0, 1] = 1
    V[1, 1] = 1 # B[0,1] index 0*2+1=1
    W[1, 1] = 1 # C[1] index 1*1+0=1
    
    # c=2: A[1] * B[1,0] -> C[0]
    U[1, 2] = 1
    V[2, 2] = 1 # B[1,0] index 1*2+0=2
    W[0, 2] = 1
    
    # c=3: A[1] * B[1,1] -> C[1]
    U[1, 3] = 1
    V[3, 3] = 1 # B[1,1] index 1*2+1=3
    W[1, 3] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS # 3, 4, 4
    
    # Fallback
    def _schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W

    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school

    try:
        # Construct hybrid decomposition
        # A: 3x4. Split into A1 (2x4) and A2 (1x4).
        # B: 4x4. Split into B1 (4x2) and B2 (4x2).
        
        # Sub-problems:
        # 1. A1 * B1 -> 2x2 (Strassen, rank 7)
        # 2. A1 * B2 -> 2x2 (Strassen, rank 7)
        # 3. A2 * B1 -> 1x2 (Schoolbook, rank 4)
        # 4. A2 * B2 -> 1x2 (Schoolbook, rank 4)
        
        U_total = []
        V_total = []
        W_total = []
        
        # We need to map the local indices to global indices.
        # Global U index: i*m + j. i in [0,2], j in [0,3].
        # Global V index: j*p + k. j in [0,3], k in [0,3].
        # Global W index: k*n + i. k in [0,3], i in [0,2].
        
        def add_block(U_loc, V_loc, W_loc, i_offset, j_offset, k_offset):
            """
            Add a block decomposition to the total matrices.
            U_loc shape: (n_loc*m_loc, r)
            V_loc shape: (m_loc*p_loc, r)
            W_loc shape: (p_loc*n_loc, r)
            
            i_offset: starting row index in A (0 or 2)
            j_offset: starting col index in A / row index in B (0 or 2)
            k_offset: starting col index in B (0 or 2)
            
            Wait, the split of B is into cols.
            B1 is cols 0,1. B2 is cols 2,3.
            So for B1, k_offset=0. For B2, k_offset=2.
            j is the middle index (0..3).
            
            For A1 (rows 0,1), i_offset=0.
            For A2 (row 2), i_offset=2.
            
            The middle index j is shared.
            For A1*B1, j is 0..3.
            For A1*B2, j is 0..3.
            For A2*B1, j is 0..3.
            For A2*B2, j is 0..3.
            
            So j_offset is always 0 for the middle index?
            No, the sub-problem dimensions are:
            A1: 2x4. B1: 4x2.
            U_loc rows: 2*4=8.
            V_loc rows: 4*2=8.
            W_loc rows: 2*2=4.
            
            Mapping U_loc[i_loc*m_loc + j_loc] to U_global[i_offset + i_loc]*m + (j_offset + j_loc)?
            No, j is the column of A.
            A1 has cols 0..3. So j_loc = j_global.
            A2 has cols 0..3. So j_loc = j_global.
            
            So j_offset = 0.
            i_offset = 0 for A1, 2 for A2.
            
            B1 has rows 0..3 (j) and cols 0..1 (k).
            V_loc[j_loc*p_loc + k_loc] where p_loc=2.
            j_loc = j_global.
            k_loc = k_global - k_offset.
            k_offset = 0 for B1, 2 for B2.
            
            W_loc[k_loc*n_loc + i_loc] where n_loc=2 (for A1) or 1 (for A2).
            k_loc = k_global - k_offset.
            i_loc = i_global - i_offset.
            """
            
            n_loc = U_loc.shape[0] // U_loc.shape[1] # This is wrong.
            # Let's pass dimensions explicitly.
            
        # Let's do it manually to be safe.
        
        # Block 1: A1 (2x4) * B1 (4x2) -> C11 (2x2)
        U1, V1, W1 = _strassen_2x2x2() # This is for 2x2x2.
        # But A1 is 2x4. B1 is 4x2.
        # We need to split A1 into two 2x2 blocks?
        # A1 = [A1_L, A1_R]. B1 = [B1_T; B1_B].
        # C11 = A1_L B1_T + A1_R B1_B.
        # This is sum of two 2x2x2 products.
        # So we need to apply Strassen to A1_L B1_T and A1_R B1_B.
        
        # Let's define a function to add a 2x2x2 Strassen block.
        
        def add_strassen_block(U_total, V_total, W_total, i_off, j_off, k_off):
            # i_off: row offset in A (0..2)
            # j_off: col offset in A (0..3) -- but A is 4 cols.
            # k_off: col offset in B (0..3)
            # This block computes C[i_off..i_off+1, k_off..k_off+1]
            # Using A[i_off..i_off+1, j_off..j_off+1] and B[j_off..j_off+1, k_off..k_off+1]?
            # No, the middle index j is the full 4.
            # Strassen 2x2x2 uses A (2x2), B (2x2).
            # Here A is 2x4, B is 4x2.
            # We can split A into A_L (2x2) and A_R (2x2).
            # And B into B_T (2x2) and B_B (2x2).
            # C = A_L B_T + A_R B_B.
            # This requires 2 Strassen blocks.
            
            # Block 1.1: A_L B_T
            U_s, V_s, W_s = _strassen_2x2x2()
            for c in range(7):
                for i_loc in range(2):
                    for j_loc in range(2):
                        u_idx = i_loc * 2 + j_loc
                        u_val = U_s[u_idx, c]
                        if u_val != 0:
                            i_glob = i_off + i_loc
                            j_glob = j_off + j_loc
                            U_total[i_glob * 4 + j_glob].append(u_val)
                
                for j_loc in range(2):
                    for k_loc in range(2):
                        v_idx = j_loc * 2 + k_loc
                        v_val = V_s[v_idx, c]
                        if v_val != 0:
                            j_glob = j_off + j_loc
                            k_glob = k_off + k_loc
                            V_total[j_glob * 4 + k_glob].append(v_val)
                
                for k_loc in range(2):
                    for i_loc in range(2):
                        w_idx = k_loc * 2 + i_loc
                        w_val = W_s[w_idx, c]
                        if w_val != 0:
                            k_glob = k_off + k_loc
                            i_glob = i_off + i_loc
                            W_total[k_glob * 3 + i_glob].append(w_val)
            
            # Block 1.2: A_R B_B
            # A_R is cols 2,3 of A1. j_off = 2.
            # B_B is rows 2,3 of B1. j_off = 2.
            # Wait, B1 is 4x2. Rows 0..3.
            # B_T is rows 0,1. B_B is rows 2,3.
            # A_L is cols 0,1. A_R is cols 2,3.
            
            U_s, V_s, W_s = _strassen_2x2x2()
            for c in range(7):
                for i_loc in range(2):
                    for j_loc in range(2):
                        u_idx = i_loc * 2 + j_loc
                        u_val = U_s[u_idx, c]
                        if u_val != 0:
                            i_glob = i_off + i_loc
                            j_glob = (j_off + 2) + j_loc # A_R offset
                            U_total[i_glob * 4 + j_glob].append(u_val)
                
                for j_loc in range(2):
                    for k_loc in range(2):
                        v_idx = j_loc * 2 + k_loc
                        v_val = V_s[v_idx, c]
                        if v_val != 0:
                            j_glob = (j_off + 2) + j_loc # B_B offset
                            k_glob = k_off + k_loc
                            V_total[j_glob * 4 + k_glob].append(v_val)
                
                for k_loc in range(2):
                    for i_loc in range(2):
                        w_idx = k_loc * 2 + i_loc
                        w_val = W_s[w_idx, c]
                        if w_val != 0:
                            k_glob = k_off + k_loc
                            i_glob = i_off + i_loc
                            W_total[k_glob * 3 + i_glob].append(w_val)

        # Initialize totals
        # We don't know the total rank yet. We'll use lists and pad later.
        # Rank 1 (A1B1): 14
        # Rank 2 (A1B2): 14
        # Rank 3 (A2B1): 4
        # Rank 4 (A2B2): 4
        # Total 36.
        
        U_total = [[] for _ in range(12)] # 3*4
        V_total = [[] for _ in range(16)] # 4*4
        W_total = [[] for _ in range(12)] # 4*3
        
        # Block 1: A1 (rows 0,1) * B1 (cols 0,1)
        # i_off=0, j_off=0 (for A_L), k_off=0 (for B1)
        # But we need to handle A_L and A_R separately as above.
        # My add_strassen_block above assumes it handles one 2x2x2.
        # I called it twice inside? No, I wrote logic for 2 blocks inside.
        # Let's call the logic for A1B1.
        
        # A1 is rows 0,1. B1 is cols 0,1.
        # A_L is j=0,1. A_R is j=2,3.
        # B_T is j=0,1. B_B is j=2,3.
        
        # Call 1: A_L B_T
        # i_off=0, j_off=0, k_off=0
        U_s, V_s, W_s = _strassen_2x2x2()
        for c in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    u_val = U_s[i_loc*2+j_loc, c]
                    if u_val != 0:
                        U_total[(0+i_loc)*4 + (0+j_loc)].append(u_val)
            for j_loc in range(2):
                for k_loc in range(2):
                    v_val = V_s[j_loc*2+k_loc, c]
                    if v_val != 0:
                        V_total[(0+j_loc)*4 + (0+k_loc)].append(v_val)
            for k_loc in range(2):
                for i_loc in range(2):
                    w_val = W_s[k_loc*2+i_loc, c]
                    if w_val != 0:
                        W_total[(0+k_loc)*3 + (0+i_loc)].append(w_val)

        # Call 2: A_R B_B
        # i_off=0, j_off=2, k_off=0
        U_s, V_s, W_s = _strassen_2x2x2()
        for c in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    u_val = U_s[i_loc*2+j_loc, c]
                    if u_val != 0:
                        U_total[(0+i_loc)*4 + (2+j_loc)].append(u_val)
            for j_loc in range(2):
                for k_loc in range(2):
                    v_val = V_s[j_loc*2+k_loc, c]
                    if v_val != 0:
                        V_total[(2+j_loc)*4 + (0+k_loc)].append(v_val)
            for k_loc in range(2):
                for i_loc in range(2):
                    w_val = W_s[k_loc*2+i_loc, c]
                    if w_val != 0:
                        W_total[(0+k_loc)*3 + (0+i_loc)].append(w_val)

        # Block 2: A1 (rows 0,1) * B2 (cols 2,3)
        # i_off=0, k_off=2
        # A_L B_T (B_T is rows 0,1 of B2? No, B2 is 4x2. Rows 0..3. Cols 2,3.)
        # B2_T is rows 0,1. B2_B is rows 2,3.
        
        # Call 3: A_L B2_T
        # i_off=0, j_off=0, k_off=2
        U_s, V_s, W_s = _strassen_2x2x2()
        for c in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    u_val = U_s[i_loc*2+j_loc, c]
                    if u_val != 0:
                        U_total[(0+i_loc)*4 + (0+j_loc)].append(u_val)
            for j_loc in range(2):
                for k_loc in range(2):
                    v_val = V_s[j_loc*2+k_loc, c]
                    if v_val != 0:
                        V_total[(0+j_loc)*4 + (2+k_loc)].append(v_val)
            for k_loc in range(2):
                for i_loc in range(2):
                    w_val = W_s[k_loc*2+i_loc, c]
                    if w_val != 0:
                        W_total[(2+k_loc)*3 + (0+i_loc)].append(w_val)

        # Call 4: A_R B2_B
        # i_off=0, j_off=2, k_off=2
        U_s, V_s, W_s = _strassen_2x2x2()
        for c in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    u_val = U_s[i_loc*2+j_loc, c]
                    if u_val != 0:
                        U_total[(0+i_loc)*4 + (2+j_loc)].append(u_val)
            for j_loc in range(2):
                for k_loc in range(2):
                    v_val = V_s[j_loc*2+k_loc, c]
                    if v_val != 0:
                        V_total[(2+j_loc)*4 + (2+k_loc)].append(v_val)
            for k_loc in range(2):
                for i_loc in range(2):
                    w_val = W_s[k_loc*2+i_loc, c]
                    if w_val != 0:
                        W_total[(2+k_loc)*3 + (0+i_loc)].append(w_val)

        # Block 3: A2 (row 2) * B1 (cols 0,1)
        # A2 is 1x4. B1 is 4x2.
        # Split B1 into B1_T (2x2) and B1_B (2x2).
        # C3 = A2_L B1_T + A2_R B1_B.
        # A2_L is cols 0,1. A2_R is cols 2,3.
        # Use Schoolbook 1x2x2.
        
        U_s, V_s, W_s = _schoolbook_1x2x2()
        for c in range(4):
            for j_loc in range(2):
                u_val = U_s[j_loc, c]
                if u_val != 0:
                    U_total[(2)*4 + (0+j_loc)].append(u_val)
            for j_loc in range(2):
                for k_loc in range(2):
                    v_val = V_s[j_loc*2+k_loc, c]
                    if v_val != 0:
                        V_total[(0+j_loc)*4 + (0+k_loc)].append(v_val)
            for k_loc in range(2):
                for i_loc in range(1):
                    w_val = W_s[k_loc*1+i_loc, c]
                    if w_val != 0:
                        W_total[(0+k_loc)*3 + (2+i_loc)].append(w_val)

        # Call 6: A2_R B1_B
        # i_off=2, j_off=2, k_off=0
        U_s, V_s, W_s = _schoolbook_1x2x2()
        for c in range(4):
            for j_loc in range(2):
                u_val = U_s[j_loc, c]
                if u_val != 0:
                    U_total[(2)*4 + (2+j_loc)].append(u_val)
            for j_loc in range(2):
                for k_loc in range(2):
                    v_val = V_s[j_loc*2+k_loc, c]
                    if v_val != 0:
                        V_total[(2+j_loc)*4 + (0+k_loc)].append(v_val)
            for k_loc in range(2):
                for i_loc in range(1):
                    w_val = W_s[k_loc*1+i_loc, c]
                    if w_val != 0:
                        W_total[(0+k_loc)*3 + (2+i_loc)].append(w_val)

        # Block 4: A2 (row 2) * B2 (cols 2,3)
        # Call 7: A2_L B2_T
        # i_off=2, j_off=0, k_off=2
        U_s, V_s, W_s = _schoolbook_1x2x2()
        for c in range(4):
            for j_loc in range(2):
                u_val = U_s[j_loc, c]
                if u_val != 0:
                    U_total[(2)*4 + (0+j_loc)].append(u_val)
            for j_loc in range(2):
                for k_loc in range(2):
                    v_val = V_s[j_loc*2+k_loc, c]
                    if v_val != 0:
                        V_total[(0+j_loc)*4 + (2+k_loc)].append(v_val)
            for k_loc in range(2):
                for i_loc in range(1):
                    w_val = W_s[k_loc*1+i_loc, c]
                    if w_val != 0:
                        W_total[(2+k_loc)*3 + (2+i_loc)].append(w_val)

        # Call 8: A2_R B2_B
        # i_off=2, j_off=2, k_off=2
        U_s, V_s, W_s = _schoolbook_1x2x2()
        for c in range(4):
            for j_loc in range(2):
                u_val = U_s[j_loc, c]
                if u_val != 0:
                    U_total[(2)*4 + (2+j_loc)].append(u_val)
            for j_loc in range(2):
                for k_loc in range(2):
                    v_val = V_s[j_loc*2+k_loc, c]
                    if v_val != 0:
                        V_total[(2+j_loc)*4 + (2+k_loc)].append(v_val)
            for k_loc in range(2):
                for i_loc in range(1):
                    w_val = W_s[k_loc*1+i_loc, c]
                    if w_val != 0:
                        W_total[(2+k_loc)*3 + (2+i_loc)].append(w_val)

        # Convert lists to numpy arrays
        # Find max length
        max_rank = max(len(row) for row in U_total + V_total + W_total)
        if max_rank == 0:
            return {"U": U_school, "V": V_school, "W": W_school}
            
        U_np = np.zeros((12, max_rank))
        V_np = np.zeros((16, max_rank))
        W_np = np.zeros((12, max_rank))
        
        for i in range(12):
            for c in range(len(U_total[i])):
                U_np[i, c] = U_total[i][c]
        for j in range(16):
            for c in range(len(V_total[j])):
                V_np[j, c] = V_total[j][c]
        for k in range(12):
            for c in range(len(W_total[k])):
                W_np[k, c] = W_total[k][c]
        
        # Verify
        valid = True
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    total = 0.0
                    for c in range(max_rank):
                        total += U_np[i*4+j, c] * V_np[j*4+k, c] * W_np[k*3+i, c]
                    if abs(total - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            best_rank = max_rank
            best_U = U_np.tolist()
            best_V = V_np.tolist()
            best_W = W_np.tolist()
            
    except Exception as e:
        pass

    return {"U": best_U, "V": best_V, "W": best_W}
