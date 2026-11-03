"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-minimal decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Block decomposition
    - Split A (4x5) into top (2x5) and bottom (2x5)
    - Split B (5x5) into left (5x2) and right (5x3)
    - C = A*B gives 4x5 result, split into 4 blocks of 2x2 and 2x3
    
    For each block:
    - C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2]
    - C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:5]*B[2:5, 2:5]
    - C[2:4, 0:2] = A[2:4, 0:2]*B[0:2, 0:2] + A[2:4, 2:5]*B[2:5, 0:2]
    - C[2:4, 2:5] = A[2:4, 0:2]*B[0:2, 2:5] + A[2:4, 2:5]*B[2:5, 2:5]
    
    Each term is a matrix product that we decompose using known low-rank algorithms.
    """
    start_time = time.time()
    deadline = 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Index convention: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    # Helper to add a decomposition for a sub-problem
    # Sub-problem: multiply A_sub (n_s x m_s) by B_sub (m_s x p_s)
    # Returns local U, V, W and we map to global indices
    
    def schoolbook_decomp(n_s, m_s, p_s):
        """Standard schoolbook decomposition of rank n_s*m_s*p_s"""
        rank = n_s * m_s * p_s
        U = np.zeros((n_s * m_s, rank))
        V = np.zeros((m_s * p_s, rank))
        W = np.zeros((p_s * n_s, rank))
        c = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U[i * m_s + j, c] = 1
                    V[j * p_s + k, c] = 1
                    W[k * n_s + i, c] = 1
                    c += 1
        return U, V, W, rank
    
    def strassen_2x2x2():
        """Strassen's algorithm for 2x2x2, rank 7"""
        # For C = A*B where A, B are 2x2
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        # 
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # 
        # c11 = M1 - M4 + M5 + M6
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M7
        
        n_s, m_s, p_s = 2, 2, 2
        rank = 7
        U = np.zeros((n_s * m_s, rank))
        V = np.zeros((m_s * p_s, rank))
        W = np.zeros((p_s * n_s, rank))
        
        # Index mapping: A row i, col j -> U index i*m_s + j = i*2 + j
        #                 B row j, col k -> V index j*p_s + k = j*2 + k
        #                 C row k, col i -> W index k*n_s + i = k*2 + i
        # 
        # A indices: a11=0, a12=1, a21=2, a22=3
        # B indices: b11=0, b12=1, b21=2, b22=3
        # C indices: c11=0, c12=1, c21=2, c22=3
        
        # M1 = (a11+a22)(b11+b22)
        U[0, 0] = 1  # a11
        U[3, 0] = 1  # a22
        V[0, 0] = 1  # b11
        V[3, 0] = 1  # b22
        W[0, 0] = 1  # c11
        W[3, 0] = 1  # c22
        
        # M2 = (a21+a22)b11
        U[2, 1] = 1  # a21
        U[3, 1] = 1  # a22
        V[0, 1] = 1  # b11
        W[2, 1] = 1  # c21
        W[3, 1] = -1 # c22 (negative)
        
        # M3 = a11(b21-b22)
        U[0, 2] = 1  # a11
        V[2, 2] = 1  # b21
        V[3, 2] = -1 # b22 (negative)
        W[1, 2] = 1  # c12
        W[3, 2] = 1  # c22
        
        # M4 = a22(b12-b11)
        U[3, 3] = 1  # a22
        V[1, 3] = 1  # b12
        V[0, 3] = -1 # b11 (negative)
        W[0, 3] = -1 # c11 (negative)
        W[2, 3] = 1  # c21
        
        # M5 = (a11+a12)b22
        U[0, 4] = 1  # a11
        U[1, 4] = 1  # a12
        V[3, 4] = 1  # b22
        W[0, 4] = 1  # c11
        W[1, 4] = 1  # c12
        
        # M6 = (a21-a11)(b11+b12)
        U[2, 5] = 1  # a21
        U[0, 5] = -1 # a11 (negative)
        V[0, 5] = 1  # b11
        V[1, 5] = 1  # b12
        W[0, 5] = 1  # c11
        W[2, 5] = 1  # c21
        
        # M7 = (a12-a22)(b21+b22)
        U[1, 6] = 1  # a12
        U[3, 6] = -1 # a22 (negative)
        V[2, 6] = 1  # b21
        V[3, 6] = 1  # b22
        W[3, 6] = 1  # c22
        
        return U, V, W, rank
    
    def rank11_2x2x3():
        """
        Rank 11 decomposition for 2x2 * 2x3 = 2x3
        This is a known result. Let me construct it.
        
        A is 2x2, B is 2x3, C is 2x3.
        
        Actually, let's use a simpler approach: split B into two 2x2 blocks plus one 2x1.
        Or use the fact that 2x2x2 has rank 7, so 2x2x3 >= 7, and we can do better than schoolbook 12.
        
        Let me try: C[:, 0:2] = A * B[:, 0:2] (rank 7 via Strassen)
                   C[:, 2:3] = A * B[:, 2:3] (rank 6 via schoolbook for 2x2x1)
        Total: 7 + 6 = 13. Not great.
        
        Alternatively, use a direct construction. Let me just use schoolbook for now: rank 12.
        """
        # For simplicity and correctness, use schoolbook: rank 12
        return schoolbook_decomp(2, 2, 3)
    
    def rank10_2x3x2():
        """
        2x3 * 3x2 = 2x2
        Schoolbook: rank 12. 
        Can we do better? Split A into two 2x2 blocks? No, A is 2x3.
        Let's use schoolbook: rank 12.
        """
        return schoolbook_decomp(2, 3, 2)
    
    def rank18_2x3x3():
        """
        2x3 * 3x3 = 2x3
        Schoolbook: rank 18.
        """
        return schoolbook_decomp(2, 3, 3)
    
    def rank9_2x2x2_v2():
        """
        Alternative: use rank 7 Strassen.
        """
        return strassen_2x2x2()
    
    # Now combine all blocks
    # Global dimensions: n=4, m=5, p=5
    # U: 4*5 = 20 rows
    # V: 5*5 = 25 rows  
    # W: 5*4 = 20 rows
    
    all_U = []
    all_V = []
    all_W = []
    
    # Block 1: C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2]
    # Term 1: A[0:2, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> C[0:2, 0:2] (2x2)
    U1, V1, W1, r1 = strassen_2x2x2()
    # Map to global: A rows 0-1, cols 0-1 -> U rows 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    #                 B rows 0-1, cols 0-1 -> V rows 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    #                 C rows 0-1, cols 0-1 -> W rows 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    
    U_map1 = [0, 1, 5, 6]  # a11, a12, a21, a22
    V_map1 = [0, 1, 5, 6]  # b11, b12, b21, b22
    W_map1 = [0, 1, 4, 5]  # c11, c12, c21, c22
    
    for c in range(r1):
        u_row = np.zeros(20)
        v_row = np.zeros(25)
        w_row = np.zeros(20)
        for local_i, local_j in [(0,0), (0,1), (1,0), (1,1)]:
            if U1[local_i*2+local_j, c] != 0:
                u_row[U_map1[local_i*2+local_j]] = U1[local_i*2+local_j, c]
        for local_j, local_k in [(0,0), (0,1), (1,0), (1,1)]:
            if V1[local_j*2+local_k, c] != 0:
                v_row[V_map1[local_j*2+local_k]] = V1[local_j*2+local_k, c]
        for local_k, local_i in [(0,0), (0,1), (1,0), (1,1)]:
            if W1[local_k*2+local_i, c] != 0:
                w_row[W_map1[local_k*2+local_i]] = W1[local_k*2+local_i, c]
        all_U.append(u_row)
        all_V.append(v_row)
        all_W.append(w_row)
    
    # Term 2: A[0:2, 2:5] (2x3) * B[2:5, 0:2] (3x2) -> C[0:2, 0:2] (2x2)
    U2, V2, W2, r2 = schoolbook_decomp(2, 3, 2)
    # A rows 0-1, cols 2-4 -> U rows 0*5+2=2, 0*5+3=3, 0*5+4=4, 1*5+2=7, 1*5+3=8, 1*5+4=9
    # B rows 2-4, cols 0-1 -> V rows 2*5+0=10, 2*5+1=11, 3*5+0=15, 3*5+1=16, 4*5+0=20, 4*5+1=21
    # C rows 0-1, cols 0-1 -> W rows 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    
    U_map2 = [2, 3, 4, 7, 8, 9]  # a11, a12, a13, a21, a22, a23
    V_map2 = [10, 11, 15, 16, 20, 21]  # b11, b12, b21, b22, b31, b32
    W_map2 = [0, 1, 4, 5]  # c11, c12, c21, c22
    
    for c in range(r2):
        u_row = np.zeros(20)
        v_row = np.zeros(25)
        w_row = np.zeros(20)
        for local_idx, val in enumerate(U2[:, c]):
            if val != 0:
                u_row[U_map2[local_idx]] = val
        for local_idx, val in enumerate(V2[:, c]):
            if val != 0:
                v_row[V_map2[local_idx]] = val
        for local_idx, val in enumerate(W2[:, c]):
            if val != 0:
                w_row[W_map2[local_idx]] = val
        all_U.append(u_row)
        all_V.append(v_row)
        all_W.append(w_row)
    
    # Block 2: C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:5]*B[2:5, 2:5]
    # Term 1: A[0:2, 0:2] (2x2) * B[0:2, 2:5] (2x3) -> C[0:2, 2:5] (2x3)
    U3, V3, W3, r3 = schoolbook_decomp(2, 2, 3)
    # A rows 0-1, cols 0-1 -> U rows 0, 1, 5, 6
    # B rows 0-1, cols 2-4 -> V rows 0*5+2=2, 0*5+3=3, 0*5+4=4, 1*5+2=7, 1*5+3=8, 1*5+4=9
    # C rows 0-1, cols 2-4 -> W rows 0*4+2=2, 0*4+3=3, 0*4+4=4, 1*4+2=6, 1*4+3=7, 1*4+4=8
    
    U_map3 = [0, 1, 5, 6]  # a11, a12, a21, a22
    V_map3 = [2, 3, 4, 7, 8, 9]  # b11, b12, b13, b21, b22, b23
    W_map3 = [2, 3, 4, 6, 7, 8]  # c11, c12, c13, c21, c22, c23
    
    for c in range(r3):
        u_row = np.zeros(20)
        v_row = np.zeros(25)
        w_row = np.zeros(20)
        for local_idx, val in enumerate(U3[:, c]):
            if val != 0:
                u_row[U_map3[local_idx]] = val
        for local_idx, val in enumerate(V3[:, c]):
            if val != 0:
                v_row[V_map3[local_idx]] = val
        for local_idx, val in enumerate(W3[:, c]):
            if val != 0:
                w_row[W_map3[local_idx]] = val
        all_U.append(u_row)
        all_V.append(v_row)
        all_W.append(w_row)
    
    # Term 2: A[0:2, 2:5] (2x3) * B[2:5, 2:5] (3x3) -> C[0:2, 2:5] (2x3)
    U4, V4, W4, r4 = schoolbook_decomp(2, 3, 3)
    # A rows 0-1, cols 2-4 -> U rows 2, 3, 4, 7, 8, 9
    # B rows 2-4, cols 2-4 -> V rows 2*5+2=12, 2*5+3=13, 2*5+4=14, 3*5+2=17, 3*5+3=18, 3*5+4=19, 4*5+2=22, 4*5+3=23, 4*5+4=24
    # C rows 0-1, cols 2-4 -> W rows 2, 3, 4, 6, 7, 8
    
    U_map4 = [2, 3, 4, 7, 8, 9]  # a11, a12, a13, a21, a22, a23
    V_map4 = [12, 13, 14, 17, 18, 19, 22, 23, 24]  # b11, b12, b13, b21, b22, b23, b31, b32, b33
    W_map4 = [2, 3, 4, 6, 7, 8]  # c11, c12, c13, c21, c22, c23
    
    for c in range(r4):
        u_row = np.zeros(20)
        v_row = np.zeros(25)
        w_row = np.zeros(20)
        for local_idx, val in enumerate(U4[:, c]):
            if val != 0:
                u_row[U_map4[local_idx]] = val
        for local_idx, val in enumerate(V4[:, c]):
            if val != 0:
                v_row[V_map4[local_idx]] = val
        for local_idx, val in enumerate(W4[:, c]):
            if val != 0:
                w_row[W_map4[local_idx]] = val
        all_U.append(u_row)
        all_V.append(v_row)
        all_W.append(w_row)
    
    # Block 3: C[2:4, 0:2] = A[2:4, 0:2]*B[0:2, 0:2] + A[2:4, 2:5]*B[2:5, 0:2]
    # Term 1: A[2:4, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> C[2:4, 0:2] (2x2)
    U5, V5, W5, r5 = strassen_2x2x2()
    # A rows 2-3, cols 0-1 -> U rows 2*5+0=10, 2*5+1=11, 3*5+0=15, 3*5+1=16
    # B rows 0-1, cols 0-1 -> V rows 0, 1, 5, 6
    # C rows 2-3, cols 0-1 -> W rows 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    
    U_map5 = [10, 11, 15, 16]  # a11, a12, a21, a22
    V_map5 = [0, 1, 5, 6]  # b11, b12, b21, b22
    W_map5 = [8, 9, 12, 13]  # c11, c12, c21, c22
    
    for c in range(r5):
        u_row = np.zeros(20)
        v_row = np.zeros(25)
        w_row = np.zeros(20)
        for local_i, local_j in [(0,0), (0,1), (1,0), (1,1)]:
            if U5[local_i*2+local_j, c] != 0:
                u_row[U_map5[local_i*2+local_j]] = U5[local_i*2+local_j, c]
        for local_j, local_k in [(0,0), (0,1), (1,0), (1,1)]:
            if V5[local_j*2+local_k, c] != 0:
                v_row[V_map5[local_j*2+local_k]] = V5[local_j*2+local_k, c]
        for local_k, local_i in [(0,0), (0,1), (1,0), (1,1)]:
            if W5[local_k*2+local_i, c] != 0:
                w_row[W_map5[local_k*2+local_i]] = W5[local_k*2+local_i, c]
        all_U.append(u_row)
        all_V.append(v_row)
        all_W.append(w_row)
    
    # Term 2: A[2:4, 2:5] (2x3) * B[2:5, 0:2] (3x2) -> C[2:4, 0:2] (2x2)
    U6, V6, W6, r6 = schoolbook_decomp(2, 3, 2)
    # A rows 2-3, cols 2-4 -> U rows 12, 13, 14, 17, 18, 19
    # B rows 2-4, cols 0-1 -> V rows 10, 11, 15, 16, 20, 21
    # C rows 2-3, cols 0-1 -> W rows 8, 9, 12, 13
    
    U_map6 = [12, 13, 14, 17, 18, 19]  # a11, a12, a13, a21, a22, a23
    V_map6 = [10, 11, 15, 16, 20, 21]  # b11, b12, b21, b22, b31, b32
    W_map6 = [8, 9, 12, 13]  # c11, c12, c21, c22
    
    for c in range(r6):
        u_row = np.zeros(20)
        v_row = np.zeros(25)
        w_row = np.zeros(20)
        for local_idx, val in enumerate(U6[:, c]):
            if val != 0:
                u_row[U_map6[local_idx]] = val
        for local_idx, val in enumerate(V6[:, c]):
            if val != 0:
                v_row[V_map6[local_idx]] = val
        for local_idx, val in enumerate(W6[:, c]):
            if val != 0:
                w_row[W_map6[local_idx]] = val
        all_U.append(u_row)
        all_V.append(v_row)
        all_W.append(w_row)
    
    # Block 4: C[2:4, 2:5] = A[2:4, 0:2]*B[0:2, 2:5] + A[2:4, 2:5]*B[2:5, 2:5]
    # Term 1: A[2:4, 0:2] (2x2) * B[0:2, 2:5] (2x3) -> C[2:4, 2:5] (2x3)
    U7, V7, W7, r7 = schoolbook_decomp(2, 2, 3)
    # A rows 2-3, cols 0-1 -> U rows 10, 11, 15, 16
    # B rows 0-1, cols 2-4 -> V rows 2, 3, 4, 7, 8, 9
    # C rows 2-3, cols 2-4 -> W rows 10, 11, 12, 14, 15, 16
    
    U_map7 = [10, 11, 15, 16]  # a11, a12, a21, a22
    V_map7 = [2, 3, 4, 7, 8, 9]  # b11, b12, b13, b21, b22, b23
    W_map7 = [10, 11, 12, 14, 15, 16]  # c11, c12, c13, c21, c22, c23
    
    for c in range(r7):
        u_row = np.zeros(20)
        v_row = np.zeros(25)
        w_row = np.zeros(20)
        for local_idx, val in enumerate(U7[:, c]):
            if val != 0:
                u_row[U_map7[local_idx]] = val
        for local_idx, val in enumerate(V7[:, c]):
            if val != 0:
                v_row[V_map7[local_idx]] = val
        for local_idx, val in enumerate(W7[:, c]):
            if val != 0:
                w_row[W_map7[local_idx]] = val
        all_U.append(u_row)
        all_V.append(v_row)
        all_W.append(w_row)
    
    # Term 2: A[2:4, 2:5] (2x3) * B[2:5, 2:5] (3x3) -> C[2:4, 2:5] (2x3)
    U8, V8, W8, r8 = schoolbook_decomp(2, 3, 3)
    # A rows 2-3, cols 2-4 -> U rows 12, 13, 14, 17, 18, 19
    # B rows 2-4, cols 2-4 -> V rows 12, 13, 14, 17, 18, 19, 22, 23, 24
    # C rows 2-3, cols 2-4 -> W rows 10, 11, 12, 14, 15, 16
    
    U_map8 = [12, 13, 14, 17, 18, 19]  # a11, a12, a13, a21, a22, a23
    V_map8 = [12, 13, 14, 17, 18, 19, 22, 23, 24]  # b11, b12, b13, b21, b22, b23, b31, b32, b33
    W_map8 = [10, 11, 12, 14, 15, 16]  # c11, c12, c13, c21, c22, c23
    
    for c in range(r8):
        u_row = np.zeros(20)
        v_row = np.zeros(25)
        w_row = np.zeros(20)
        for local_idx, val in enumerate(U8[:, c]):
            if val != 0:
                u_row[U_map8[local_idx]] = val
        for local_idx, val in enumerate(V8[:, c]):
            if val != 0:
                v_row[V_map8[local_idx]] = val
        for local_idx, val in enumerate(W8[:, c]):
            if val != 0:
                w_row[W_map8[local_idx]] = val
        all_U.append(u_row)
        all_V.append(v_row)
        all_W.append(w_row)
    
    # Assemble final matrices
    U_final = np.array(all_U).T  # rank x 20
    V_final = np.array(all_V).T  # rank x 25
    W_final = np.array(all_W).T  # rank x 20
    
    # Convert to list of lists with integer entries
    U_list = [[int(x) for x in row] for row in U_final.T]
    V_list = [[int(x) for x in row] for row in V_final.T]
    W_list = [[int(x) for x in row] for row in W_final.T]
    
    rank = len(U_list)
    
    # Verify correctness
    def verify(U_l, V_l, W_l):
        n, m, p = 4, 5, 5
        rank = len(U_l)
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give delta structure
        # Actually, the tensor decomposition means:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T[i,j,k] = 1 if this corresponds to a valid multiplication term, else 0
        # For matrix mult: (AB)[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
        
        # Let's verify by computing the implied matrix multiplication
        # and comparing with standard matmul on random inputs
        random.seed(42)
        A = np.random.randint(-10, 10, size=(n, m)).astype(float)
        B = np.random.randint(-10, 10, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        U_arr = np.array(U_l)
        V_arr = np.array(V_l)
        W_arr = np.array(W_l)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # This should equal 1 (since matrix mult tensor has all 1s)
                    # Actually no - the tensor T[i,j,k] for matrix mult is 1 for all i,j,k
                    # Let me just compute C[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] * A[i,j]*B[j,k]
                    # That's not right either.
                    
                    # The correct check: 
                    # C[i,k] = sum_j A[i,j] * B[j,k]
                    # And from decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k]) ... no
                    # 
                    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
                    # Tensor: T[i,j,k] such that C[i,k] = sum_{j,k'} T[i,j,k'] A[i,j] B[j',k]
                    # For matrix mult, T[i,j,k] = delta[j,k] ... no
                    # 
                    # Standard form: (AB)[i,k] = sum_j A[i,j] B[j,k]
                    # So the tensor has T[i,j,k] = 1 if j==k' ... I'm confusing myself.
                    # 
                    # Let me just use the definition: 
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should reconstruct the identity
                    # for the multiplication tensor.
                    
                    # Simplest verification: build the full tensor and check
                    pass
        
        # Alternative verification: 
        # The decomposition satisfies: for all i,j,k,
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # (since the matrix multiplication tensor is all ones)
        
        U_arr = np.array(U_l)
        V_arr = np.array(V_l) 
        W_arr = np.array(W_l)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    is_valid = verify(U_list, V_list, W_list)
    if not is_valid:
        # Fall back to schoolbook
        U_sch = [[0] * 100 for _ in range(20)]
        V_sch = [[0] * 100 for _ in range(25)]
        W_sch = [[0] * 100 for _ in range(20)]
        c = 0
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    U_sch[i*5+j][c] = 1
                    V_sch[j*5+k][c] = 1
                    W_sch[k*4+i][c] = 1
                    c += 1
        return {"U": U_sch, "V": V_sch, "W": W_sch}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
