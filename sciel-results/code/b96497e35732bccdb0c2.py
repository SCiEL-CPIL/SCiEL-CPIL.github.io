"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 20 x rank  
    # W: (p*n) x rank = 20 x rank
    # Convention: W[k*n+i, r]
    
    start_time = time.time()
    deadline = 145  # 5s safety margin before 150s kill
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        if not U or not V or not W:
            return False
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p  # 80
    
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # r=0: M1 = (A11+A22)*(B11+B22)
        U[0][0] = 1
        U[3][0] = 1
        V[0][0] = 1
        V[3][0] = 1
        W[0][0] = 1
        W[3][0] = 1
        
        # r=1: M2 = (A21+A22)*B11
        U[2][1] = 1
        U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # r=2: M3 = A11*(B12-B22)
        U[0][2] = 1
        V[1][2] = 1
        V[3][2] = -1
        W[1][2] = 1
        
        # r=3: M4 = A22*(B21-B11)
        U[3][3] = 1
        V[2][3] = 1
        V[0][3] = -1
        W[0][3] = 1
        W[2][3] = 1
        
        # r=4: M5 = (A11+A12)*B22
        U[0][4] = 1
        U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1
        W[1][4] = 1
        
        # r=5: M6 = (A21-A11)*(B11+B12)
        U[2][5] = 1
        U[0][5] = -1
        V[0][5] = 1
        V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (A12-A22)*(B21+B22)
        U[1][6] = 1
        U[3][6] = -1
        V[2][6] = 1
        V[3][6] = 1
        W[0][6] = 1
        W[3][6] = 1
        
        return U, V, W
    
    def strassen_2x2_2x3():
        """Return U, V, W for 2x2 * 2x3 -> 2x3 using Strassen-style.
        U: 4x9, V: 6x9, W: 6x9
        Convention: U[i*2+j, r], V[j*3+k, r], W[k*2+i, r]
        Uses 3 Strassen 2x2*2x2 blocks sharing some structure.
        """
        U = [[0]*9 for _ in range(4)]
        V = [[0]*9 for _ in range(6)]
        W = [[0]*9 for _ in range(6)]
        
        # We decompose 2x2 * 2x3 as three 2x2*2x2 blocks:
        # C[:,0:2] = A * B[:,0:2] and C[:,2:3] = A * B[:,2:3]
        # But we can share some computations.
        # 
        # Alternative: use the fact that 2x2*2x3 can be done with rank 9
        # by using Strassen for the first 2 columns and schoolbook for the 3rd,
        # but sharing some of the A-side computations.
        #
        # Let's use a direct construction:
        # M1 = (A11+A22)*(B11+B21) -> contributes to C10, C20
        # M2 = (A21+A22)*B11 -> contributes to C20
        # M3 = A11*(B12-B21) -> contributes to C11
        # M4 = A22*(B20-B10) -> contributes to C10, C20
        # M5 = (A11+A12)*B21 -> contributes to C10, C11
        # M6 = (A21-A11)*(B10+B11) -> contributes to C21
        # M7 = (A12-A22)*(B20+B21) -> contributes to C10, C11
        # M8 = A11*B12 -> contributes to C12
        # M9 = A22*B22 -> contributes to C22
        # Wait, this isn't quite right. Let me use a cleaner approach.
        
        # Actually, let's just use 3 separate Strassen decompositions for 3 2x2*2x2 blocks
        # But that gives rank 21, which is worse.
        
        # Better: use the following 9-term decomposition:
        # For 2x2 * 2x3 -> 2x3:
        # Let A = [[a11, a12], [a21, a22]], B = [[b10, b11, b12], [b20, b21, b22]]
        # C = [[c10, c11, c12], [c20, c21, c22]]
        
        # r=0: (a11+a22)*(b10+b20) -> c10 += 1, c20 += 1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[2][0] = 1  # W[k*2+i]: C[0,0]=W[0], C[1,0]=W[2]
        
        # r=1: (a21+a22)*b10 -> c20 += 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # r=2: a11*(b11-b20) -> c11 += 1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1  # C[0,1]=W[1]
        
        # r=3: a22*(b20-b10) -> c10 += 1, c20 += 1
        U[3][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # r=4: (a11+a12)*b20 -> c10 += -1, c11 += 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # r=5: (a21-a11)*(b10+b11) -> c21 += 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # C[1,1]=W[3]
        
        # r=6: (a12-a22)*(b20+b21) -> c10 += 1, c11 += 1
        U[1][6] = 1; U[3][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1; W[1][6] = 1
        
        # r=7: a11*b12 -> c12 += 1
        U[0][7] = 1
        V[2][7] = 1  # V[j*3+k]: B[0,2]=V[2]
        W[4][7] = 1  # C[0,2]=W[4]
        
        # r=8: a22*b22 -> c22 += 1
        U[3][8] = 1
        V[5][8] = 1  # B[1,2]=V[5]
        W[5][8] = 1  # C[1,2]=W[5]
        
        return U, V, W
    
    def verify_small(U, V, W, n_s, m_s, p_s):
        """Verify a small decomposition."""
        rank = len(U[0]) if U else 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    s = 0
                    for r in range(rank):
                        s += U[i*m_s+j][r] * V[j*p_s+k][r] * W[k*n_s+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def build_recursive():
        """Build 4x4*4x5 using recursive tensor product of 2x2 and 2x3 blocks."""
        U_list = []
        V_list = []
        W_list = []
        
        sU, sV, sW = strassen_2x2()
        tU, tV, tW = strassen_2x2_2x3()
        
        def add_block(u_block, v_block, w_block, a_row, a_col, b_row, b_col, c_row, c_col, n_block, m_block, p_block):
            """Add a block decomposition to the global lists.
            a_row, a_col: top-left of A block in full A (4x4)
            b_row, b_col: top-left of B block in full B (4x5)
            c_row, c_col: top-left of C block in full C (4x5)
            n_block, m_block, p_block: dimensions of the block
            """
            rank = len(u_block[0])
            for r in range(rank):
                U_new = [0]*16
                V_new = [0]*20
                W_new = [0]*20
                for si in range(n_block * m_block):
                    val = u_block[si][r]
                    if val != 0:
                        # si is local index in A block: local_i * m_block + local_j
                        local_i = si // m_block
                        local_j = si % m_block
                        global_i = a_row + local_i
                        global_j = a_col + local_j
                        U_new[global_i * 4 + global_j] = val
                for sj in range(m_block * p_block):
                    val = v_block[sj][r]
                    if val != 0:
                        local_j = sj // p_block
                        local_k = sj % p_block
                        global_j = b_row + local_j
                        global_k = b_col + local_k
                        V_new[global_j * 5 + global_k] = val
                for sk in range(p_block * n_block):
                    val = w_block[sk][r]
                    if val != 0:
                        # sk is local index in C block: local_k * n_block + local_i
                        local_k = sk // n_block
                        local_i = sk % n_block
                        global_k = c_col + local_k
                        global_i = c_row + local_i
                        W_new[global_k * 4 + global_i] = val
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
        
        # 4x4 * 4x5 decomposed into 2x2 blocks:
        # A is 4x4, B is 4x5, C is 4x5
        # Split into 2x2 blocks:
        # A11 (2x2), A12 (2x2), A21 (2x2), A22 (2x2)
        # B11 (2x2), B12 (2x3), B21 (2x2), B22 (2x3)
        # C11 (2x2), C12 (2x3), C21 (2x2), C22 (2x3)
        
        # Block (1,1): C11 += A11*B11 (2x2 * 2x2 -> 2x2)
        add_block(sU, sV, sW, 0, 0, 0, 0, 0, 0, 2, 2, 2)
        
        # Block (1,2): C12 += A11*B12 (2x2 * 2x3 -> 2x3)
        add_block(tU, tV, tW, 0, 0, 0, 2, 0, 2, 2, 2, 3)
        
        # Block (2,1): C21 += A21*B11 (2x2 * 2x2 -> 2x2)
        add_block(sU, sV, sW, 2, 0, 0, 0, 2, 0, 2, 2, 2)
        
        # Block (2,2): C22 += A21*B12 (2x2 * 2x3 -> 2x3)
        add_block(tU, tV, tW, 2, 0, 0, 2, 2, 2, 2, 2, 3)
        
        # Block (3,1): C11 += A12*B21 (2x2 * 2x2 -> 2x2)
        add_block(sU, sV, sW, 0, 2, 2, 0, 0, 0, 2, 2, 2)
        
        # Block (3,2): C12 += A12*B22 (2x2 * 2x3 -> 2x3)
        add_block(tU, tV, tW, 0, 2, 2, 2, 0, 2, 2, 2, 3)
        
        # Block (4,1): C21 += A22*B21 (2x2 * 2x2 -> 2x2)
        add_block(sU, sV, sW, 2, 2, 2, 0, 2, 0, 2, 2, 2)
        
        # Block (4,2): C22 += A22*B22 (2x2 * 2x3 -> 2x3)
        add_block(tU, tV, tW, 2, 2, 2, 2, 2, 2, 2, 2, 3)
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Verify the small decompositions
    if verify_small(*strassen_2x2(), 2, 2, 2):
        if verify_small(*strassen_2x2_2x3(), 2, 2, 3):
            recursive_result = build_recursive()
            rank = len(recursive_result["U"])
            if verify(recursive_result["U"], recursive_result["V"], recursive_result["W"]):
                if rank < best_rank:
                    best = recursive_result
                    best_rank = rank
        else:
            # Fallback: use schoolbook for 2x2*2x3
            def schoolbook_2x2_2x3():
                U = [[0]*6 for _ in range(4)]
                V = [[0]*6 for _ in range(6)]
                W = [[0]*6 for _ in range(6)]
                col = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            U[i*2+j][col] = 1
                            V[j*3+k][col] = 1
                            W[k*2+i][col] = 1
                            col += 1
                return U, V, W
            
            tU, tV, tW = schoolbook_2x2_2x3()
            recursive_result = build_recursive()
            rank = len(recursive_result["U"])
            if verify(recursive_result["U"], recursive_result["V"], recursive_result["W"]):
                if rank < best_rank:
                    best = recursive_result
                    best_rank = rank
    else:
        # Strassen 2x2 is wrong, use schoolbook for everything
        pass
    
    # Try to merge identical terms to reduce rank
    def merge_terms(U, V, W):
        """Merge identical (U,V,W) triples by combining coefficients."""
        if not U or not V or not W:
            return {"U": U, "V": V, "W": W}
        term_map = defaultdict(lambda: [0, 0, 0])
        rank = len(U[0])
        for r in range(rank):
            u_tuple = tuple(U[i][r] for i in range(16))
            v_tuple = tuple(V[i][r] for i in range(20))
            w_tuple = tuple(W[i][r] for i in range(20))
            key = (u_tuple, v_tuple, w_tuple)
            term_map[key][0] += 1
        
        U_merged = []
        V_merged = []
        W_merged = []
        for (u_t, v_t, w_t), count in term_map.items():
            if count == 0:
                continue
            if all(x == 0 for x in u_t) and all(x == 0 for x in v_t) and all(x == 0 for x in w_t):
                continue
            u_list = list(u_t)
            placed = False
            for i in range(len(u_list)):
                if u_list[i] != 0:
                    u_list[i] *= count
                    placed = True
                    break
            if not placed:
                v_list = list(v_t)
                for i in range(len(v_list)):
                    if v_list[i] != 0:
                        v_list[i] *= count
                        placed = True
                        break
                if placed:
                    u_list = list(u_t)
                else:
                    w_list = list(w_t)
                    for i in range(len(w_list)):
                        if w_list[i] != 0:
                            w_list[i] *= count
                            break
                    else:
                        continue
            else:
                v_list = list(v_t)
                w_list = list(w_t)
            
            U_merged.append(u_list)
            V_merged.append(v_list)
            W_merged.append(w_list)
        
        return {"U": U_merged, "V": V_merged, "W": W_merged}
    
    if best_rank > 0:
        merged = merge_terms(best["U"], best["V"], best["W"])
        if merged["U"] and verify(merged["U"], merged["V"], merged["W"]):
            if len(merged["U"]) < best_rank:
                best = merged
                best_rank = len(merged["U"])
    
    return best
    
# EVOLVE-BLOCK-END
