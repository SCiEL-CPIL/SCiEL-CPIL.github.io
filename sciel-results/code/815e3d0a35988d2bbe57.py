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
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        W[0][0] = 1  # C11 (k=0,i=0 -> k*2+i=0)
        W[3][0] = 1  # C22 (k=1,i=1 -> k*2+i=3)
        
        # r=1: M2 = (A21+A22)*B11
        U[2][1] = 1  # A21
        U[3][1] = 1  # A22
        V[0][1] = 1  # B11
        W[2][1] = 1  # C21 (k=1,i=0 -> k*2+i=2)
        
        # r=2: M3 = A11*(B12-B22)
        U[0][2] = 1  # A11
        V[1][2] = 1  # B12
        V[3][2] = -1  # B22
        W[1][2] = 1  # C12 (k=0,i=1 -> k*2+i=1)
        
        # r=3: M4 = A22*(B21-B11)
        U[3][3] = 1  # A22
        V[2][3] = 1  # B21
        V[0][3] = -1  # B11
        W[0][3] = 1  # C11
        W[2][3] = 1  # C21
        
        # r=4: M5 = (A11+A12)*B22
        U[0][4] = 1  # A11
        U[1][4] = 1  # A12
        V[3][4] = 1  # B22
        W[0][4] = -1  # C11
        W[1][4] = 1  # C12
        
        # r=5: M6 = (A21-A11)*(B11+B12)
        U[2][5] = 1  # A21
        U[0][5] = -1  # A11
        V[0][5] = 1  # B11
        V[1][5] = 1  # B12
        W[3][5] = 1  # C22
        
        # r=6: M7 = (A12-A22)*(B21+B22)
        U[1][6] = 1  # A12
        U[3][6] = -1  # A22
        V[2][6] = 1  # B21
        V[3][6] = 1  # B22
        W[0][6] = 1  # C11
        W[3][6] = 1  # C22
        
        return U, V, W
    
    def test_strassen():
        """Test Strassen 2x2 decomposition."""
        U, V, W = strassen_2x2()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def build_strassen_block():
        """Build 4x4*4x5 using Strassen for 2x2*2x2 and schoolbook for 2x2*2x3."""
        U_list = []
        V_list = []
        W_list = []
        
        sU, sV, sW = strassen_2x2()
        
        def add_strassen(a_map, b_map, c_map):
            """Add a Strassen-based 2x2*2x2 block multiplication."""
            for r in range(7):
                U_new = [0]*16
                V_new = [0]*20
                W_new = [0]*20
                for si, val in enumerate(sU):
                    if val != 0:
                        U_new[a_map[si]] = val
                for sj, val in enumerate(sV):
                    if val != 0:
                        V_new[b_map[sj]] = val
                for sk, val in enumerate(sW):
                    if val != 0:
                        W_new[c_map[sk]] = val
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
        
        def add_schoolbook_2x2x2x3(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            """Add schoolbook terms for 2x2 * 2x3 block multiplication.
            a_row_start, a_col_start: starting row/col of A block in full A (4x4)
            b_row_start, b_col_start: starting row/col of B block in full B (4x5)
            c_row_start, c_col_start: starting row/col of C block in full C (4x5)
            """
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(a_row_start+i)*4 + (a_col_start+j)] = 1
                        V_new[(b_row_start+j)*5 + (b_col_start+k)] = 1
                        W_new[(c_col_start+k)*4 + (c_row_start+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)
        
        # Block (1,1): C11 += A11*B11  (2x2 * 2x2 -> 2x2)
        # A11: rows 0-1, cols 0-1
        # B11: rows 0-1, cols 0-1
        # C11: rows 0-1, cols 0-1
        a_map1 = [0, 1, 4, 5]      # A11[0,0], A11[0,1], A11[1,0], A11[1,1]
        b_map1 = [0, 1, 5, 6]      # B11[0,0], B11[0,1], B11[1,0], B11[1,1]
        c_map1 = [0, 4, 1, 5]      # C11[0,0], C11[1,0], C11[0,1], C11[1,1]
        # W[k*4+i]: C11[0,0]=W[0], C11[1,0]=W[4], C11[0,1]=W[1], C11[1,1]=W[5]
        # Wait: W index is k*n+i where k is col, i is row
        # C11[i,k] local -> full W[k*4+i]
        # C11[0,0] -> W[0*4+0] = W[0]
        # C11[0,1] -> W[1*4+0] = W[4]
        # C11[1,0] -> W[0*4+1] = W[1]
        # C11[1,1] -> W[1*4+1] = W[5]
        # Strassen W indices: 0=C[0,0], 1=C[1,0], 2=C[0,1], 3=C[1,1]
        # So: W_s[0] -> W[0], W_s[1] -> W[1], W_s[2] -> W[4], W_s[3] -> W[5]
        c_map1 = [0, 1, 4, 5]
        add_strassen(a_map1, b_map1, c_map1)
        
        # Block (1,2): C12 += A11*B12  (2x2 * 2x3 -> 2x3)
        # A11: rows 0-1, cols 0-1
        # B12: rows 0-1, cols 2-4
        # C12: rows 0-1, cols 2-4
        add_schoolbook_2x2x2x3(0, 0, 0, 2, 0, 2)
        
        # Block (2,1): C21 += A21*B11  (2x2 * 2x2 -> 2x2)
        # A21: rows 2-3, cols 0-1
        # B11: rows 0-1, cols 0-1
        # C21: rows 2-3, cols 0-1
        a_map2 = [8, 9, 12, 13]    # A21[0,0]=U[8], A21[0,1]=U[9], A21[1,0]=U[12], A21[1,1]=U[13]
        b_map2 = [0, 1, 5, 6]
        # C21[i,k] local -> full W[k*4+(2+i)]
        # C21[0,0] -> W[0*4+2] = W[2]
        # C21[0,1] -> W[1*4+2] = W[6]
        # C21[1,0] -> W[0*4+3] = W[3]
        # C21[1,1] -> W[1*4+3] = W[7]
        # Strassen W: 0->W[2], 1->W[3], 2->W[6], 3->W[7]
        c_map2 = [2, 3, 6, 7]
        add_strassen(a_map2, b_map2, c_map2)
        
        # Block (2,2): C22 += A21*B12  (2x2 * 2x3 -> 2x3)
        # A21: rows 2-3, cols 0-1
        # B12: rows 0-1, cols 2-4
        # C22: rows 2-3, cols 2-4
        add_schoolbook_2x2x2x3(2, 0, 0, 2, 2, 2)
        
        # Block (3,1): C11 += A12*B21  (2x2 * 2x2 -> 2x2)
        # A12: rows 0-1, cols 2-3
        # B21: rows 2-3, cols 0-1
        # C11: rows 0-1, cols 0-1
        a_map3 = [2, 3, 6, 7]      # A12[0,0]=U[2], A12[0,1]=U[3], A12[1,0]=U[6], A12[1,1]=U[7]
        b_map3 = [10, 11, 15, 16]  # B21[0,0]=V[10], B21[0,1]=V[11], B21[1,0]=V[15], B21[1,1]=V[16]
        c_map3 = [0, 1, 4, 5]      # Same as C11
        add_strassen(a_map3, b_map3, c_map3)
        
        # Block (3,2): C12 += A12*B22  (2x2 * 2x3 -> 2x3)
        # A12: rows 0-1, cols 2-3
        # B22: rows 2-3, cols 2-4
        # C12: rows 0-1, cols 2-4
        add_schoolbook_2x2x2x3(0, 2, 2, 2, 0, 2)
        
        # Block (4,1): C21 += A22*B21  (2x2 * 2x2 -> 2x2)
        # A22: rows 2-3, cols 2-3
        # B21: rows 2-3, cols 0-1
        # C21: rows 2-3, cols 0-1
        a_map4 = [10, 11, 14, 15]  # A22[0,0]=U[10], A22[0,1]=U[11], A22[1,0]=U[14], A22[1,1]=U[15]
        b_map4 = [10, 11, 15, 16]
        c_map4 = [2, 3, 6, 7]      # Same as C21
        add_strassen(a_map4, b_map4, c_map4)
        
        # Block (4,2): C22 += A22*B22  (2x2 * 2x3 -> 2x3)
        # A22: rows 2-3, cols 2-3
        # B22: rows 2-3, cols 2-4
        # C22: rows 2-3, cols 2-4
        add_schoolbook_2x2x2x3(2, 2, 2, 2, 2, 2)
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    if test_strassen():
        strassen_result = build_strassen_block()
        rank = len(strassen_result["U"])
        if verify(strassen_result["U"], strassen_result["V"], strassen_result["W"]):
            if rank < best_rank:
                best = strassen_result
                best_rank = rank
    
    # Try to merge identical terms to reduce rank
    def merge_terms(U, V, W):
        """Merge identical (U,V,W) triples by combining coefficients."""
        term_map = defaultdict(int)
        for r in range(len(U[0])):
            u_tuple = tuple(U[i][r] for i in range(16))
            v_tuple = tuple(V[i][r] for i in range(20))
            w_tuple = tuple(W[i][r] for i in range(20))
            key = (u_tuple, v_tuple, w_tuple)
            term_map[key] += 1
        
        U_merged = []
        V_merged = []
        W_merged = []
        for (u_t, v_t, w_t), count in term_map.items():
            if count == 0:
                continue
            # Check if all are zero
            if all(x == 0 for x in u_t) and all(x == 0 for x in v_t) and all(x == 0 for x in w_t):
                continue
            # Put the count multiplier in the first non-zero entry of U
            u_list = list(u_t)
            placed = False
            for i in range(len(u_list)):
                if u_list[i] != 0:
                    u_list[i] *= count
                    placed = True
                    break
            if not placed:
                # Try V
                v_list = list(v_t)
                for i in range(len(v_list)):
                    if v_list[i] != 0:
                        v_list[i] *= count
                        placed = True
                        break
                if placed:
                    u_list = list(u_t)
                else:
                    # Try W
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
