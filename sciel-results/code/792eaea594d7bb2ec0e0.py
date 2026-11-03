"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    def strassen_2x2_terms():
        """Return list of (U_vec, V_vec, W_vec) for 2x2x2 Strassen.
        Each vec has length 4.
        U index: i*2+j, V index: j*2+k, W index: k*2+i
        """
        terms = []
        # M1 = (a+d)(e+h): C00+=M1, C11+=M1
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # M2 = (c+d)e: C10+=M2, C11-=M2
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        # M3 = a(f-h): C01+=M3, C11+=M3
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
        # M4 = d(g-e): C00+=M4, C10+=M4
        terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
        # M5 = (a+b)h: C00-=M5, C01+=M5
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        # M6 = (c-a)(e+f): C11+=M6
        terms.append(([ -1,0,1,0], [1,1,0,0], [0,0,0,1]))
        # M7 = (b-d)(g+h): C00+=M7
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        return terms
    
    def strassen_4x4_terms():
        """Return list of (U_vec, V_vec, W_vec) for 4x4x4 Strassen (rank 49)."""
        s2 = strassen_2x2_terms()
        
        outer_strassen = [
            {'a': [(1,(0,0)), (1,(1,1))], 'b': [(1,(0,0)), (1,(1,1))], 'c': [(1,(0,0)), (1,(1,1))]},
            {'a': [(1,(1,0)), (1,(1,1))], 'b': [(1,(0,0))], 'c': [(1,(1,0)), (-1,(1,1))]},
            {'a': [(1,(0,0))], 'b': [(1,(0,1)), (-1,(1,1))], 'c': [(1,(0,1)), (1,(1,1))]},
            {'a': [(1,(1,1))], 'b': [(1,(1,0)), (-1,(0,0))], 'c': [(1,(0,0)), (1,(1,0))]},
            {'a': [(1,(0,0)), (1,(0,1))], 'b': [(1,(1,1))], 'c': [(-1,(0,0)), (1,(0,1))]},
            {'a': [(1,(1,0)), (-1,(0,0))], 'b': [(1,(0,0)), (1,(0,1))], 'c': [(1,(1,1))]},
            {'a': [(1,(0,1)), (-1,(1,1))], 'b': [(1,(1,0)), (1,(1,1))], 'c': [(1,(0,0))]},
        ]
        
        all_terms = []
        for outer in outer_strassen:
            for inner in s2:
                U_4 = [0]*16
                V_4 = [0]*16
                W_4 = [0]*16
                
                for coeff_a, (bi, bj) in outer['a']:
                    for ii in range(2):
                        for jj in range(2):
                            idx_u = (bi*2+ii)*4 + (bj*2+jj)
                            idx_2x2 = ii*2+jj
                            U_4[idx_u] += coeff_a * inner[0][idx_2x2]
                
                for coeff_b, (bj, bk) in outer['b']:
                    for jj in range(2):
                        for kk in range(2):
                            idx_v = (bj*2+jj)*4 + (bk*2+kk)
                            idx_2x2 = jj*2+kk
                            V_4[idx_v] += coeff_b * inner[1][idx_2x2]
                
                for coeff_c, (bi, bk) in outer['c']:
                    for ii in range(2):
                        for kk in range(2):
                            idx_w = (bk*2+kk)*4 + (bi*2+ii)
                            idx_2x2 = kk*2+ii
                            W_4[idx_w] += coeff_c * inner[2][idx_2x2]
                
                all_terms.append((U_4, V_4, W_4))
        
        return all_terms
    
    def schoolbook_4x4x1_terms():
        """Return list of (U_vec, V_vec, W_vec) for 4x4 x 4x1 schoolbook (rank 16)."""
        terms = []
        for i in range(4):
            for j in range(4):
                U = [0]*16
                V = [0]*4
                W = [0]*4
                U[i*4+j] = 1
                V[j] = 1
                W[i] = 1
                terms.append((U, V, W))
        return terms
    
    s4_terms = strassen_4x4_terms()
    s4b_terms = strassen_4x4_terms()
    sb_terms = schoolbook_4x4x1_terms()
    
    total_rank = len(s4_terms) + len(s4b_terms) + len(sb_terms)
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    col = 0
    for U_t, V_t, W_t in s4_terms:
        for i in range(16):
            if U_t[i] != 0:
                U[i][col] = U_t[i]
        for j in range(16):
            if V_t[j] != 0:
                V[j][col] = V_t[j]
        for k in range(16):
            if W_t[k] != 0:
                W[k][col] = W_t[k]
        col += 1
    
    for U_t, V_t, W_t in s4b_terms:
        for i in range(16):
            if U_t[i] != 0:
                U[i][col] = U_t[i]
        for j in range(16):
            if V_t[j] != 0:
                V[j+16][col] = V_t[j]
        for k in range(16):
            if W_t[k] != 0:
                W[k+16][col] = W_t[k]
        col += 1
    
    for U_t, V_t, W_t in sb_terms:
        for i in range(16):
            if U_t[i] != 0:
                U[i][col] = U_t[i]
        for j in range(4):
            if V_t[j] != 0:
                V[j+32][col] = V_t[j]
        for k in range(4):
            if W_t[k] != 0:
                W[k+32][col] = W_t[k]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
