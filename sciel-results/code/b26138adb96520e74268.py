"""Matrix multiplication 4,4,8 rank-minimal search."""
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_factors():
    """Returns U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Indices:
    # A: i=0,1; j=0,1 -> a = i*2 + j
    # B: j=0,1; k=0,1 -> b = j*2 + k
    # C: k=0,1; i=0,1 -> c = k*2 + i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0 = (A00+A11)(B00+B11)
    # A00: a=0, A11: a=3
    # B00: b=0, B11: b=3
    # C00: c=1 (k=0,i=1? No c=k*2+i. C00 is i=0,k=0 -> c=0. C11 is i=1,k=1 -> c=3)
    # Wait, C indices: c = k*n + i. n=2.
    # C[i,k] is at index c = k*2 + i.
    # C00 (i=0,k=0) -> c=0.
    # C01 (i=0,k=1) -> c=2.
    # C10 (i=1,k=0) -> c=1.
    # C11 (i=1,k=1) -> c=3.
    
    # M0 contributes to C00 (+1) and C11 (+1).
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M1 = (A10+A11)B00
    # A10: a=2, A11: a=3
    # B00: b=0
    # C10 (+1), C11 (-1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M2 = A00(B01-B11)
    # A00: a=0
    # B01: b=1, B11: b=3
    # C01 (+1), C11 (+1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M3 = A11(B10-B00)
    # A11: a=3
    # B10: b=2, B00: b=0
    # C00 (+1), C10 (+1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M4 = (A00+A01)B11
    # A00: a=0, A01: a=1
    # B11: b=3
    # C00 (-1), C01 (+1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M5 = (A10-A00)(B00+B01)
    # A10: a=2, A00: a=0
    # B00: b=0, B01: b=1
    # C11 (+1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M6 = (A01-A11)(B10+B11)
    # A01: a=1, A11: a=3
    # B10: b=2, B11: b=3
    # C00 (+1), C11 (+1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _construct_4x4x4_strassen():
    """Constructs rank-49 decomposition for 4x4x4 using recursive Strassen."""
    n = 4
    m = 4
    p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    base_U, base_V, base_W = _strassen_2x2_factors()
    
    # The 7 Strassen terms for 4x4 (blocks of size 2)
    # Each term r corresponds to a 2x2x2 multiplication of the blocks.
    # We need to map the block indices to global indices.
    
    # Block (I, J) of A is at global row i = 2*I + i', col j = 2*J + j'.
    # Block (J, K) of B is at global row j = 2*J + j', col k = 2*K + k'.
    # Block (K, I) of C is at global row k = 2*K + k', col i = 2*I + i'.
    
    # Term r defines:
    # U_r: linear comb of A blocks
    # V_r: linear comb of B blocks
    # W_r: linear comb of C blocks (output)
    
    # We iterate over the 7 terms.
    # For each term, we have a 2x2x2 problem on the blocks.
    # The factors for this 2x2x2 problem are base_U, base_V, base_W.
    # We need to "expand" these into the 4x4x4 factors.
    
    for r in range(7):
        # Determine the block coefficients for U_r and V_r
        # Based on the standard Strassen formulas for 2x2 blocks:
        
        # U_r blocks (coeffs for A_IJ)
        # V_r blocks (coeffs for B_JK)
        # W_r blocks (coeffs for C_KI)
        
        # We define the 2x2 matrices of coefficients for U_r, V_r, W_r
        # Note: The base_U/V/W are for the *scalar* entries of the 2x2 block matrices.
        # So if U_r = c00*A00 + c01*A01 + c10*A10 + c11*A11,
        # then for each scalar entry (i', j') in the block, the coefficient is cIJ.
        
        u_coeff = [[0,0],[0,0]]
        v_coeff = [[0,0],[0,0]]
        w_coeff = [[0,0],[0,0]]
        
        if r == 0:
            u_coeff = [[1,0],[0,1]]
            v_coeff = [[1,0],[0,1]]
            w_coeff = [[1,0],[0,1]] # C00 += M0, C11 += M0
        elif r == 1:
            u_coeff = [[0,0],[1,1]]
            v_coeff = [[1,0],[0,0]]
            w_coeff = [[0,0],[1,-1]] # C10 += M1, C11 -= M1
        elif r == 2:
            u_coeff = [[1,0],[0,0]]
            v_coeff = [[0,1],[0,-1]]
            w_coeff = [[0,1],[0,1]] # C01 += M2, C11 += M2
        elif r == 3:
            u_coeff = [[0,0],[0,1]]
            v_coeff = [[-1,0],[1,0]]
            w_coeff = [[1,0],[1,0]] # C00 += M3, C10 += M3
        elif r == 4:
            u_coeff = [[1,1],[0,0]]
            v_coeff = [[0,0],[0,1]]
            w_coeff = [[-1,1],[0,0]] # C00 -= M4, C01 += M4
        elif r == 5:
            u_coeff = [[-1,0],[1,0]]
            v_coeff = [[1,1],[0,0]]
            w_coeff = [[0,0],[0,1]] # C11 += M5
        elif r == 6:
            u_coeff = [[0,1],[0,-1]]
            v_coeff = [[0,1],[1,1]]
            w_coeff = [[1,0],[0,1]] # C00 += M6, C11 += M6
            
        # Now, for each of the 7 scalar multiplications in the base 2x2x2 problem:
        for s in range(7):
            rank_idx = r * 7 + s
            
            # For each entry in the 2x2 block matrices
            for I in range(2):
                for J in range(2):
                    coeff_u = u_coeff[I][J]
                    if coeff_u == 0: continue
                    
                    # For each scalar (i', j') in block A_IJ
                    for i_prime in range(2):
                        for j_prime in range(2):
                            a_idx = (2*I + i_prime) * 4 + (2*J + j_prime)
                            # The coefficient in base_U for this scalar position
                            # base_U is for 2x2x2. Indices:
                            # a_base = i_prime*2 + j_prime
                            val = base_U[a_idx_base][s]
                            if val != 0:
                                U[a_idx][rank_idx] += coeff_u * val
                            
            for J in range(2):
                for K in range(2):
                    coeff_v = v_coeff[J][K]
                    if coeff_v == 0: continue
                    
                    for j_prime in range(2):
                        for k_prime in range(2):
                            b_idx = (2*J + j_prime) * 4 + (2*K + k_prime)
                            b_idx_base = j_prime*2 + k_prime
                            val = base_V[b_idx_base][s]
                            if val != 0:
                                V[b_idx][rank_idx] += coeff_v * val
                            
            for K in range(2):
                for I in range(2):
                    coeff_w = w_coeff[K][I]
                    if coeff_w == 0: continue
                    
                    for k_prime in range(2):
                        for i_prime in range(2):
                            c_idx = (2*K + k_prime) * 4 + (2*I + i_prime)
                            c_idx_base = k_prime*2 + i_prime
                            val = base_W[c_idx_base][s]
                            if val != 0:
                                W[c_idx][rank_idx] += coeff_w * val
                                
    return U, V, W

def _construct_4x4x8_strassen():
    """Constructs rank-98 decomposition for 4x4x8."""
    U4, V4, W4 = _construct_4x4x4_strassen()
    
    n = 4
    m = 4
    p = 8
    rank = 98
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Left half (cols 0-3 of B, cols 0-3 of C)
    for i in range(n*m):
        for r in range(49):
            U[i][r] = U4[i][r]
            
    for j in range(m):
        for k in range(4):
            b_idx = j*4 + k
            for r in range(49):
                V[b_idx][r] = V4[b_idx][r]
                
    for k in range(4):
        for i in range(n):
            c_idx = k*4 + i
            for r in range(49):
                W[c_idx][r] = W4[c_idx][r]
                
    # Right half (cols 4-7 of B, cols 4-7 of C)
    for j in range(m):
        for k in range(4, 8):
            b_idx = j*4 + k
            k_local = k - 4
            b_idx_local = j*4 + k_local
            for r in range(49):
                V[b_idx][49 + r] = V4[b_idx_local][r]
                
    for k in range(4, 8):
        for i in range(n):
            c_idx = k*4 + i
            k_local = k - 4
            c_idx_local = k_local*4 + i
            for r in range(49):
                W[c_idx][49 + r] = W4[c_idx_local][r]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _construct_4x4x8_strassen()
    return {"U": U, "V": V, "W": W}
