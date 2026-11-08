"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """Strassen-based decomposition for 4x4 matrix multiplication, rank 49."""
    n, m, p = DIMENSIONS
    
    # We want to find U, V, W such that:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) = C[i,k]
    # This is equivalent to:
    # sum_c W[k*n+i, c] * (U_c . A_row_i) * (V_c . B_col_k) = C[i,k]
    
    # For 4x4, we use Strassen on 2x2 blocks.
    # A is divided into 4 blocks: A00, A01, A10, A11 (each 2x2)
    # B is divided into 4 blocks: B00, B01, B10, B11 (each 2x2)
    # C is divided into 4 blocks: C00, C01, C10, C11 (each 2x2)
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix product.
    # We decompose each Mk into 4 scalar products (schoolbook for 2x2).
    # Total rank = 7 * 4 = 28? No.
    # The rank of 2x2 matrix multiplication is 7.
    # So each Mk (which is a 2x2 x 2x2 product) has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # We need to construct U, V, W for the full 4x4 case.
    # Each of the 7 Strassen multiplications Mk is itself a 2x2 matrix multiplication.
    # Let's denote the 2x2 multiplication tensor decomposition as having rank 7.
    # Let the 7 rank-1 terms for 2x2 be (u_r, v_r, w_r) for r=1..7.
    # Then for Mk = X_k * Y_k, we have:
    # Mk[a,b] = sum_{r=1}^7 w_r * (sum_j u_r[a,j] X_k[a,j]) * (sum_j v_r[j,b] Y_k[j,b])
    
    # X_k and Y_k are linear combinations of A-blocks and B-blocks.
    # X_k[a,j] = sum_{(bi,bj) in S_k} c_s * A[bi*2+a, bj*2+j]
    # Y_k[j,b] = sum_{(bi2,bj2) in T_k} c_t * B[bi2*2+j, bj2*2+b]
    
    # Substituting these into the 2x2 decomposition:
    # Mk[a,b] = sum_{r=1}^7 w_r * (sum_j u_r[a,j] sum_{(bi,bj) in S_k} c_s A[bi*2+a, bj*2+j]) * (sum_j v_r[j,b] sum_{(bi2,bj2) in T_k} c_t B[bi2*2+j, bj2*2+b])
    
    # This gives 7 * 7 = 49 terms for the whole 4x4 product.
    # Each term corresponds to a specific (k, r) pair.
    # Let's index the full rank by (k, r) where k in 1..7, r in 1..7.
    # Total 49 terms.
    
    # We need to define the 2x2 Strassen decomposition (u_r, v_r, w_r).
    # Standard 2x2 Strassen:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The rank-1 terms for the 2x2 tensor are defined by the bilinear forms.
    # For each M_r, we have a term: w_r * (linear combo of A) * (linear combo of B).
    # Let's define the U, V, W entries for the 2x2 case.
    # For a specific M_r, the contribution to C[i,j] is w_r * U_2x2[i, r] * V_2x2[j, r] ?
    # No, the 2x2 decomposition is:
    # C[i,j] = sum_{r=1}^7 W_2x2[i,j,r] * (sum_{k} U_2x2[i,k,r] A[i,k]) * (sum_{k} V_2x2[k,j,r] B[k,j])
    
    # We can hardcode the 2x2 Strassen U, V, W matrices.
    # U_2x2 shape (2*2, 7) = (4, 7)
    # V_2x2 shape (2*2, 7) = (4, 7)
    # W_2x2 shape (2*2, 7) = (4, 7)
    
    # Let's construct U_2x2, V_2x2, W_2x2 for the 2x2 case.
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # M1: (a+d)(e+h)
    # A part: a+d -> U[0,0]=1, U[1,1]=1 (for a=0,0 and d=1,1)
    # B part: e+h -> V[0,0]=1, V[1,1]=1 (for e=0,0 and h=1,1)
    # W part: contributes to C11 (+1), C22 (+1)
    # W[0,0] += 1 (C11)
    # W[3,0] += 1 (C22)
    
    # M2: (c+d)e
    # A part: c+d -> U[2,0]=1, U[3,1]=1 (c=1,0, d=1,1)
    # B part: e -> V[0,0]=1
    # W part: contributes to C21 (+1), C22 (-1)
    # W[2,1] += 1 (C21)
    # W[3,1] -= 1 (C22)
    
    # M3: a(f-h)
    # A part: a -> U[0,0]=1
    # B part: f-h -> V[0,1]=1, V[1,1]=-1
    # W part: contributes to C12 (+1), C22 (+1)
    # W[1,2] += 1 (C12)
    # W[3,2] += 1 (C22)
    
    # M4: d(g-e)
    # A part: d -> U[1,1]=1
    # B part: g-e -> V[1,0]=1, V[0,0]=-1
    # W part: contributes to C11 (+1), C21 (+1)
    # W[0,3] += 1 (C11)
    # W[2,3] += 1 (C21)
    
    # M5: (a+b)h
    # A part: a+b -> U[0,0]=1, U[0,1]=1
    # B part: h -> V[1,1]=1
    # W part: contributes to C11 (-1), C12 (+1)
    # W[0,4] -= 1 (C11)
    # W[1,4] += 1 (C12)
    
    # M6: (c-a)(e+f)
    # A part: c-a -> U[2,0]=1, U[0,0]=-1
    # B part: e+f -> V[0,0]=1, V[0,1]=1
    # W part: contributes to C22 (+1)
    # W[3,5] += 1 (C22)
    
    # M7: (b-d)(g+h)
    # A part: b-d -> U[0,1]=1, U[1,1]=-1
    # B part: g+h -> V[1,0]=1, V[1,1]=1
    # W part: contributes to C11 (+1)
    # W[0,6] += 1 (C11)
    
    # Let's build the 2x2 matrices U2, V2, W2.
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0 (M1)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1 (M2)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # r=2 (M3)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # r=3 (M4)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # r=4 (M5)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # r=5 (M6)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # r=6 (M7)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Now, for 4x4, we have 7 block multiplications M1..M7.
    # Each Mk uses the same 2x2 decomposition structure, but the inputs are linear combos of blocks.
    # Let's define the block combinations for 4x4 Strassen.
    # S_k: coefficients for A blocks. T_k: coefficients for B blocks.
    # A blocks: 00, 01, 10, 11.
    # B blocks: 00, 01, 10, 11.
    
    S = {
        0: {0: 1, 3: 1}, # M1: A00 + A11
        1: {2: 1, 3: 1}, # M2: A10 + A11
        2: {0: 1},       # M3: A00
        3: {3: 1},       # M4: A11
        4: {0: 1, 1: 1}, # M5: A00 + A01
        5: {2: 1, 0: -1},# M6: A10 - A00
        6: {1: 1, 3: -1} # M7: A01 - A11
    }
    T = {
        0: {0: 1, 3: 1}, # M1: B00 + B11
        1: {0: 1},       # M2: B00
        2: {1: 1, 3: -1},# M3: B01 - B11
        3: {2: 1, 0: -1},# M4: B10 - B00
        4: {3: 1},       # M5: B11
        5: {0: 1, 1: 1}, # M6: B00 + B01
        6: {2: 1, 3: 1}  # M7: B10 + B11
    }
    
    # C block combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_combos = {
        (0,0): {0: 1, 3: 1, 4: -1, 6: 1},
        (0,1): {2: 1, 4: 1},
        (1,0): {1: 1, 3: 1},
        (1,1): {0: 1, 1: -1, 2: 1, 5: 1}
    }
    
    # We construct U, V, W for 4x4.
    # Rank = 7 * 7 = 49.
    # Index c = k * 7 + r, where k is the Strassen multiplication index (0..6), r is the 2x2 rank index (0..6).
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for k in range(7):
        for r in range(7):
            c = k * 7 + r
            # U part
            # U_4x4[i*4+j, c]
            # U2[i2*2+j2, r] is the coefficient for A[i2, j2] in the 2x2 block.
            # For 4x4, the input to Mk is X_k.
            # X_k[a, b] = sum_{(bi,bj) in S_k} c_s * A[bi*2+a, bj*2+b]
            # So U_4x4[(bi*2+a)*4 + (bj*2+b), c] += c_s * U2[a*2+b, r]
            
            for (bi, bj), cs in S[k].items():
                for a in range(2):
                    for b in range(2):
                        i_idx = bi * 2 + a
                        j_idx = bj * 2 + b
                        u2_val = U2[a*2+b][r]
                        if u2_val != 0:
                            U[i_idx*4 + j_idx][c] += cs * u2_val
            
            # V part
            # V_4x4[j*4+k, c]
            # Y_k[a, b] = sum_{(bi2,bj2) in T_k} c_t * B[bi2*2+a, bj2*2+b]
            # V_4x4[(bi2*2+a)*4 + (bj2*2+b), c] += c_t * V2[a*2+b, r]
            
            for (bi2, bj2), ct in T[k].items():
                for a in range(2):
                    for b in range(2):
                        j_idx = bi2 * 2 + a
                        k_idx = bj2 * 2 + b
                        v2_val = V2[a*2+b][r]
                        if v2_val != 0:
                            V[j_idx*4 + k_idx][c] += ct * v2_val
            
            # W part
            # W_4x4[k*4+i, c]
            # The result Mk is added to C blocks according to C_combos.
            # Mk[a, b] contributes to C[ci*2+a, cj*2+b].
            # W_4x4[(ci*2+a)*4 + (cj*2+b), c] += sign * W2[a*2+b, r]
            
            for (ci, cj), signs in C_combos.items():
                if k not in signs:
                    continue
                sign = signs[k]
                for a in range(2):
                    for b in range(2):
                        k_idx = ci * 2 + a
                        i_idx = cj * 2 + b
                        w2_val = W2[a*2+b][r]
                        if w2_val != 0:
                            W[k_idx*4 + i_idx][c] += sign * w2_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
