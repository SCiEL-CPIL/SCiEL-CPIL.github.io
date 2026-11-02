"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS
    # Laderman's rank-23 algorithm for 3x3 matrix multiplication
    # U indices: A[i][j] = i*3+j -> A00=0,A01=1,A02=2,A10=3,A11=4,A12=5,A20=6,A21=7,A22=8
    # V indices: B[j][k] = j*3+k -> B00=0,B01=1,B02=2,B10=3,B11=4,B12=5,B20=6,B21=7,B22=8
    # W indices: C[i][k] = k*3+i -> C00=0,C10=1,C20=2,C01=3,C11=4,C21=5,C02=6,C12=7,C22=8
    
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def set_term(u_coeffs, v_coeffs, w_coeffs):
        """u_coeffs, v_coeffs, w_coeffs are dicts {index: coeff}"""
        nonlocal col
        for idx, c in u_coeffs.items():
            U[idx][col] = c
        for idx, c in v_coeffs.items():
            V[idx][col] = c
        for idx, c in w_coeffs.items():
            W[idx][col] = c
        col += 1
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    set_term({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    set_term({3:1, 4:1}, {0:1}, {1:1, 4:-1})
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    set_term({0:1}, {1:1, 4:-1}, {3:1, 4:1})
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    set_term({4:1}, {3:1, 0:-1}, {0:1, 1:1})
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    set_term({0:1, 1:1}, {4:1}, {0:-1, 3:1})
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    set_term({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    set_term({1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # M8 = (A02+A20)(B20+B02) -> C00 += A02*B20, C02 += A02*B02... 
    # Wait: A02*B20 -> C[0][0], A02*B02 -> invalid (j mismatch)
    # Let me reconsider. A[i,j]*B[j,k]:
    # A02: i=0,j=2. B20: j=2,k=0. Valid -> C[0][0]
    # A02: i=0,j=2. B02: j=0,k=2. INVALID (j=2 != j=0)
    # A20: i=2,j=0. B20: j=2,k=0. INVALID (j=0 != j=2)
    # A20: i=2,j=0. B02: j=0,k=2. Valid -> C[2][2]
    # So (A02+A20)(B20+B02) gives A02*B20 + A02*B02(invalid) + A20*B20(invalid) + A20*B02
    # This doesn't work as a single rank-1 term in the tensor decomposition sense.
    # 
    # In the tensor decomposition, a rank-1 term with U having nonzeros at {2,6}, V at {6,2}, W at {0,8}
    # would contribute to T[2,6,0], T[2,2,0], T[6,6,0], T[6,2,0], T[2,6,8], T[2,2,8], T[6,6,8], T[6,2,8]
    # T[2,6,0] = A02*B20 -> C00 (valid)
    # T[2,2,0] = A02*B02 -> this would be A[0,2]*B[0,2] which is NOT a valid mat-mult term
    # So this approach creates spurious terms.
    
    # Let me use a different strategy. I'll use the known Laderman decomposition.
    # After careful derivation, here are the 23 terms:
    
    # M8 = A02*B20 -> C00
    set_term({2:1}, {6:1}, {0:1})
    
    # M9 = A02*B21 -> C01
    set_term({2:1}, {7:1}, {3:1})
    
    # M10 = A12*B20 -> C10
    set_term({5:1}, {6:1}, {1:1})
    
    # M11 = A12*B21 -> C11
    set_term({5:1}, {7:1}, {4:1})
    
    # M12 = A20*B00 -> C20
    set_term({6:1}, {0:1}, {2:1})
    
    # M13 = A21*B10 -> C20
    set_term({7:1}, {3:1}, {2:1})
    
    # M14 = A20*B01 -> C21
    set_term({6:1}, {1:1}, {5:1})
    
    # M15 = A21*B11 -> C21
    set_term({7:1}, {4:1}, {5:1})
    
    # M16 = A00*B02 -> C02
    set_term({0:1}, {2:1}, {6:1})
    
    # M17 = A01*B12 -> C02
    set_term({1:1}, {5:1}, {6:1})
    
    # M18 = A10*B02 -> C12
    set_term({3:1}, {2:1}, {7:1})
    
    # M19 = A11*B12 -> C12
    set_term({4:1}, {5:1}, {7:1})
    
    # M20 = A22*B20 -> C20
    set_term({8:1}, {6:1}, {2:1})
    
    # M21 = A22*B21 -> C21
    set_term({8:1}, {7:1}, {5:1})
    
    # M22 = A20*B02 -> C22
    set_term({6:1}, {2:1}, {8:1})
    
    # M23 = A21*B12 -> C22
    set_term({7:1}, {5:1}, {8:1})
    
    # Wait, that's 23 terms but I'm missing A02*B22, A12*B22, A22*B22 for C02, C12, C22
    # Let me recount. The full decomposition needs:
    # C00: A00*B00+A01*B10+A02*B20 (Strassen handles A00*B00+A01*B10 part, need A02*B20)
    # C01: A00*B01+A01*B11+A02*B21 (Strassen handles first two, need A02*B21)
    # C02: A00*B02+A01*B12+A02*B22
    # C10: A10*B00+A11*B10+A12*B20 (Strassen handles first two, need A12*B20)
    # C11: A10*B01+A11*B11+A12*B21 (Strassen handles first two, need A12*B21)
    # C12: A10*B02+A11*B12+A12*B22
    # C20: A20*B00+A21*B10+A22*B20
    # C21: A20*B01+A21*B11+A22*B21
    # C22: A20*B02+A21*B12+A22*B22
    
    # So I need terms for:
    # A02*B20, A02*B21, A02*B22 (3 terms)
    # A12*B20, A12*B21, A12*B22 (3 terms)
    # A20*B00, A20*B01, A20*B02 (3 terms)
    # A21*B10, A21*B11, A21*B12 (3 terms)
    # A22*B20, A22*B21, A22*B22 (3 terms)
    # A00*B02, A01*B12 (2 terms)
    # A10*B02, A11*B12 (2 terms)
    # Total: 3+3+3+3+3+2+2 = 19 border terms + 7 Strassen = 26
    
    # To get to 23, I need to save 3 terms. The Laderman algorithm does this by
    # combining some of these. Let me try:
    # Combine A02*B22 and A12*B22: (A02+A12)*B22 gives both C02 and C12 contributions
    # but also gives spurious A02*B22->C12 and A12*B22->C02.
    # 
    # Actually in the tensor formulation, a term with U={2:1,5:1}, V={8:1}, W={6:1,7:1}
    # gives T[2,8,6]=1, T[2,8,7]=1, T[5,8,6]=1, T[5,8,7]=1
    # T[2,8,6] = A02*B22 -> C02 (correct)
    # T[2,8,7] = A02*B22 -> C12 (spurious!)
    # T[5,8,6] = A12*B22 -> C02 (spurious!)
    # T[5,8,7] = A12*B22 -> C12 (correct)
    # So this doesn't work.
    
    # I think rank 23 requires a more sophisticated combination. Let me just keep rank 26
    # but verify it's correct, since I'm not confident about the Laderman implementation.
    
    # Actually, let me reset and just return the known-good rank-26 decomposition.
    # The above 23 terms I listed are incomplete/incorrect.
    
    # Let me rebuild properly with rank 26.
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1
    
    # 1. Strassen 2x2 core (rank 7)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. Border terms (rank 19)
    # A02*B20 -> C00
    add_term(2, 6, 0, 1)
    # A02*B21 -> C01
    add_term(2, 7, 3, 1)
    # A12*B20 -> C10
    add_term(5, 6, 1, 1)
    # A12*B21 -> C11
    add_term(5, 7, 4, 1)
    # A00*B02 -> C02
    add_term(0, 2, 6, 1)
    # A01*B12 -> C02
    add_term(1, 5, 6, 1)
    # A10*B02 -> C12
    add_term(3, 2, 7, 1)
    # A11*B12 -> C12
    add_term(4, 5, 7, 1)
    # A02*B22 -> C02
    add_term(2, 8, 6, 1)
    # A12*B22 -> C12
    add_term(5, 8, 7, 1)
    # A20*B00 -> C20
    add_term(6, 0, 2, 1)
    # A21*B10 -> C20
    add_term(7, 3, 2, 1)
    # A20*B01 -> C21
    add_term(6, 1, 5, 1)
    # A21*B11 -> C21
    add_term(7, 4, 5, 1)
    # A22*B20 -> C20
    add_term(8, 6, 2, 1)
    # A22*B21 -> C21
    add_term(8, 7, 5, 1)
    # A20*B02 -> C22
    add_term(6, 2, 8, 1)
    # A21*B12 -> C22
    add_term(7, 5, 8, 1)
    # A22*B22 -> C22
    add_term(8, 8, 8, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
