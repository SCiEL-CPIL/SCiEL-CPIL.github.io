"""Optimized 4x4x4 matrix multiplication tensor decomposition using Strassen's algorithm."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_decomposition():
    """
    Return U, V, W for 2x2 matrix multiplication using Strassen's 7-multiplication algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*m+j, r] for A[i,j], V[j*p+k, r] for B[j,k], W[k*n+i, r] for C[i,k]
    where n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each multiplication r, we have coefficients in U (for A), V (for B), W (for C)
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    
    # M1: (A00+A11) * (B00+B11), contributes +1 to C00, +1 to C11
    U[0][0] += 1; U[3][0] += 1  # A00, A11
    V[0][0] += 1; V[3][0] += 1  # B00, B11
    W[0][0] += 1; W[3][0] += 1  # C00 (k=0,i=0), C11 (k=1,i=1)
    
    # M2: (A10+A11) * B00, contributes +1 to C10
    U[2][1] += 1; U[3][1] += 1  # A10, A11
    V[0][1] += 1                # B00
    W[1][1] += 1                # C10 (k=0,i=1)
    
    # M3: A00 * (B01-B11), contributes +1 to C01, -1 to C11
    U[0][2] += 1                # A00
    V[1][2] += 1; V[3][2] += -1  # B01, -B11
    W[2][2] += 1; W[3][2] += -1  # C01 (k=1,i=0), -C11 (k=1,i=1)
    
    # M4: A11 * (B10-B00), contributes +1 to C00, +1 to C10
    U[3][3] += 1                # A11
    V[2][3] += 1; V[0][3] += -1  # B10, -B00
    W[0][3] += 1; W[1][3] += 1  # C00, C10
    
    # M5: (A00+A01) * B11, contributes -1 to C00, +1 to C01
    U[0][4] += 1; U[1][4] += 1  # A00, A01
    V[3][4] += 1                # B11
    W[0][4] += -1; W[2][4] += 1  # -C00, C01
    
    # M6: (A10-A00) * (B00+B01), contributes +1 to C11
    U[2][5] += 1; U[0][5] += -1  # A10, -A00
    V[0][5] += 1; V[1][5] += 1  # B00, B01
    W[3][5] += 1                # C11
    
    # M7: (A01-A11) * (B10+B11), contributes +1 to C00, -1 to C11
    U[1][6] += 1; U[3][6] += -1  # A01, -A11
    V[2][6] += 1; V[3][6] += 1  # B10, B11
    W[0][6] += 1; W[3][6] += -1  # C00, -C11
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_4x4_decomposition():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using two levels of Strassen.
    
    Level 1: Partition 4x4 into 2x2 blocks, use 7 block multiplications.
    Level 2: Each 2x2 block multiplication uses Strassen's 7 scalar multiplications.
    Total: 7 * 7 = 49 scalar multiplications.
    """
    n = m = p = 4
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # Block-level Strassen terms for 4x4:
    # A = [[A00, A01],     B = [[B00, B01],
    #      [A10, A11]]           [B10, B11]]
    # where Aij is 2x2, with Aij[a,b] = A[2*i+a, 2*j+b]
    
    # Strassen for 4x4 (block level):
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    block_terms = [
        # M1: P = A00+A11, Q = B00+B11, C: C00+=1, C11+=1
        {'P': {(0,0): 1, (1,1): 1}, 'Q': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        # M2: P = A10+A11, Q = B00, C: C10+=1
        {'P': {(1,0): 1, (1,1): 1}, 'Q': {(0,0): 1}, 'C': {(1,0): 1}},
        # M3: P = A00, Q = B01-B11, C: C01+=1, C11+=-1
        {'P': {(0,0): 1}, 'Q': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): -1}},
        # M4: P = A11, Q = B10-B00, C: C00+=1, C10+=1
        {'P': {(1,1): 1}, 'Q': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        # M5: P = A00+A01, Q = B11, C: C00+=-1, C01+=1
        {'P': {(0,0): 1, (0,1): 1}, 'Q': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        # M6: P = A10-A00, Q = B00+B01, C: C11+=1
        {'P': {(1,0): 1, (0,0): -1}, 'Q': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        # M7: P = A01-A11, Q = B10+B11, C: C00+=1, C11+=-1
        {'P': {(0,1): 1, (1,1): -1}, 'Q': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): -1}},
    ]
    
    for rb, term in enumerate(block_terms):
        P_coeffs = term['P']  # (bi,bj) -> coeff
        Q_coeffs = term['Q']  # (bi,bj) -> coeff
        C_coeffs = term['C']  # (ci,cj) -> coeff
        
        for rs in range(7):
            r = rb * 7 + rs  # global rank index
            
            # For U: P[a,b] = sum over (bi,bj) in P_coeffs: coeff * A[2*bi+a, 2*bj+b]
            # The rs-th Strassen term has U2[a*2+b][rs] as coefficient for P[a,b]
            for (bi, bj), pc in P_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        u2_coeff = U2[a * 2 + b][rs]
                        if u2_coeff != 0:
                            ai = 2 * bi + a
                            aj = 2 * bj + b
                            U[ai * 4 + aj][r] += pc * u2_coeff
            
            # For V: Q[a,b] = sum over (bi,bj) in Q_coeffs: coeff * B[2*bi+a, 2*bj+b]
            for (bi, bj), qc in Q_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        v2_coeff = V2[a * 2 + b][rs]
                        if v2_coeff != 0:
                            bi_idx = 2 * bi + a
                            bj_idx = 2 * bj + b
                            V[bi_idx * 4 + bj_idx][r] += qc * v2_coeff
            
            # For W: R[a,b] = C[2*ci+a, 2*cj+b]
            # W index: k*n+i = ccol*4+crow
            for (ci, cj), cc in C_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        w2_coeff = W2[a * 2 + b][rs]
                        if w2_coeff != 0:
                            crow = 2 * ci + a
                            ccol = 2 * cj + b
                            W[ccol * 4 + crow][r] += cc * w2_coeff
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Construct schoolbook rank n*m*p decomposition."""
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

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4x4 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} where U, V, W are factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank-49 Strassen decomposition
    U, V, W = _strassen_4x4_decomposition()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
