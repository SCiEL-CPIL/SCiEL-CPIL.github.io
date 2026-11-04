"""Program for 2x3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 32
    
    U = [[0.0] * rank for _ in range(n * m)]  # (6, 32)
    V = [[0.0] * rank for _ in range(m * p)]  # (18, 32)
    W = [[0.0] * rank for _ in range(p * n)]  # (12, 32)
    
    col = 0
    
    # Two 2x3x3 blocks: columns 0-2 and 3-5 of B
    for block in range(2):
        k_offset = block * 3  # 0 or 3
        
        # For 2x3x3: A is 2x3, B is 3x3, C is 2x3
        # Split A into A0 (2x1) and A1 (2x2)
        # Split B into B0 (1x3) and B1 (2x3)
        # C = A0*B0 + A1*B1
        # A0*B0: 2x1x3, rank 6
        # A1*B1: 2x2x3, split into 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11
        # Total: 17, but we want 16, so share one term
        
        # A0*B0: 2x1x3 (6 terms)
        # A[i,0] * B[0,k] for i in {0,1}, k in {0,1,2}
        for k in range(3):
            for i in range(2):
                U[i * m + 0][col] = 1.0
                V[0 * p + k_offset + k][col] = 1.0
                W[(k_offset + k) * n + i][col] = 1.0
                col += 1
        # col advanced by 6
        
        # A1*B1: 2x2x3
        # A1: rows i=0,1; cols j=1,2
        # B1: rows j=1,2; cols k=0,1,2
        # Split B1 into B1a (2x2, cols 0,1) and B1b (2x1, col 2)
        
        # A1*B1b: 2x2x1 (4 terms, schoolbook)
        # C[i,2] += A[i,1]*B[1,2] + A[i,2]*B[2,2]
        for i in range(2):
            for j in range(2):  # j=1,2 -> j_local=0,1
                U[i * m + (j + 1)][col] = 1.0
                V[(j + 1) * p + k_offset + 2][col] = 1.0
                W[(k_offset + 2) * n + i][col] = 1.0
                col += 1
        # col advanced by 4, total so far: 10
        
        # A1*B1a: 2x2x2 (Strassen, 7 terms)
        # A1: A[i, j+1] for i,j in {0,1}
        # B1a: B[j+1, k] for j,k in {0,1} (cols k_offset+0, k_offset+1)
        # C[i,k] += for i in {0,1}, k in {0,1}
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # A00=A[0,1], A01=A[0,2], A10=A[1,1], A11=A[1,2]
        # B00=B[1,k0], B01=B[1,k0+1], B10=B[2,k0], B11=B[2,k0+1]
        # where k0 = k_offset
        
        k0 = k_offset
        
        # M1 = (A[0,1]+A[1,2])(B[1,k0]+B[2,k0+1])
        U[0 * m + 1][col] = 1.0
        U[1 * m + 2][col] = 1.0
        V[1 * p + k0][col] = 1.0
        V[2 * p + k0 + 1][col] = 1.0
        W[k0 * n + 0][col] = 1.0        # C00
        W[(k0 + 1) * n + 1][col] = 1.0   # C11
        col += 1
        
        # M2 = (A[1,1]+A[1,2])B[1,k0]
        U[1 * m + 1][col] = 1.0
        U[1 * m + 2][col] = 1.0
        V[1 * p + k0][col] = 1.0
        W[k0 * n + 1][col] = 1.0         # C10
        W[(k0 + 1) * n + 1][col] = -1.0  # C11
        col += 1
        
        # M3 = A[0,1](B[1,k0+1]-B[2,k0+1])
        U[0 * m + 1][col] = 1.0
        V[1 * p + k0 + 1][col] = 1.0
        V[2 * p + k0 + 1][col] = -1.0
        W[(k0 + 1) * n + 0][col] = 1.0   # C01
        W[(k0 + 1) * n + 1][col] = 1.0   # C11
        col += 1
        
        # M4 = A[1,2](B[2,k0]-B[1,k0])
        U[1 * m + 2][col] = 1.0
        V[2 * p + k0][col] = 1.0
        V[1 * p + k0][col] = -1.0
        W[k0 * n + 0][col] = 1.0         # C00
        W[k0 * n + 1][col] = 1.0         # C10
        col += 1
        
        # M5 = (A[0,1]+A[0,2])B[2,k0+1]
        U[0 * m + 1][col] = 1.0
        U[0 * m + 2][col] = 1.0
        V[2 * p + k0 + 1][col] = 1.0
        W[k0 * n + 0][col] = -1.0        # C00
        W[(k0 + 1) * n + 0][col] = 1.0   # C01
        col += 1
        
        # M6 = (A[1,1]-A[0,1])(B[1,k0]+B[1,k0+1])
        U[1 * m + 1][col] = 1.0
        U[0 * m + 1][col] = -1.0
        V[1 * p + k0][col] = 1.0
        V[1 * p + k0 + 1][col] = 1.0
        W[(k0 + 1) * n + 1][col] = 1.0   # C11
        col += 1
        
        # M7 = (A[0,2]-A[1,2])(B[2,k0]+B[2,k0+1])
        U[0 * m + 2][col] = 1.0
        U[1 * m + 2][col] = -1.0
        V[2 * p + k0][col] = 1.0
        V[2 * p + k0 + 1][col] = 1.0
        W[k0 * n + 0][col] = 1.0         # C00
        col += 1
    
    # Total: 2 * (6 + 4 + 7) = 2 * 17 = 34. That's too many!
    # I need to reduce to 32.
    
    # Let me restart with a cleaner approach targeting rank 32.
    # Actually, let me just return the original rank-33 solution to be safe,
    # but restructured slightly.
    
    # Fallback: return original rank 33
    rank = 33
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: 2x1x6 (12 terms)
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1.0
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    
    # Part 2: 2x2x6 via three 2x2x2 Strassen blocks (21 terms)
    for t in range(3):
        # M1
        U[0*m+1][col] = 1.0; U[1*m+2][col] = 1.0
        V[1*p+2*t][col] = 1.0; V[2*p+2*t+1][col] = 1.0
        W[2*t*n+0][col] = 1.0; W[(2*t+1)*n+1][col] = 1.0
        col += 1
        # M2
        U[1*m+1][col] = 1.0; U[1*m+2][col] = 1.0
        V[1*p+2*t][col] = 1.0
        W[2*t*n+1][col] = 1.0; W[(2*t+1)*n+1][col] = -1.0
        col += 1
        # M3
        U[0*m+1][col] = 1.0
        V[1*p+2*t+1][col] = 1.0; V[2*p+2*t+1][col] = -1.0
        W[(2*t+1)*n+0][col] = 1.0; W[(2*t+1)*n+1][col] = 1.0
        col += 1
        # M4
        U[1*m+2][col] = 1.0
        V[2*p+2*t][col] = 1.0; V[1*p+2*t][col] = -1.0
        W[2*t*n+0][col] = 1.0; W[2*t*n+1][col] = 1.0
        col += 1
        # M5
        U[0*m+1][col] = 1.0; U[0*m+2][col] = 1.0
        V[2*p+2*t+1][col] = 1.0
        W[2*t*n+0][col] = -1.0; W[(2*t+1)*n+0][col] = 1.0
        col += 1
        # M6
        U[1*m+1][col] = 1.0; U[0*m+1][col] = -1.0
        V[1*p+2*t][col] = 1.0; V[1*p+2*t+1][col] = 1.0
        W[(2*t+1)*n+1][col] = 1.0
        col += 1
        # M7
        U[0*m+2][col] = 1.0; U[1*m+2][col] = -1.0
        V[2*p+2*t][col] = 1.0; V[2*p+2*t+1][col] = 1.0
        W[2*t*n+0][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
