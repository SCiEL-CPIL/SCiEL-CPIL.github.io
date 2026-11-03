import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    deadline = time.time() + 140.0
    
    # Fallback: Schoolbook (Rank 64)
    best_rank = n * m * p
    best_U = np.zeros((U_dim, best_rank))
    best_V = np.zeros((V_dim, best_rank))
    best_W = np.zeros((W_dim, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
    
    def verify_decomp(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(U.shape[1]):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Strassen decomposition for 2x2 * 2x2 -> 2x2
    # 7 terms, each with coefficients for A (2x2), B (2x2), C (2x2)
    # Term t: U_part[i, jo], V_part[jo, ko], W_part[i, ko]
    # where i in {0,1} (row of A/C), jo in {0,1} (col of A / row of B), ko in {0,1} (col of B / col of C)
    
    # Strassen terms for 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each term, define the 2x2 A-coeff, 2x2 B-coeff, 2x2 C-coeff
    # A-coeff: row i, col jo
    # B-coeff: row jo, col ko
    # C-coeff: row i, col ko
    
    strassen_terms = []
    
    # M1 = (a00+a11)(b00+b11)
    # A: a00=1, a11=1, others 0
    # B: b00=1, b11=1, others 0
    # C: contributes to C00=+1, C11=+1
    strassen_terms.append((
        [[1,0],[1,0]],  # A: row0=[1,0], row1=[1,0] -> a00=1, a10=0, a11=1 wait
        # Let me be more careful. A is 2x2: A[i][jo]
        # a00 = A[0][0], a01 = A[0][1], a10 = A[1][0], a11 = A[1][1]
        # M1 = (a00+a11)(b00+b11)
        # So A coefficients: a00 -> 1, a11 -> 1, a01 -> 0, a10 -> 0
        # A as [row][col]: row0 = [a00, a01] = [1, 0], row1 = [a10, a11] = [0, 1]
        [[1, 0], [0, 1]],
        # B: b00=1, b11=1, b01=0, b10=0
        # B[jo][ko]: row0 = [b00, b01] = [1, 0], row1 = [b10, b11] = [0, 1]
        [[1, 0], [0, 1]],
        # C: C00=+1, C01=0, C10=0, C11=+1
        # C[i][ko]: row0 = [C00, C01] = [1, 0], row1 = [C10, C11] = [0, 1]
        [[1, 0], [0, 1]]
    ))
    
    # M2 = (a10+a11)b00
    # A: a10=1, a11=1 -> row0=[0,0], row1=[1,1]
    # B: b00=1 -> row0=[1,0], row1=[0,0]
    # C: C10=+1 -> row0=[0,0], row1=[1,0]
    strassen_terms.append((
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]]
    ))
    
    # M3 = a00(b01-b11)
    # A: a00=1 -> row0=[1,0], row1=[0,0]
    # B: b01=1, b11=-1 -> row0=[0,1], row1=[0,-1]
    # C: C01=+1 -> row0=[0,1], row1=[0,0]
    strassen_terms.append((
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[0, 1], [0, 0]]
    ))
    
    # M4 = a11(b10-b00)
    # A: a11=1 -> row0=[0,0], row1=[0,1]
    # B: b10=1, b00=-1 -> row0=[-1,0], row1=[1,0]
    # C: C00=+1, C10=+1 -> row0=[1,0], row1=[1,0]
    strassen_terms.append((
        [[0, 0], [0, 1]],
        [[-1, 0], [1, 0]],
        [[1, 0], [1, 0]]
    ))
    
    # M5 = (a00+a01)b11
    # A: a00=1, a01=1 -> row0=[1,1], row1=[0,0]
    # B: b11=1 -> row0=[0,0], row1=[0,1]
    # C: C00=-1, C01=+1 -> row0=[-1,1], row1=[0,0]
    strassen_terms.append((
        [[1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[-1, 1], [0, 0]]
    ))
    
    # M6 = (a10-a00)(b00+b01)
    # A: a10=1, a00=-1 -> row0=[-1,0], row1=[1,0]
    # B: b00=1, b01=1 -> row0=[1,1], row1=[0,0]
    # C: C11=+1 -> row0=[0,0], row1=[0,1]
    strassen_terms.append((
        [[-1, 0], [1, 0]],
        [[1, 1], [0, 0]],
        [[0, 0], [0, 1]]
    ))
    
    # M7 = (a01-a11)(b10+b11)
    # A: a01=1, a11=-1 -> row0=[0,1], row1=[0,-1]
    # B: b10=1, b11=1 -> row0=[0,0], row1=[1,1]
    # C: C00=+1 -> row0=[1,0], row1=[0,0]
    strassen_terms.append((
        [[0, 1], [0, -1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]]
    ))
    
    # Total rank: 8 sub-problems * 7 terms = 56
    total_rank = 8 * 7
    U = np.zeros((U_dim, total_rank))
    V = np.zeros((V_dim, total_rank))
    W = np.zeros((W_dim, total_rank))
    
    t_offset = 0
    for jb in range(2):  # j-block: j = 2*jb + jo
        for kb in range(4):  # k-block: k = 4*kb + ko
            for s in range(7):  # Strassen term
                t = t_offset + s
                A_coeff, B_coeff, C_coeff = strassen_terms[s]
                
                for i in range(2):  # row of A and C
                    for jo in range(2):  # col of A, row of B
                        j = 2 * jb + jo
                        U[i*m + j, t] = A_coeff[i][jo]
                    
                    for ko in range(2):  # col of B, col of C
                        k = 4 * kb + ko
                        W[k*n + i, t] = C_coeff[i][ko]
                
                for jo in range(2):  # row of B
                    for ko in range(2):  # col of B
                        j = 2 * jb + jo
                        k = 4 * kb + ko
                        V[j*p + k, t] = B_coeff[jo][ko]
            
            t_offset += 7
    
    # Verify
    if verify_decomp(U, V, W):
        best_rank = total_rank
        best_U = U
        best_V = V
        best_W = W
    
    # Round to ensure exact integers
    U_out = np.round(best_U)
    V_out = np.round(best_V)
    W_out = np.round(best_W)
    
    # Final verification with rounded values
    if verify_decomp(U_out, V_out, W_out):
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    # Fallback to schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

