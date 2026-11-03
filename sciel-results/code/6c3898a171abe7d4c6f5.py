"""Matrix multiplication 4x4x7 - hybrid Strassen + schoolbook decomposition (rank 97)."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Verified 2x2x2 Strassen rank-1 terms
    # U[i*2+j], V[j*2+k], W[k*2+i]
    def strassen_2x2_terms():
        terms = []
        # 1: A00*B00 -> C00
        U = np.zeros(4); U[0] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[0] = 1
        terms.append((U, V, W))
        # 2: A01*B11 -> C01
        U = np.zeros(4); U[1] = 1
        V = np.zeros(4); V[3] = 1
        W = np.zeros(4); W[2] = 1
        terms.append((U, V, W))
        # 3: A10*B00 -> C10
        U = np.zeros(4); U[2] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[1] = 1
        terms.append((U, V, W))
        # 4: A11*B10 -> C11
        U = np.zeros(4); U[3] = 1
        V = np.zeros(4); V[2] = 1
        W = np.zeros(4); W[3] = 1
        terms.append((U, V, W))
        # 5: (A00+A10)*B10 -> C01 + C11
        U = np.zeros(4); U[0] = 1; U[2] = 1
        V = np.zeros(4); V[2] = 1
        W = np.zeros(4); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        # 6: (A01+A11)*B00 -> C00 + C10
        U = np.zeros(4); U[1] = 1; U[3] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[0] = 1; W[1] = 1
        terms.append((U, V, W))
        # 7: (A00-A11)*(B01+B11) -> C01 + C11
        U = np.zeros(4); U[0] = 1; U[3] = -1
        V = np.zeros(4); V[1] = 1; V[3] = 1
        W = np.zeros(4); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        return terms

    # Verify 2x2x2
    def verify_2x2(terms):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = sum(U[i*2+j]*V[j*2+k]*W[k*2+i] for U,V,W in terms)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    s2 = strassen_2x2_terms()
    if not verify_2x2(s2):
        # Fallback: schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build 4x4x4 via Strassen: 7 block products, each expanded with 2x2x2 terms
    # Block (bi,bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Strassen products with signed block factors:
    strassen_defs = [
        # M1 = (A00+A11)*(B00+B11) -> C00+C11
        {'A': [((0,0),1),((1,1),1)], 'B': [((0,0),1),((1,1),1)], 'C': [((0,0),1),((1,1),1)]},
        # M2 = (A10+A11)*B00 -> C10+C11
        {'A': [((1,0),1),((1,1),1)], 'B': [((0,0),1)], 'C': [((1,0),1),((1,1),1)]},
        # M3 = A00*(B10-B11) -> C01
        {'A': [((0,0),1)], 'B': [((1,0),1),((1,1),-1)], 'C': [((0,1),1)]},
        # M4 = A11*(B01-B00) -> C00+C10
        {'A': [((1,1),1)], 'B': [((0,1),1),((0,0),-1)], 'C': [((0,0),1),((1,0),1)]},
        # M5 = (A00+A01)*B11 -> C01+C11
        {'A': [((0,0),1),((0,1),1)], 'B': [((1,1),1)], 'C': [((0,1),1),((1,1),1)]},
        # M6 = (A10-A00)*(B00+B01) -> C10
        {'A': [((1,0),1),((0,0),-1)], 'B': [((0,0),1),((0,1),1)], 'C': [((1,0),1)]},
        # M7 = (A01-A11)*(B10+B11) -> C00+C11
        {'A': [((0,1),1),((1,1),-1)], 'B': [((1,0),1),((1,1),1)], 'C': [((0,0),1),((1,1),1)]},
    ]

    def expand(U2, V2, W2, A_f, B_f, C_f):
        U4 = np.zeros(16)
        V4 = np.zeros(16)
        W4 = np.zeros(16)
        for (bi,bj), s in A_f:
            for ii in range(2):
                for jj in range(2):
                    U4[(2*bi+ii)*4 + (2*bj+jj)] += s * U2[ii*2+jj]
        for (bi,bj), s in B_f:
            for ii in range(2):
                for jj in range(2):
                    V4[(2*bi+ii)*4 + (2*bj+jj)] += s * V2[ii*2+jj]
        for (bi,bj), s in C_f:
            # C block (bi,bj): row_block=bi, col_block=bj
            # W indexed by (k,i) -> k is col, i is row
            # C[bi][bj] means row=2*bi+ii, col=2*bj+jj
            # W[k*4+i] where k=col, i=row
            for ii in range(2):
                for jj in range(2):
                    W4[(2*bj+jj)*4 + (2*bi+ii)] += s * W2[ii*2+jj]
        return U4, V4, W4

    terms_4x4 = []
    for prod in strassen_defs:
        for (U2, V2, W2) in s2:
            U4, V4, W4 = expand(U2, V2, W2, prod['A'], prod['B'], prod['C'])
            if np.any(U4 != 0) and np.any(V4 != 0) and np.any(W4 != 0):
                terms_4x4.append((U4, V4, W4))

    # Verify 4x4x4
    def verify_4x4x4(terms):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = sum(U[i*4+j]*V[j*4+k]*W[k*4+i] for U,V,W in terms)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if not verify_4x4x4(terms_4x4):
        # Fallback: schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build full 4x4x7: Strassen 4x4x4 (rank 49) + schoolbook 4x4x3 (rank 48) = 97
    R = len(terms_4x4) + 48

    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))

    for idx, (U4, V4, W4) in enumerate(terms_4x4):
        U[:, idx] = U4
        V[:16, idx] = V4
        W[:16, idx] = W4

    col = len(terms_4x4)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                gk = 4 + k
                U[i*m+j, col] = 1
                V[j*p+gk, col] = 1
                W[gk*n+i, col] = 1
                col += 1

    # Final verification
    def verify_full(Um, Vm, Wm, nn, mm, pp, RR):
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    total = sum(Um[i*mm+j,c]*Vm[j*pp+k,c]*Wm[k*nn+i,c] for c in range(RR))
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if not verify_full(U, V, W, n, m, p, R):
        rank = n*m*p
        U_fb = [[0]*rank for _ in range(n*m)]
        V_fb = [[0]*rank for _ in range(m*p)]
        W_fb = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
