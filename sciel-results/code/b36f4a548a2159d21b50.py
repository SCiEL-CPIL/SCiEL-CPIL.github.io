import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(rank, U, V, W):
        """Check T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_strassen_4x4x4():
        """Construct rank-49 decomposition for 4x4x4 using Strassen."""
        rank = 49
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*4)] # Note: p=4 here for this helper
        W = [[0]*rank for _ in range(4*n)]

        # Scalar Strassen 2x2x2 coefficients
        # U_scalar[i, j, s], V_scalar[j, k, s], W_scalar[k, i, s]
        # s in 0..6
        
        # M0 = (a+d)(e+h)
        # A0 = [[1,0],[0,1]], B0 = [[1,0],[0,1]]
        # C00 += 1, C11 += 1
        U_s = [
            [[1, 0], [0, 1]], # s=0
            [[0, 0], [1, 1]], # s=1 (c+d)e -> A1=[[0,0],[1,1]]? No.
            # Let's re-derive scalar Strassen coeffs carefully.
            # M1 = (c+d)e. A1 = [[0,0],[1,1]], B1 = [[1,0],[0,0]]
            # M2 = a(f-h). A2 = [[1,0],[0,0]], B2 = [[0,1],[0,-1]]
            # M3 = d(g-e). A3 = [[0,0],[0,1]], B3 = [[0,0],[1,-1]]? No. B3 = [[0,0],[1,0]] - [[1,0],[0,0]]?
            # B3 = [[0,0],[1,-1]]? No.
            # d(g-e). A3 = [[0,0],[0,1]]. B3 = [[0,0],[1,-1]]?
            # Let's check B3. e=B00, g=B10.
            # B3 = [[0,0],[1,-1]] gives row 1: g - e. Correct.
            # M4 = (a+b)h. A4 = [[1,1],[0,0]], B4 = [[0,0],[0,1]]
            # M5 = (c-a)(e+f). A5 = [[-1,0],[1,0]], B5 = [[1,1],[0,0]]
            # M6 = (b-d)(g+h). A6 = [[0,1],[0,-1]], B6 = [[0,0],[1,1]]
        ]
        
        # Correct Scalar Strassen Definitions:
        # s=0: A=[[1,0],[0,1]], B=[[1,0],[0,1]]
        # s=1: A=[[0,0],[1,1]], B=[[1,0],[0,0]]
        # s=2: A=[[1,0],[0,0]], B=[[0,1],[0,-1]]
        # s=3: A=[[0,0],[0,1]], B=[[0,0],[1,-1]]
        # s=4: A=[[1,1],[0,0]], B=[[0,0],[0,1]]
        # s=5: A=[[-1,0],[1,0]], B=[[1,1],[0,0]]
        # s=6: A=[[0,1],[0,-1]], B=[[0,0],[1,1]]
        
        U_scalar = [
            [[1,0],[0,1]],
            [[0,0],[1,1]],
            [[1,0],[0,0]],
            [[0,0],[0,1]],
            [[1,1],[0,0]],
            [[-1,0],[1,0]],
            [[0,1],[0,-1]]
        ]
        V_scalar = [
            [[1,0],[0,1]],
            [[1,0],[0,0]],
            [[0,1],[0,-1]],
            [[0,0],[1,-1]],
            [[0,0],[0,1]],
            [[1,1],[0,0]],
            [[0,0],[1,1]]
        ]
        W_scalar = [
            [[1,0],[0,1]], # C00=1, C11=1
            [[0,0],[0,0]], # C10=1, C11=-1 -> W1: C10=1, C11=-1?
            # W_s[k, i] contributes to C[i, k]? No, W[k, i].
            # C[i, k] = sum U[i,j] V[j,k] W[k,i].
            # So W[k, i] is the coefficient for C[i, k].
            # s=0: C00=1, C11=1. W0[0,0]=1, W0[1,1]=1.
            # s=1: C10=1, C11=-1. W1[0,1]=1, W1[1,1]=-1.
            # s=2: C11=1. W2[1,1]=1.
            # s=3: C00=1. W3[0,0]=1.
            # s=4: C01=1, C11=1. W4[1,0]=1, W4[1,1]=1.
            # s=5: C10=1, C00=-1. W5[0,1]=1, W5[0,0]=-1.
            # s=6: C01=1, C00=-1. W6[1,0]=1, W6[0,0]=-1.
            
            [[1,0],[0,1]],
            [[0,0],[1,-1]],
            [[0,0],[0,1]],
            [[1,0],[0,0]],
            [[0,1],[0,1]],
            [[-1,1],[0,0]],
            [[-1,1],[0,0]] # Wait, s=6: C01=1, C00=-1. W6[1,0]=1, W6[0,0]=-1.
        ]
        # Fix W_scalar
        W_scalar = [
            [[1,0],[0,1]],
            [[0,0],[1,-1]],
            [[0,0],[0,1]],
            [[1,0],[0,0]],
            [[0,1],[0,1]],
            [[-1,1],[0,0]],
            [[-1,1],[0,0]]
        ]
        # Check s=6: C01=1, C00=-1.
        # W6[k, i]. k=1, i=0 -> 1. k=0, i=0 -> -1.
        # W6 = [[-1, 0], [1, 0]].
        W_scalar[6] = [[-1, 0], [1, 0]]
        
        # Block Strassen Coefficients
        # A_t = sum a[t, ib, jb] A[ib, jb]
        # B_t = sum b[t, ib, jb] B[ib, jb]
        # C_ij = sum c[t, ib, jb] M_t
        
        A_block = [
            [[1,0],[0,1]], # t=0: A00+A11
            [[0,0],[1,1]], # t=1: A10+A11
            [[1,0],[0,0]], # t=2: A00
            [[0,0],[0,1]], # t=3: A11
            [[1,1],[0,0]], # t=4: A00+A01
            [[-1,0],[1,0]],# t=5: A10-A00
            [[0,1],[0,-1]] # t=6: A01-A11
        ]
        B_block = [
            [[1,0],[0,1]], # t=0: B00+B11
            [[1,0],[0,0]], # t=1: B00
            [[0,1],[0,-1]],# t=2: B01-B11
            [[0,0],[1,-1]],# t=3: B10-B00
            [[0,0],[0,1]], # t=4: B11
            [[1,1],[0,0]], # t=5: B00+B01
            [[0,0],[1,1]]  # t=6: B10+B11
        ]
        C_block = [
            [[1,0],[0,1]], # t=0: C00+1, C11+1
            [[0,0],[1,-1]],# t=1: C10+1, C11-1
            [[0,0],[0,1]], # t=2: C11+1
            [[1,0],[0,0]], # t=3: C00+1
            [[0,1],[0,1]], # t=4: C01+1, C11+1
            [[-1,1],[0,0]],# t=5: C00-1, C10+1
            [[-1,1],[0,0]] # t=6: C00-1, C01+1? No.
        ]
        # Check t=6: C01+1, C00-1.
        # C_block[6][ib, jb] corresponds to C[ib, jb].
        # C00 is [0,0]. C01 is [0,1].
        # So C_block[6] should have -1 at [0,0] and 1 at [0,1].
        C_block[6] = [[-1, 1], [0, 0]]

        for t in range(7):
            for s in range(7):
                c_idx = t * 7 + s
                for ib in range(2):
                    for jb in range(2):
                        for is_ in range(2):
                            for js_ in range(2):
                                i = 2 * ib + is_
                                j = 2 * jb + js_
                                U[i * 4 + j][c_idx] += A_block[t][ib][jb] * U_scalar[s][is_][js_]
                                
                                # V
                                # B_t[ib, jb] * B_scalar[s][jb, kb]?
                                # V[j, k]. j is row of B, k is col of B.
                                # j = 2 jb + js_? No.
                                # B is 4x4.
                                # V index: row j (0..3), col k (0..3).
                                # j corresponds to row of B.
                                # In A_block, jb is col of A block, which is row of B block.
                                # So j = 2 jb + js_?
                                # Wait. A is 4x4. B is 4x4.
                                # A_block[ib, jb] multiplies A[ib, jb].
                                # B_block[ib, jb] multiplies B[ib, jb].
                                # In product A B, row of B is col of A.
                                # So row index of B is jb.
                                # So j (row of B) = 2 jb + js_.
                                # k (col of B) = 2 kb + ks_.
                                # But B_scalar[s] is 2x2.
                                # B_scalar[s][jb, kb]?
                                # No, B_scalar[s] is the matrix B_s.
                                # B_s = sum B_block[ib, jb] B[ib, jb].
                                # B[ib, jb] is a 2x2 block.
                                # B[ib, jb][js_, ks_] = B[2 ib + js_, 2 jb + ks_].
                                # So V[j, k, c] where j = 2 ib + js_, k = 2 jb + ks_.
                                # V[j, k, c] += B_block[t][ib][jb] * V_scalar[s][js_][ks_]
                                
                                pass # V and W handled below

                for ib in range(2):
                    for jb in range(2):
                        for js_ in range(2):
                            for ks_ in range(2):
                                j = 2 * ib + js_
                                k = 2 * jb + ks_
                                V[j * 4 + k][c_idx] += B_block[t][ib][jb] * V_scalar[s][js_][ks_]
                                
                                # W
                                # C_block[t][ib, jb] * W_scalar[s][kb, ib]?
                                # C[ib, jb] is a 2x2 block.
                                # C[ib, jb][is_, js_] = C[2 ib + is_, 2 jb + js_].
                                # W[k, i]. k is col of C, i is row of C.
                                # k = 2 jb + ks_, i = 2 ib + is_.
                                # W[k, i, c] += C_block[t][ib][jb] * W_scalar[s][ks_][is_]
                                
                for ib in range(2):
                    for jb in range(2):
                        for is_ in range(2):
                            for ks_ in range(2):
                                i = 2 * ib + is_
                                k = 2 * jb + ks_
                                W[k * 4 + i][c_idx] += C_block[t][ib][jb] * W_scalar[s][ks_][is_]

        return U, V, W

    def make_strassen_4x4x5():
        """Construct rank-53 decomposition for 4x4x5."""
        rank = 53
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]

        U49, V49, W49 = make_strassen_4x4x4()
        
        for i in range(n*m):
            for c in range(49):
                U[i][c] = U49[i][c]
                W[c*4+i][c] = W49[c*4+i][c] # W49 is 16x49. W is 20x53.
                # W49[k*4+i]. k is 0..3.
                # W[k*n+i]. n=4.
                # So W[k*4+i][c] = W49[k*4+i][c].
        
        for j in range(m*4):
            for c in range(49):
                V[j][c] = V49[j][c]
                # V49[j*4+k]. j is 0..3, k is 0..3.
                # V is 4x5.
                # V[j*5+k].
                # V49 index j*4+k.
                # We need to map V49[j, k] to V[j, k] for k=0..3.
                # V[j, 4] is 0.
                pass

        # Remap V49 to V
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V[j*5 + k][c] = V49[j*4 + k][c]
        
        # Add 4 terms for the 5th column
        for j in range(4):
            c = 49 + j
            for i in range(4):
                U[i*4 + j][c] = 1
            V[j*5 + 4][c] = 1
            for i in range(4):
                W[4*4 + i][c] = 1

        return {"U": U, "V": V, "W": W}

    best = make_strassen_4x4x5()
    best_rank = 53

    if not verify(best_rank, best["U"], best["V"], best["W"]):
        # Fallback
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = r

    return best

