import time
import numpy as np

def _build_T(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _build_strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication (rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: T[i*m+j, j*p+k, k*n+i] = 1
    For 2x2: n=m=p=2
    U index: i*m+j = i*2+j
    V index: j*p+k = j*2+k
    W index: k*n+i = k*2+i
    """
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: A-part, shape (4,7) where row = i*2+j
    # V: B-part, shape (4,7) where row = j*2+k
    # W: C-part, shape (4,7) where row = k*2+i
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Term 1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> rows 0,3
    U[0, 0] = 1; U[3, 0] = 1
    # V: B00=1, B11=1 -> rows 0,3
    V[0, 0] = 1; V[3, 0] = 1
    # W: C00+=1, C11+=1 -> W rows: C00: k=0,i=0 -> row 0; C11: k=1,i=1 -> row 3
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10+=1, C11-=1 -> C10: k=0,i=1 -> row 2; C11: k=1,i=1 -> row 3
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C01+=1, C11+=1 -> C01: k=1,i=0 -> row 1; C11: k=1,i=1 -> row 3
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C00+=1, C10+=1 -> C00: row 0; C10: row 2
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00-=1, C01+=1 -> C00: row 0; C01: row 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11+=1 -> row 3
    W[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00+=1 -> row 0
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """Recursive Strassen: 4x4 via 7 block-mults of 2x2, each via Strassen 2x2.
    Returns U, V, W as lists of lists, rank 49."""
    # First get the 2x2 Strassen decomposition
    U2, V2, W2 = _build_strassen_2x2()
    
    # Now for 4x4, we partition into 2x2 blocks.
    # The 4x4 Strassen operates on 2x2 blocks.
    # Let A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # Similarly B, C.
    
    # Strassen on the 2x2 block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Ml is a 2x2 matrix product of 2x2 matrices.
    # We decompose each Ml = Xl * Yl where Xl, Yl are 2x2 matrices
    # (linear combos of A-blocks / B-blocks).
    # Then each Xl*Yl uses Strassen 2x2: 7 scalar multiplications.
    
    # Block offsets for 4x4:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # For each of the 7 Ml, define Xl and Yl as 2x2 matrices of (block, sign)
    # Xl[r][c] = sum of (sign * A_block[r][c])
    
    # M1: X = A00+A11, Y = B00+B11
    # X[r][c] = A00[r][c] + A11[r][c]
    # Y[r][c] = B00[r][c] + B11[r][c]
    
    # I'll define the 7 (X, Y) pairs where each entry is a list of (block_idx, sign)
    # block_idx: 0=A00/B00, 1=A01/B01, 2=A10/B10, 3=A11/B11
    
    Ms = {
        0: {  # M1
            'X': [[[(0,1),(3,1)],[(0,1),(3,1)]],
                  [[(0,1),(3,1)],[(0,1),(3,1)]]],
            'Y': [[[(0,1),(3,1)],[(0,1),(3,1)]],
                  [[(0,1),(3,1)],[(0,1),(3,1)]]],
            'C': [((0,0),1), ((1,1),1)]  # contributes to C00 with +1, C11 with +1
        },
        1: {  # M2
            'X': [[[(2,1),(3,1)],[(2,1),(3,1)]],
                  [[(2,1),(3,1)],[(2,1),(3,1)]]],
            'Y': [[[(0,1)],[(0,1)]],
                  [[(0,1)],[(0,1)]]],
            'C': [((1,0),1), ((1,1),-1)]
        },
        2: {  # M3
            'X': [[[(0,1)],[(0,1)]],
                  [[(0,1)],[(0,1)]]],
            'Y': [[[(1,1),(3,-1)],[(1,1),(3,-1)]],
                  [[(1,1),(3,-1)],[(1,1),(3,-1)]]],
            'C': [((0,1),1), ((1,1),1)]
        },
        3: {  # M4
            'X': [[[(3,1)],[(3,1)]],
                  [[(3,1)],[(3,1)]]],
            'Y': [[[(2,1),(0,-1)],[(2,1),(0,-1)]],
                  [[(2,1),(0,-1)],[(2,1),(0,-1)]]],
            'C': [((0,0),1), ((1,0),1)]
        },
        4: {  # M5
            'X': [[[(0,1),(1,1)],[(0,1),(1,1)]],
                  [[(0,1),(1,1)],[(0,1),(1,1)]]],
            'Y': [[[(3,1)],[(3,1)]],
                  [[(3,1)],[(3,1)]]],
            'C': [((0,0),-1), ((0,1),1)]
        },
        5: {  # M6
            'X': [[[(2,1),(0,-1)],[(2,1),(0,-1)]],
                  [[(2,1),(0,-1)],[(2,1),(0,-1)]]],
            'Y': [[[(0,1),(1,1)],[(0,1),(1,1)]],
                  [[(0,1),(1,1)],[(0,1),(1,1)]]],
            'C': [((1,1),1)]
        },
        6: {  # M7
            'X': [[[(1,1),(3,-1)],[(1,1),(3,-1)]],
                  [[(1,1),(3,-1)],[(1,1),(3,-1)]]],
            'Y': [[[(2,1),(3,1)],[(2,1),(3,1)]],
                  [[(2,1),(3,1)],[(2,1),(3,1)]]],
            'C': [((0,0),1)]
        },
    }
    
    # Block offsets: block b -> (row_offset, col_offset)
    # 0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)
    block_off = {(0): (0,0), (1): (0,2), (2): (2,0), (3): (2,2)}
    
    # For each Ml, for each of the 7 Strassen 2x2 terms (r,d,c in the 2x2 product),
    # and for each C-block it contributes to:
    # Term: Xl[r][d] * Yl[d][c] contributes to C_l[r][c]
    # C_l is a 2x2 block of the final C at position (bi,bj) with sign_w
    
    rank = 7 * 7  # 49
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    
    col = 0
    for l in range(7):
        Xl = Ms[l]['X']
        Yl = Ms[l]['Y']
        C_contrib = Ms[l]['C']
        
        # Use Strassen 2x2 decomposition for the 2x2 product Xl * Yl
        # The 7 terms of Strassen 2x2:
        # For a 2x2 matrix X and Y, Strassen gives 7 products:
        # P1 = (X00+X11)(Y00+Y11)
        # P2 = (X10+X11)Y00
        # P3 = X00(Y01-Y11)
        # P4 = X11(Y10-Y00)
        # P5 = (X00+X01)Y11
        # P6 = (X10-X00)(Y00+Y01)
        # P7 = (X01-X11)(Y10+Y11)
        
        # I need to combine the "outer" Strassen (block level) with the "inner" Strassen (2x2 level).
        # 
        # Actually, let me think differently. The 2x2 product Xl * Yl where Xl, Yl are 2x2 matrices
        # with entries that are linear combos of A/B entries.
        # 
        # I'll use the Strassen 2x2 U2, V2, W2 directly.
        # For the inner Strassen, the "A" matrix is Xl (2x2) and "B" matrix is Yl (2x2).
        # U2[i*2+j, c] gives the coefficient of A_entry(i,j) in the c-th Strassen term.
        # V2[j*2+k, c] gives the coefficient of B_entry(j,k) in the c-th Strassen term.
        # W2[k*2+i, c] gives the coefficient for C_entry(k,i) from the c-th term.
        
        for c_inner in range(7):
            # For each entry of Xl (which is a 2x2 matrix), multiply by U2 coefficient
            # Xl[r][d] is a linear combo of A-block entries
            # U2[i*2+j, c_inner] is the coefficient for A[i][j] in term c_inner
            
            for r in range(2):
                for d in range(2):
                    # Xl[r][d] contributes to A-position (r,d) in the inner 2x2
                    coeff_a = U2[r*2+d, c_inner]
                    if coeff_a == 0:
                        continue
                    for (blk, sign) in Xl[r][d]:
                        ro, co = block_off[blk]
                        a_row = ro + r
                        a_col = co + d
                        U[a_row*4 + a_col, col] += coeff_a * sign
            
            for d in range(2):
                for c in range(2):
                    # Yl[d][c] contributes to B-position (d,c) in the inner 2x2
                    coeff_b = V2[d*2+c, c_inner]
                    if coeff_b == 0:
                        continue
                    for (blk, sign) in Yl[d][c]:
                        ro, co = block_off[blk]
                        b_row = ro + d
                        b_col = co + c
                        V[b_row*4 + b_col, col] += coeff_b * sign
            
            for r in range(2):
                for c in range(2):
                    # C_l[r][c] gets contribution from W2[c*2+r, c_inner]
                    coeff_c = W2[c*2+r, c_inner]
                    if coeff_c == 0:
                        continue
                    for (cbi, cbj), sign_w in C_contrib:
                        w_row = cbi * 2 + r
                        w_col = cbj * 2 + c
                        # W index: k*n + i = w_col*4 + w_row
                        W[w_col*4 + w_row, col] += coeff_c * sign_w
            
            col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify(U, V, W, T):
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    rank = U.shape[1]
    T_recon = np.zeros(T.shape, dtype=np.int64)
    for c in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.array_equal(T_recon, T)

def construct():
    n, m, p = 4, 4, 4
    T = _build_T(n, m, p)
    
    U, V, W = _build_strassen_4x4()
    
    if _verify(U, V, W, T):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: schoolbook
        rank = 64
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
