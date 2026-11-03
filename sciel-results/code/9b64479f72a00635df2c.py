import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """
    Verify exact decomposition using the trace convention W[k*n+i, r].
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    """
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Pre-compute to speed up verification slightly
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i, j, k] == 1 for all i, j, k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    s += u_row[c] * v_row[c] * w_row[c]
                if s != 1:
                    return False
    return True

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 multiplication.
    Rank 7.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    # Strassen factors:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a11-a10)(b00+b10)
    # m7 = (a00-a01)(b01+b11)
    
    # U corresponds to A factors
    # V corresponds to B factors
    # W corresponds to C factors (reconstruction)
    
    # Let's define the 7 products explicitly.
    # Product c computes: sum U[i,j,c] A[i,j] * V[j,k,c] B[j,k]
    # Wait, the tensor is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Standard Strassen is for C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] is not quite C[i,k].
    # The tensor M_{n,m,p} has entries M[i,j,k] = 1.
    # The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T[i,j,k] = 1 means sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # This implies that for a fixed c, the term is U[i,j,c] A[i,j] * V[j,k,c] B[j,k] * W[k,i,c].
    # Wait, W is on the "output" side.
    # Usually, decomposition is T = sum_c u_c \otimes v_c \otimes w_c.
    # Here T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # This matches if we view W as acting on the output indices (k,i) -> C[i,k].
    # Actually, the index order in W is [k*n + i]. So W[k,i,c] is the coefficient for C[i,k] from product c.
    
    # Let's construct the 7 Strassen terms.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # U[i,j,c]: coefficient for A[i,j] in product c
    # V[j,k,c]: coefficient for B[j,k] in product c
    # W[k,i,c]: coefficient for C[i,k] in product c
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: m1 = (a00+a11)(b00+b11)
    # U[0,0]=1, U[1,1]=1
    # V[0,0]=1, V[1,1]=1
    # C contribution: c00 += m1, c01 += m1, c10 += m1, c11 += m1
    # Wait, m1 contributes to all?
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Let's map carefully.
    # m1 = (A00+A11)(B00+B11)
    # U[0,0,0]=1, U[1,1,0]=1
    # V[0,0,0]=1, V[1,1,0]=1
    # W[0,0,0]=1 (C00), W[0,1,0]=1 (C01), W[1,0,0]=1 (C10), W[1,1,0]=1 (C11)
    # Wait, W index is [k*n+i]. n=2.
    # C00 is i=0, k=0 -> idx 0*2+0=0.
    # C01 is i=0, k=1 -> idx 1*2+0=2.
    # C10 is i=1, k=0 -> idx 0*2+1=1.
    # C11 is i=1, k=1 -> idx 1*2+1=3.
    
    # Let's re-verify Strassen formulas:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # m1: (A00+A11)(B00+B11)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[2][0]=1; W[1][0]=1; W[3][0]=1
    
    # m2: (A10+A11)B00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1 # C10 += m2
    W[3][1]=-1 # C11 -= m2
    
    # m3: A00(B01-B11)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[2][2]=1 # C01 += m3
    W[3][2]=1 # C11 += m3
    
    # m4: A11(B10-B00)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1 # C00 += m4
    W[1][3]=1 # C10 += m4
    
    # m5: (A00+A10)B11
    U[0][4]=1; U[2][4]=1
    V[3][4]=1
    W[0][4]=-1 # C00 -= m5
    W[2][4]=1 # C01 += m5
    
    # m6: (A11-A10)(B00+B10)
    U[3][5]=1; U[2][5]=-1
    V[0][5]=1; V[2][5]=1
    W[3][5]=1 # C11 += m6
    
    # m7: (A00-A01)(B01+B11)
    U[0][6]=1; U[1][6]=-1
    V[1][6]=1; V[3][6]=1
    W[0][6]=1 # C00 += m7
    
    return U, V, W

def _construct_2x3x5_strassen():
    """
    Construct a rank-28 decomposition for 2x3 * 3x5.
    
    Decomposition strategy:
    Split A (2x3) into A_L (2x2) and A_R (2x1).
    Split B (3x5) into B_T (2x5) and B_B (1x5).
    
    C = A_L B_T + A_R B_B
    
    A_L B_T is 2x2 * 2x5.
    A_R B_B is 2x1 * 1x5.
    
    We can treat 2x2 * 2x5 as two 2x2 * 2x2 multiplications?
    No, 2x2 * 2x5 can be done by splitting B_T into B_T1 (2x2) and B_T2 (2x3)?
    Or just use the general tensor product structure.
    
    Let's use the property that rank(M_{n,m,p}) <= rank(M_{n,m',p}) + rank(M_{n,m-m',p}).
    Here m=3. Split into m'=2 and m''=1.
    rank(M_{2,3,5}) <= rank(M_{2,2,5}) + rank(M_{2,1,5}).
    
    rank(M_{2,1,5}) = 2*1*5 = 10 (schoolbook).
    rank(M_{2,2,5}) <= rank(M_{2,2,2}) * ceil(5/2)? No.
    M_{2,2,5} can be seen as 2x2 * 2x5.
    We can split the 5 columns into 2+2+1?
    M_{2,2,5} = M_{2,2,2} \oplus M_{2,2,2} \oplus M_{2,2,1}?
    If so, rank = 7 + 7 + 4 = 18.
    Total rank = 18 + 10 = 28.
    
    Let's verify this construction.
    
    A = [[a00, a01, a02], [a10, a11, a12]]
    B = [[b00, b01, b02, b03, b04],
         [b10, b11, b12, b13, b14],
         [b20, b21, b22, b23, b24]]
         
    A_L = [[a00, a01], [a10, a11]]
    A_R = [[a02], [a12]]
    
    B_T = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14]]
    B_B = [[b20, b21, b22, b23, b24]]
    
    C = A_L B_T + A_R B_B
    
    Part 1: A_L B_T (2x2 * 2x5)
    Split B_T into B_T1 (2x2) and B_T2 (2x3).
    B_T1 = [[b00, b01], [b10, b11]]
    B_T2 = [[b02, b03, b04], [b12, b13, b14]]
    
    A_L B_T1 (2x2 * 2x2): Rank 7 (Strassen)
    A_L B_T2 (2x2 * 2x3):
    Split B_T2 into B_T2a (2x2) and B_T2b (2x1).
    B_T2a = [[b02, b03], [b12, b13]]
    B_T2b = [[b04], [b14]]
    
    A_L B_T2a (2x2 * 2x2): Rank 7 (Strassen)
    A_L B_T2b (2x2 * 2x1):
    This is equivalent to 2x2 * 2x1.
    We can view this as 2x1 * 1x1 twice? No.
    A_L is 2x2. B_T2b is 2x1.
    C_part = A_L B_T2b.
    c00 = a00 b04 + a01 b14
    c10 = a10 b04 + a11 b14
    This is rank 4?
    c00 = a00 b04 + a01 b14
    c10 = a10 b04 + a11 b14
    These are 4 distinct bilinear terms if we don't share?
    Actually, we can use rank 4 schoolbook for 2x2*2x1.
    Or we can use the fact that M_{2,2,1} has rank 4.
    
    So Part 1 (2x2*2x5) rank = 7 (for 2x2) + 7 (for 2x2) + 4 (for 2x1) = 18.
    
    Part 2: A_R B_B (2x1 * 1x5)
    A_R = [[a02], [a12]]
    B_B = [[b20, b21, b22, b23, b24]]
    C_part = A_R B_B
    c00 = a02 b20
    c01 = a02 b21
    c02 = a02 b22
    c03 = a02 b23
    c04 = a02 b24
    c10 = a12 b20
    c11 = a12 b21
    c12 = a12 b22
    c13 = a12 b23
    c14 = a12 b24
    This is 10 terms. Rank 10.
    
    Total Rank = 18 + 10 = 28.
    
    Now we need to map these to the global U, V, W matrices.
    
    Global Dimensions: n=2, m=3, p=5.
    U: 2*3 = 6 rows.
    V: 3*5 = 15 rows.
    W: 5*2 = 10 rows.
    
    We will build U, V, W with 28 columns.
    
    Mapping:
    A indices: (i, j) where i in {0,1}, j in {0,1,2}.
    B indices: (j, k) where j in {0,1,2}, k in {0,1,2,3,4}.
    C indices: (i, k) where i in {0,1}, k in {0,1,2,3,4}.
    
    Term 1: A_L B_T1 (2x2 * 2x2)
    A_L uses A[0,0], A[0,1], A[1,0], A[1,1].
    B_T1 uses B[0,0], B[0,1], B[1,0], B[1,1].
    Output C[0,0], C[0,1], C[1,0], C[1,1].
    
    We have Strassen U, V, W for 2x2.
    Let's call them Us, Vs, Ws.
    Us[i,j,c] corresponds to A_L[i,j].
    Vs[j,k,c] corresponds to B_T1[j,k].
    Ws[k,i,c] corresponds to C[0+i, 0+k].
    
    So for c in 0..6:
    U[0,0,c] = Us[0,0,c]  # A[0,0]
    U[0,1,c] = Us[0,1,c]  # A[0,1]
    U[1,0,c] = Us[1,0,c]  # A[1,0]
    U[1,1,c] = Us[1,1,c]  # A[1,1]
    
    V[0,0,c] = Vs[0,0,c]  # B[0,0]
    V[0,1,c] = Vs[0,1,c]  # B[0,1]
    V[1,0,c] = Vs[1,0,c]  # B[1,0]
    V[1,1,c] = Vs[1,1,c]  # B[1,1]
    
    W[0,0,c] = Ws[0,0,c]  # C[0,0] (k=0, i=0)
    W[1,0,c] = Ws[1,0,c]  # C[0,1] (k=1, i=0)
    W[0,1,c] = Ws[0,1,c]  # C[1,0] (k=0, i=1)
    W[1,1,c] = Ws[1,1,c]  # C[1,1] (k=1, i=1)
    
    Term 2: A_L B_T2a (2x2 * 2x2)
    B_T2a uses B[0,2], B[0,3], B[1,2], B[1,3].
    Output C[0,2], C[0,3], C[1,2], C[1,3].
    Offset columns 7..13.
    
    For c in 7..13 (local c' = c-7):
    U[0,0,c] = Us[0,0,c']
    U[0,1,c] = Us[0,1,c']
    U[1,0,c] = Us[1,0,c']
    U[1,1,c] = Us[1,1,c']
    
    V[0,2,c] = Vs[0,0,c']  # B[0,2]
    V[0,3,c] = Vs[0,1,c']  # B[0,3]
    V[1,2,c] = Vs[1,0,c']  # B[1,2]
    V[1,3,c] = Vs[1,1,c']  # B[1,3]
    
    W[2,0,c] = Ws[0,0,c']  # C[0,2] (k=2, i=0)
    W[3,0,c] = Ws[1,0,c']  # C[0,3] (k=3, i=0)
    W[2,1,c] = Ws[0,1,c']  # C[1,2] (k=2, i=1)
    W[3,1,c] = Ws[1,1,c']  # C[1,3] (k=3, i=1)
    
    Term 3: A_L B_T2b (2x2 * 2x1)
    B_T2b uses B[0,4], B[1,4].
    Output C[0,4], C[1,4].
    Offset columns 14..17.
    Rank 4. Schoolbook.
    c=14: A[0,0] B[0,4] -> C[0,4]
    c=15: A[0,1] B[1,4] -> C[0,4]
    c=16: A[1,0] B[0,4] -> C[1,4]
    c=17: A[1,1] B[1,4] -> C[1,4]
    
    U[0,0,14]=1; V[0,4,14]=1; W[4,0,14]=1
    U[0,1,15]=1; V[1,4,15]=1; W[4,0,15]=1
    U[1,0,16]=1; V[0,4,16]=1; W[4,1,16]=1
    U[1,1,17]=1; V[1,4,17]=1; W[4,1,17]=1
    
    Term 4: A_R B_B (2x1 * 1x5)
    A_R uses A[0,2], A[1,2].
    B_B uses B[2,0]..B[2,4].
    Output C[0,0]..C[0,4], C[1,0]..C[1,4].
    Offset columns 18..27.
    Rank 10. Schoolbook.
    
    c=18: A[0,2] B[2,0] -> C[0,0]
    c=19: A[0,2] B[2,1] -> C[0,1]
    c=20: A[0,2] B[2,2] -> C[0,2]
    c=21: A[0,2] B[2,3] -> C[0,3]
    c=22: A[0,2] B[2,4] -> C[0,4]
    c=23: A[1,2] B[2,0] -> C[1,0]
    c=24: A[1,2] B[2,1] -> C[1,1]
    c=25: A[1,2] B[2,2] -> C[1,2]
    c=26: A[1,2] B[2,3] -> C[1,3]
    c=27: A[1,2] B[2,4] -> C[1,4]
    
    U[0,2,18]=1; V[2,0,18]=1; W[0,0,18]=1
    U[0,2,19]=1; V[2,1,19]=1; W[1,0,19]=1
    U[0,2,20]=1; V[2,2,20]=1; W[2,0,20]=1
    U[0,2,21]=1; V[2,3,21]=1; W[3,0,21]=1
    U[0,2,22]=1; V[2,4,22]=1; W[4,0,22]=1
    
    U[1,2,23]=1; V[2,0,23]=1; W[0,1,23]=1
    U[1,2,24]=1; V[2,1,24]=1; W[1,1,24]=1
    U[1,2,25]=1; V[2,2,25]=1; W[2,1,25]=1
    U[1,2,26]=1; V[2,3,26]=1; W[3,1,26]=1
    U[1,2,27]=1; V[2,4,27]=1; W[4,1,27]=1
    
    This should yield a valid rank-28 decomposition.
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    Us, Vs, Ws = _strassen_2x2()
    
    # Term 1: A_L B_T1 (cols 0-6)
    for c in range(7):
        U[0][c] = Us[0][c]
        U[1][c] = Us[1][c]
        U[2][c] = Us[2][c] # A[1,0] is index 1*2+0 = 2? 
        # Wait. U index is i*m + j.
        # i=0, j=0 -> 0. i=0, j=1 -> 1. i=1, j=0 -> 2. i=1, j=1 -> 3.
        # So U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11.
        
        # Us indices:
        # Us[0] is A_L[0,0] -> A[0,0] -> U[0]
        # Us[1] is A_L[0,1] -> A[0,1] -> U[1]
        # Us[2] is A_L[1,0] -> A[1,0] -> U[2]
        # Us[3] is A_L[1,1] -> A[1,1] -> U[3]
        
        # So:
        U[0][c] = Us[0][c]
        U[1][c] = Us[1][c]
        U[2][c] = Us[2][c]
        U[3][c] = Us[3][c]
        
        # V indices:
        # Vs[0] is B_T1[0,0] -> B[0,0] -> V[0*5+0] = V[0]
        # Vs[1] is B_T1[0,1] -> B[0,1] -> V[0*5+1] = V[1]
        # Vs[2] is B_T1[1,0] -> B[1,0] -> V[1*5+0] = V[5]
        # Vs[3] is B_T1[1,1] -> B[1,1] -> V[1*5+1] = V[6]
        
        V[0][c] = Vs[0][c]
        V[1][c] = Vs[1][c]
        V[5][c] = Vs[2][c]
        V[6][c] = Vs[3][c]
        
        # W indices:
        # Ws[0] is C[0,0] -> k=0, i=0 -> W[0*2+0] = W[0]
        # Ws[1] is C[0,1] -> k=1, i=0 -> W[1*2+0] = W[2]
        # Ws[2] is C[1,0] -> k=0, i=1 -> W[0*2+1] = W[1]
        # Ws[3] is C[1,1] -> k=1, i=1 -> W[1*2+1] = W[3]
        
        W[0][c] = Ws[0][c]
        W[2][c] = Ws[1][c]
        W[1][c] = Ws[2][c]
        W[3][c] = Ws[3][c]
        
    # Term 2: A_L B_T2a (cols 7-13)
    for c in range(7, 14):
        c_local = c - 7
        U[0][c] = Us[0][c_local]
        U[1][c] = Us[1][c_local]
        U[2][c] = Us[2][c_local]
        U[3][c] = Us[3][c_local]
        
        # B_T2a:
        # B[0,2] -> V[2]
        # B[0,3] -> V[3]
        # B[1,2] -> V[7]
        # B[1,3] -> V[8]
        
        V[2][c] = Vs[0][c_local]
        V[3][c] = Vs[1][c_local]
        V[7][c] = Vs[2][c_local]
        V[8][c] = Vs[3][c_local]
        
        # C[0,2] -> k=2, i=0 -> W[2*2+0] = W[4]
        # C[0,3] -> k=3, i=0 -> W[3*2+0] = W[6]
        # C[1,2] -> k=2, i=1 -> W[2*2+1] = W[5]
        # C[1,3] -> k=3, i=1 -> W[3*2+1] = W[7]
        
        W[4][c] = Ws[0][c_local]
        W[6][c] = Ws[1][c_local]
        W[5][c] = Ws[2][c_local]
        W[7][c] = Ws[3][c_local]
        
    # Term 3: A_L B_T2b (cols 14-17)
    # c=14: A[0,0] B[0,4] -> C[0,4]
    U[0][14] = 1; V[4][14] = 1; W[8][14] = 1
    # c=15: A[0,1] B[1,4] -> C[0,4]
    U[1][15] = 1; V[9][15] = 1; W[8][15] = 1
    # c=16: A[1,0] B[0,4] -> C[1,4]
    U[2][16] = 1; V[4][16] = 1; W[9][16] = 1
    # c=17: A[1,1] B[1,4] -> C[1,4]
    U[3][17] = 1; V[9][17] = 1; W[9][17] = 1
    
    # Term 4: A_R B_B (cols 18-27)
    # c=18: A[0,2] B[2,0] -> C[0,0]
    U[4][18] = 1; V[10][18] = 1; W[0][18] = 1
    # c=19: A[0,2] B[2,1] -> C[0,1]
    U[4][19] = 1; V[11][19] = 1; W[2][19] = 1
    # c=20: A[0,2] B[2,2] -> C[0,2]
    U[4][20] = 1; V[12][20] = 1; W[4][20] = 1
    # c=21: A[0,2] B[2,3] -> C[0,3]
    U[4][21] = 1; V[13][21] = 1; W[6][21] = 1
    # c=22: A[0,2] B[2,4] -> C[0,4]
    U[4][22] = 1; V[14][22] = 1; W[8][22] = 1
    
    # c=23: A[1,2] B[2,0] -> C[1,0]
    U[5][23] = 1; V[10][23] = 1; W[1][23] = 1
    # c=24: A[1,2] B[2,1] -> C[1,1]
    U[5][24] = 1; V[11][24] = 1; W[3][24] = 1
    # c=25: A[1,2] B[2,2] -> C[1,2]
    U[5][25] = 1; V[12][25] = 1; W[5][25] = 1
    # c=26: A[1,2] B[2,3] -> C[1,3]
    U[5][26] = 1; V[13][26] = 1; W[7][26] = 1
    # c=27: A[1,2] B[2,4] -> C[1,4]
    U[5][27] = 1; V[14][27] = 1; W[9][27] = 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 30
    
    # Try the structured Strassen-based rank 28 decomposition
    try:
        candidate = _construct_2x3x5_strassen()
        if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            return candidate
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
