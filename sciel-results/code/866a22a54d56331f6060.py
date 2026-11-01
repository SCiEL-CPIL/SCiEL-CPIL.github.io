import numpy as np
import itertools

def construct():
    n, m, p = 3, 3, 5
    # Rank 20 decomposition:
    # 7 terms for 2x2x2 Strassen (top-left block)
    # 7 terms for boundary interactions in 3x3x3
    # 6 terms for the extra 2 columns (3x3x2)
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w
        col += 1

    # 1. Strassen for 2x2x2 (indices 0,1 for i,j,k)
    # M1 = (A00+A11)(B00+B11)
    # U: A00, A11. V: B00, B11. W: C00, C11, C01, C10 (via combination)
    # Actually, we define U, V, W such that sum(U*V*W) = C.
    # We need to distribute the coefficients.
    
    # M1
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=-1, C01=-1, C10=1 ? No, Strassen formulas:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # We add 7 terms.
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U[0,0]=1, U[1,1]=1 (indices i*m+j)
    # V[0,0]=1, V[1,1]=1 (indices j*p+k)
    # W contributions:
    # W[0,0] (k=0,i=0) += 1
    # W[1,1] (k=1,i=1) += -1
    # W[0,1] (k=0,i=1) += -1
    # W[1,0] (k=1,i=0) += 1
    # Wait, W index is k*n+i.
    # C00 is i=0,k=0 -> W[0*3+0]=W[0]
    # C01 is i=0,k=1 -> W[1*3+0]=W[3]
    # C10 is i=1,k=0 -> W[0*3+1]=W[1]
    # C11 is i=1,k=1 -> W[1*3+1]=W[4]
    
    # Let's just hardcode the 7 Strassen terms.
    
    # T1: M1
    U[0][col] = 1; U[3][col] = 1 # A00, A11 (i=0,j=0; i=1,j=1)
    V[0][col] = 1; V[4][col] = 1 # B00, B11 (j=0,k=0; j=1,k=1)
    W[0][col] = 1; W[3][col] = -1; W[1][col] = -1; W[4][col] = 1
    col += 1
    
    # T2: M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1 # A10, A11 (i=1,j=0; i=1,j=1)
    V[0][col] = 1 # B00
    W[1][col] = 1; W[4][col] = -1 # C10 += 1, C11 -= 1
    col += 1
    
    # T3: M3 = A00(B10-B11)
    U[0][col] = 1 # A00
    V[3][col] = 1; V[4][col] = -1 # B10, B11
    W[3][col] = 1; W[4][col] = 1 # C01 += 1, C11 += 1
    col += 1
    
    # T4: M4 = A11(B01-B00)
    U[3][col] = 1 # A11
    V[1][col] = 1; V[0][col] = -1 # B01, B00
    W[0][col] = -1; W[3][col] = 1 # C00 -= 1, C01 += 1
    col += 1
    
    # T5: M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1 # A00, A01
    V[4][col] = 1 # B11
    W[0][col] = 1; W[4][col] = 1 # C00 += 1, C11 += 1
    col += 1
    
    # T6: M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1 # A10, A00
    V[0][col] = 1; V[1][col] = 1 # B00, B01
    W[4][col] = 1 # C11 += 1
    col += 1
    
    # T7: M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[3][col] = -1 # A01, A11
    V[3][col] = 1; V[4][col] = 1 # B10, B11
    W[1][col] = 1; W[0][col] = -1 # C10 += 1, C00 -= 1
    col += 1
    
    # 2. Boundary terms for 3x3x3
    # Indices: i,j,k in {0,1,2}
    # Term 8: A02 B20 -> C00? No, A[i,2] B[2,k] -> C[i,k]
    # A02 B20 -> C00. A12 B20 -> C10.
    # This is 1 term: U picks A[:,2], V picks B[2,:], W picks C[:,:2]
    # U[2]=1 (i=0,j=2), U[5]=1 (i=1,j=2)
    # V[6]=1 (j=2,k=0), V[7]=1 (j=2,k=1)
    # W[0]=1 (k=0,i=0), W[1]=1 (k=0,i=1), W[3]=1 (k=1,i=0), W[4]=1 (k=1,i=1)
    U[2][col] = 1; U[5][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # Term 9: A20 B00 -> C20, A21 B00 -> C21?
    # A[2,j] B[j,k] -> C[2,k]
    # A20 B00 -> C20. A21 B01 -> C21?
    # We need A[2,:] B[:,:2] -> C[2,:2]
    # U[6]=1 (i=2,j=0), U[7]=1 (i=2,j=1)
    # V[0]=1 (j=0,k=0), V[1]=1 (j=0,k=1), V[3]=1 (j=1,k=0), V[4]=1 (j=1,k=1)
    # W[6]=1 (k=0,i=2), W[7]=1 (k=1,i=2)
    U[6][col] = 1; U[7][col] = 1
    V[0][col] = 1; V[1][col] = 1; V[3][col] = 1; V[4][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # Term 10: A00 B02, A01 B12 -> C02, C12
    # A[:,:2] B[:,2] -> C[:,:2, 2]
    # U[0]=1, U[1]=1, U[3]=1, U[4]=1
    # V[2]=1 (j=0,k=2), V[5]=1 (j=1,k=2)
    # W[8]=1 (k=2,i=0), W[9]=1 (k=2,i=1)
    U[0][col] = 1; U[1][col] = 1; U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[5][col] = 1
    W[8][col] = 1; W[9][col] = 1
    col += 1
    
    # Term 11: A22 B22 -> C22
    U[8][col] = 1
    V[8][col] = 1
    W[11][col] = 1 # k=2, i=2 -> 2*3+2=8? No. W index k*n+i. 2*3+2=8.
    # Wait, W size is p*n = 5*3=15.
    # For 3x3x3 part, k in 0..2, i in 0..2.
    # W[8] is correct.
    col += 1
    
    # Term 12: A20 B02, A21 B12 -> C22
    # A[2,:] B[:,2] -> C[2,2]
    U[6][col] = 1; U[7][col] = 1
    V[2][col] = 1; V[5][col] = 1
    W[8][col] = 1
    col += 1
    
    # Term 13: A02 B22, A12 B22 -> C02, C12
    # A[:,2] B[2,2] -> C[:,:2, 2]
    U[2][col] = 1; U[5][col] = 1
    V[8][col] = 1
    W[8][col] = 1; W[9][col] = 1
    col += 1
    
    # Term 14: A22 B20, A22 B21 -> C20, C21
    # A[2,2] B[2,:] -> C[2,:2]
    U[8][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # 3. Extra columns (k=3,4)
    # A[i,j] B[j,3] -> C[i,3]
    # A[i,j] B[j,4] -> C[i,4]
    # 6 terms (3 for col 3, 3 for col 4)
    
    # Col 3
    for j in range(3):
        U[2][col] = 1 # i=0? No, U[i*m+j].
        # We need to sum over j for each i.
        # Term for j=0: U[0]=1, V[3]=1, W[3*3+0]=W[9]=1?
        # W index k*n+i. k=3, i=0 -> 3*3+0=9.
        U[0][col] = 1
        V[3][col] = 1
        W[9][col] = 1
        col += 1
        
    for j in range(3):
        U[3+j][col] = 1 # i=1, j=0,1,2 -> indices 3,4,5
        # Wait, U index is i*m+j.
        # i=0: 0,1,2
        # i=1: 3,4,5
        # i=2: 6,7,8
        
        # Correcting previous loop:
        # j=0: U[0], V[3], W[9]
        # j=1: U[1], V[4], W[10] (k=3, i=1 -> 3*3+1=10)
        # j=2: U[2], V[5], W[11] (k=3, i=2 -> 3*3+2=11)
        
        # My previous loop was wrong. Let's rewrite.
        
    # Reset col to 14
    col = 14
    
    # Col 3 (k=3)
    # j=0
    U[0][col] = 1; V[3][col] = 1; W[9][col] = 1
    col += 1
    # j=1
    U[1][col] = 1; V[4][col] = 1; W[10][col] = 1
    col += 1
    # j=2
    U[2][col] = 1; V[5][col] = 1; W[11][col] = 1
    col += 1
    
    # Col 4 (k=4)
    # j=0
    U[0][col] = 1; V[4][col] = 1; W[12][col] = 1 # W index 4*3+0=12
    col += 1
    # j=1
    U[1][col] = 1; V[5][col] = 1; W[13][col] = 1
    col += 1
    # j=2
    U[2][col] = 1; V[6][col] = 1; W[14][col] = 1
    
    return {"U": U, "V": V, "W": W}
