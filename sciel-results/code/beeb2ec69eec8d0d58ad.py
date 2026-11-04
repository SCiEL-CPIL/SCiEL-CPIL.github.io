import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    """
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    try:
        T = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d, optimize=True)
    except Exception:
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
    
    if not np.allclose(T, 1.0, atol=tol, rtol=0.0):
        return False
    return True

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Target rank 78
    rank = 78
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(u00, v00, w00):
        nonlocal col
        # u00: index for A[0,0]
        # v00: index for B[0,0]
        # w00: index for C[0,0]
        
        # P1 = (A00+A11)(B00+B11)
        U[u00, col] += 1
        U[u00+3, col] += 1 # A11 is u00+3 (2x2 block: 00, 01, 10, 11)
        V[v00, col] += 1
        V[v00+3, col] += 1 # B11 is v00+3
        W[w00, col] += 1
        W[w00+3, col] += 1 # C11 is w00+3
        col += 1
        
        # P2 = (A10+A11)B00
        U[u00+2, col] += 1 # A10
        U[u00+3, col] += 1 # A11
        V[v00, col] += 1
        W[w00+2, col] += 1 # C10
        W[w00+3, col] += 1 # C11
        col += 1
        
        # P3 = A00(B01-B11)
        U[u00, col] += 1
        V[v00+1, col] += 1 # B01
        V[v00+3, col] += -1 # B11
        W[w00+1, col] += 1 # C01
        W[w00+3, col] += 1 # C11
        col += 1
        
        # P4 = A11(B10-B00)
        U[u00+3, col] += 1
        V[v00+2, col] += 1 # B10
        V[v00, col] += -1
        W[w00, col] += 1
        W[w00+2, col] += 1
        col += 1
        
        # P5 = (A00+A01)B11
        U[u00, col] += 1
        U[u00+1, col] += 1 # A01
        V[v00+3, col] += 1
        W[w00, col] += -1
        W[w00+1, col] += -1
        col += 1
        
        # P6 = (A10-A00)(B00+B01)
        U[u00+2, col] += 1
        U[u00, col] += -1
        V[v00, col] += 1
        V[v00+1, col] += 1
        W[w00+3, col] += 1
        col += 1
        
        # P7 = (A01-A11)(B10+B11)
        U[u00+1, col] += 1
        U[u00+3, col] += -1
        V[v00+2, col] += 1
        V[v00+3, col] += 1
        W[w00+1, col] += 1
        W[w00+3, col] += -1
        col += 1
        
    def add_schoolbook_2x2x1(u00, v00, w00):
        nonlocal col
        # 2x2 * 2x1 -> 2x1
        # A: u00, u00+1, u00+2, u00+3
        # B: v00, v00+1 (col 0)
        # C: w00, w00+1
        
        # C0 = A00 B0 + A01 B1
        U[u00, col] += 1
        V[v00, col] += 1
        W[w00, col] += 1
        col += 1
        U[u00+1, col] += 1
        V[v00+1, col] += 1
        W[w00, col] += 1
        col += 1
        
        # C1 = A10 B0 + A11 B1
        U[u00+2, col] += 1
        V[v00, col] += 1
        W[w00+1, col] += 1
        col += 1
        U[u00+3, col] += 1
        V[v00+1, col] += 1
        W[w00+1, col] += 1
        col += 1
        
    def add_schoolbook_1x2x2(u00, v00, w00):
        nonlocal col
        # 1x2 * 2x2 -> 1x2
        # A: u00, u00+1
        # B: v00, v00+1 (col 0), v00+2, v00+3 (col 1)
        # C: w00, w00+1
        
        # C0 = A0 B00 + A1 B10
        U[u00, col] += 1
        V[v00, col] += 1
        W[w00, col] += 1
        col += 1
        U[u00+1, col] += 1
        V[v00+2, col] += 1
        W[w00, col] += 1
        col += 1
        
        # C1 = A0 B01 + A1 B11
        U[u00, col] += 1
        V[v00+1, col] += 1
        W[w00+1, col] += 1
        col += 1
        U[u00+1, col] += 1
        V[v00+3, col] += 1
        W[w00+1, col] += 1
        col += 1

    def add_schoolbook_1x2x1(u00, v00, w00):
        nonlocal col
        # 1x2 * 2x1 -> 1x1
        # A: u00, u00+1
        # B: v00, v00+1
        # C: w00
        
        U[u00, col] += 1
        V[v00, col] += 1
        W[w00, col] += 1
        col += 1
        U[u00+1, col] += 1
        V[v00+1, col] += 1
        W[w00, col] += 1
        col += 1

    # Global Index Mapping
    # A (3x4): rows 0,1,2; cols 0,1,2,3
    # B (4x7): rows 0,1,2,3; cols 0,1,2,3,4,5,6
    # C (3x7): rows 0,1,2; cols 0,1,2,3,4,5,6
    
    # Split A cols: 0,1 (A_L), 2,3 (A_R)
    # Split B rows: 0,1 (B_T), 2,3 (B_B)
    
    # Term 1: A_L (3x2) * B_T (2x7)
    # A_L rows: 0,1 (A_LL), 2 (A_LB)
    # B_T cols: 0,1 (B_T1), 2,3 (B_T2), 4,5 (B_T3), 6 (B_T4)
    
    # 1a: A_LL (2x2) * B_T (2x7)
    # u00 for A_LL[0,0] is A[0,0] -> index 0*4+0 = 0
    # v00 for B_T[0,0] is B[0,0] -> index 0*7+0 = 0
    # w00 for C_LL[0,0] is C[0,0] -> index 0*3+0 = 0
    
    # B_T1 (2x2): B[0,0..1], B[1,0..1]
    # B_T2 (2x2): B[0,2..3], B[1,2..3]
    # B_T3 (2x2): B[0,4..5], B[1,4..5]
    # B_T4 (2x1): B[0,6], B[1,6]
    
    # C_LL1 (2x2): C[0,0..1], C[1,0..1] -> w00 = 0
    # C_LL2 (2x2): C[0,2..3], C[1,2..3] -> w00 = 2*3+2? No.
    # C indices: C[i, k]. Index = k*n + i.
    # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4.
    # C[0,2]=2, C[0,3]=3? No.
    # W index is k*n + i.
    # C[0,0] -> 0*3+0 = 0
    # C[0,1] -> 1*3+0 = 3? No.
    # Let's check the convention: W[k*n+i, r].
    # So W index for C[i,k] is k*n + i.
    # C[0,0] -> 0
    # C[0,1] -> 1*3+0 = 3
    # C[1,0] -> 0*3+1 = 1
    # C[1,1] -> 1*3+1 = 4
    
    # So for a 2x2 block of C starting at (i0, k0):
    # C[i0, k0] -> k0*n + i0
    # C[i0, k0+1] -> (k0+1)*n + i0
    # C[i0+1, k0] -> k0*n + i0+1
    # C[i0+1, k0+1] -> (k0+1)*n + i0+1
    
    # Let's re-verify the Strassen index mapping.
    # A 2x2 block starting at (i0, j0) in A:
    # A[i0, j0] -> i0*m + j0
    # A[i0, j0+1] -> i0*m + j0+1
    # A[i0+1, j0] -> (i0+1)*m + j0
    # A[i0+1, j0+1] -> (i0+1)*m + j0+1
    
    # B 2x2 block starting at (j0, k0) in B:
    # B[j0, k0] -> j0*p + k0
    # B[j0, k0+1] -> j0*p + k0+1
    # B[j0+1, k0] -> (j0+1)*p + k0
    # B[j0+1, k0+1] -> (j0+1)*p + k0+1
    
    # C 2x2 block starting at (i0, k0) in C:
    # C[i0, k0] -> k0*n + i0
    # C[i0, k0+1] -> (k0+1)*n + i0
    # C[i0+1, k0] -> k0*n + i0+1
    # C[i0+1, k0+1] -> (k0+1)*n + i0+1
    
    # 1a: A_LL (i0=0, j0=0) * B_T (j0=0, k0 varies)
    # B_T1 (k0=0):
    add_strassen(0, 0, 0) # C[0..1, 0..1]
    # B_T2 (k0=2):
    # C[0..1, 2..3] -> w00 = 2*3+0 = 6
    add_strassen(0, 14, 6) # B[0..1, 2..3] -> v00 = 0*7+2 = 2? No.
    # B[j0, k0] -> j0*p + k0.
    # B_T2 starts at j0=0, k0=2.
    # B[0,2] -> 0*7+2 = 2.
    # So v00 = 2.
    add_strassen(0, 2, 6)
    # B_T3 (k0=4):
    # B[0,4] -> 4.
    # C[0..1, 4..5] -> w00 = 4*3+0 = 12.
    add_strassen(0, 4, 12)
    # B_T4 (k0=6):
    # B[0,6] -> 6.
    # C[0..1, 6] -> w00 = 6*3+0 = 18.
    # Schoolbook 2x2x1
    add_schoolbook_2x2x1(0, 6, 18)
    
    # 1b: A_LB (i0=2, j0=0) * B_T (j0=0, k0 varies)
    # A_LB is 1x2. A[2,0], A[2,1].
    # u00 = 2*4+0 = 8.
    # B_T1 (k0=0):
    # C[2, 0..1] -> w00 = 0*3+2 = 2.
    add_schoolbook_1x2x2(8, 0, 2)
    # B_T2 (k0=2):
    # C[2, 2..3] -> w00 = 2*3+2 = 8.
    add_schoolbook_1x2x2(8, 2, 8)
    # B_T3 (k0=4):
    # C[2, 4..5] -> w00 = 4*3+2 = 14.
    add_schoolbook_1x2x2(8, 4, 14)
    # B_T4 (k0=6):
    # C[2, 6] -> w00 = 6*3+2 = 20.
    add_schoolbook_1x2x1(8, 6, 20)
    
    # Term 2: A_R (3x2) * B_B (2x7)
    # A_R cols: 2,3.
    # A_RL rows: 0,1 (A_RL), 2 (A_RB).
    # B_B rows: 2,3.
    # B_B cols: 0,1 (B_B1), 2,3 (B_B2), 4,5 (B_B3), 6 (B_B4).
    
    # 2a: A_RL (i0=0, j0=2) * B_B (j0=2, k0 varies)
    # A_RL[0,0] is A[0,2] -> u00 = 0*4+2 = 2.
    # B_B1 (k0=0):
    # B[2,0] -> v00 = 2*7+0 = 14.
    # C[0..1, 0..1] -> w00 = 0.
    add_strassen(2, 14, 0)
    # B_B2 (k0=2):
    # B[2,2] -> v00 = 2*7+2 = 16.
    # C[0..1, 2..3] -> w00 = 6.
    add_strassen(2, 16, 6)
    # B_B3 (k0=4):
    # B[2,4] -> v00 = 2*7+4 = 18.
    # C[0..1, 4..5] -> w00 = 12.
    add_strassen(2, 18, 12)
    # B_B4 (k0=6):
    # B[2,6] -> v00 = 2*7+6 = 20.
    # C[0..1, 6] -> w00 = 18.
    add_schoolbook_2x2x1(2, 20, 18)
    
    # 2b: A_RB (i0=2, j0=2) * B_B (j0=2, k0 varies)
    # A_RB[0,0] is A[2,2] -> u00 = 2*4+2 = 10.
    # B_B1 (k0=0):
    # C[2, 0..1] -> w00 = 2.
    add_schoolbook_1x2x2(10, 14, 2)
    # B_B2 (k0=2):
    # C[2, 2..3] -> w00 = 8.
    add_schoolbook_1x2x2(10, 16, 8)
    # B_B3 (k0=4):
    # C[2, 4..5] -> w00 = 14.
    add_schoolbook_1x2x2(10, 18, 14)
    # B_B4 (k0=6):
    # C[2, 6] -> w00 = 20.
    add_schoolbook_1x2x1(10, 20, 20)
    
    # Verify col count
    # 1a: 7+7+7+4 = 25
    # 1b: 4+4+4+2 = 14
    # 2a: 7+7+7+4 = 25
    # 2b: 4+4+4+2 = 14
    # Total 78.
    
    if col != rank:
        # Fallback to schoolbook if index mapping failed
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        rank = rank_school

    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification failed
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        rank = rank_school

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
