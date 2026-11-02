import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _construct_2x2x2_strassen():
    """
    Returns U, V, W for 2x2x2 matrix multiplication, rank 7.
    U: (2*2, 7) = (4, 7)
    V: (2*2, 7) = (4, 7)
    W: (2*2, 7) = (4, 7)
    Indices:
    U[i*2+j]
    V[j*2+k]
    W[k*2+i]
    """
    R = 7
    U = np.zeros((4, R), dtype=float)
    V = np.zeros((4, R), dtype=float)
    W = np.zeros((4, R), dtype=float)
    
    # A indices: 0:11, 1:12, 2:21, 3:22
    # B indices: 0:11, 1:12, 2:21, 3:22
    # C indices (k*2+i): 0:11(k=0,i=0), 1:21(k=0,i=1), 2:12(k=1,i=0), 3:22(k=1,i=1)
    
    # M1 = A11(B11+B22)
    U[0, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1 # C11
    W[3, 0] = 1 # C22
    
    # M2 = A22(B11-B22)
    U[3, 1] = 1
    V[0, 1] = 1
    V[3, 1] = -1
    W[0, 1] = -1 # C11
    W[3, 1] = 1  # C22
    
    # M3 = (A11+A22)B22
    U[0, 2] = 1
    U[3, 2] = 1
    V[3, 2] = 1
    W[0, 2] = -1 # C11
    W[3, 2] = 1  # C22
    
    # M4 = (A22-A11)B11
    U[3, 3] = 1
    U[0, 3] = -1
    V[0, 3] = 1
    W[0, 3] = 1  # C11
    W[1, 3] = 1  # C21? Wait. C21 = A21B11 + A22B21.
    # Let's check Strassen formulas carefully.
    # C11 = M1 + M4 - M3 + M7
    # C12 = M5 + M6
    # C21 = M5 + M2
    # C22 = M1 - M2 + M3 + M6
    
    # My W assignments above:
    # M1: W[0]=1, W[3]=1. (C11, C22)
    # M2: W[0]=-1, W[3]=1. (C11, C22)
    # M3: W[0]=-1, W[3]=1. (C11, C22)
    # M4: W[0]=1. Need W[1]=1 for C21?
    # C21 index is 1 (k=0, i=1).
    # C12 index is 2 (k=1, i=0).
    
    # Let's redo W assignments based on formulas.
    # C11 (idx 0): M1(1) + M4(1) - M3(1) + M7(1)
    # C12 (idx 2): M5(1) + M6(1)
    # C21 (idx 1): M5(1) + M2(1)
    # C22 (idx 3): M1(1) - M2(1) + M3(1) + M6(1)
    
    W[0, 0] = 1; W[3, 0] = 1
    W[0, 1] = -1; W[3, 1] = -1 # Wait, M2 in C22 is -1. In C21 is +1.
    # Formula: C21 = M5 + M2. So W[1, 1] = 1.
    # Formula: C22 = ... - M2 ... So W[3, 1] = -1.
    W[1, 1] = 1
    W[3, 1] = -1
    
    # M3: C11 -1, C22 +1.
    W[0, 2] = -1
    W[3, 2] = 1
    
    # M4: C11 +1.
    W[0, 3] = 1
    
    # M5 = (A11+A12)B21
    U[0, 4] = 1
    U[1, 4] = 1
    V[2, 4] = 1
    # C12 +1, C21 +1
    W[2, 4] = 1
    W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    # C12 +1, C22 +1
    W[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12+A22)(B21+B22)
    U[1, 6] = 1
    U[3, 6] = 1
    V[2, 6] = 1
    V[3, 6] = 1
    # C11 +1
    W[0, 6] = 1
    
    return U, V, W

def _construct_2x2x1_rank4():
    """
    Returns U, V, W for 2x2x1 matrix multiplication, rank 4.
    U: (4, 4)
    V: (2, 4)  # m=2, p=1 -> 2*1=2 rows.
    W: (2, 4)  # p=1, n=2 -> 1*2=2 rows. Index k*2+i. k=0. So indices 0,1.
    """
    R = 4
    U = np.zeros((4, R), dtype=float)
    V = np.zeros((2, R), dtype=float)
    W = np.zeros((2, R), dtype=float)
    
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    
    # M1 = A11 B1
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # M2 = A12 B2
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # M3 = A21 B1
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # M4 = A22 B2
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _construct_2x1x3_rank6():
    """
    Returns U, V, W for 2x1x3 matrix multiplication, rank 6.
    U: (2*1, 6) = (2, 6)
    V: (1*3, 6) = (3, 6)
    W: (3*2, 6) = (6, 6)
    """
    R = 6
    U = np.zeros((2, R), dtype=float)
    V = np.zeros((3, R), dtype=float)
    W = np.zeros((6, R), dtype=float)
    
    # C1k = A1 Bk
    # C2k = A2 Bk
    
    for k in range(3):
        # Term 2k: A1 Bk
        U[0, 2*k] = 1
        V[k, 2*k] = 1
        W[k*2+0, 2*k] = 1 # C1k (i=0, k=k) -> idx k*2+0
        
        # Term 2k+1: A2 Bk
        U[1, 2*k+1] = 1
        V[k, 2*k+1] = 1
        W[k*2+1, 2*k+1] = 1 # C2k (i=1, k=k) -> idx k*2+1
        
    return U, V, W

def _construct_2x2x3_rank11():
    """
    Combine 2x2x2 (rank 7) and 2x2x1 (rank 4) for 2x2x3.
    U: (4, 11)
    V: (6, 11)  # m=2, p=3 -> 6 rows.
    W: (6, 11)  # p=3, n=2 -> 6 rows.
    """
    R = 11
    U = np.zeros((4, R), dtype=float)
    V = np.zeros((6, R), dtype=float)
    W = np.zeros((6, R), dtype=float)
    
    U2, V2, W2 = _construct_2x2x2_strassen()
    U1, V1, W1 = _construct_2x2x1_rank4()
    
    # Part 1: k in {0,1} (indices 0,1 in V and W)
    U[:, :7] = U2
    # V2 is (4,7) -> maps to V indices j*p+k. p=3.
    # V2[j*2+k]. j in 0,1. k in 0,1.
    # We need V[j*3+k].
    # For k=0,1:
    V[0*3+0, :7] = V2[0*2+0, :7]
    V[1*3+0, :7] = V2[1*2+0, :7]
    V[0*3+1, :7] = V2[0*2+1, :7]
    V[1*3+1, :7] = V2[1*2+1, :7]
    
    # W2 is (4,7) -> maps to W indices k*2+i.
    # We need W[k*2+i] for k=0,1.
    W[0*2+0, :7] = W2[0*2+0, :7]
    W[1*2+0, :7] = W2[0*2+1, :7] # Wait. W2 index is k*2+i.
    # In 2x2x2, k=0,1. i=0,1.
    # W2[0] is k=0,i=0. W2[1] is k=0,i=1.
    # W2[2] is k=1,i=0. W2[3] is k=1,i=1.
    # In 2x2x3, W index is k*2+i.
    # k=0 -> idx 0,1. k=1 -> idx 2,3.
    # So W[0:4, :7] = W2.
    W[0:4, :7] = W2
    
    # Part 2: k=2 (index 2 in V and W)
    # U1 is (4,4).
    U[:, 7:11] = U1
    # V1 is (2,4). m=2, p=1.
    # V1[j*1+k]. j=0,1. k=0.
    # In 2x2x3, V index j*3+2.
    V[0*3+2, 7:11] = V1[0*1+0, :]
    V[1*3+2, 7:11] = V1[1*1+0, :]
    
    # W1 is (2,4). p=1, n=2.
    # W1[k*2+i]. k=0. i=0,1.
    # In 2x2x3, W index k*2+i. k=2.
    # idx 2*2+0 = 4. idx 2*2+1 = 5.
    W[4, 7:11] = W1[0, :]
    W[5, 7:11] = W1[1, :]
    
    return U, V, W

def _construct_2x3x3_rank17():
    """
    Combine 2x2x3 (rank 11) and 2x1x3 (rank 6) for 2x3x3.
    U: (6, 17)
    V: (9, 17)
    W: (6, 17)
    """
    R = 17
    U = np.zeros((6, R), dtype=float)
    V = np.zeros((9, R), dtype=float)
    W = np.zeros((6, R), dtype=float)
    
    U2, V2, W2 = _construct_2x2x3_rank11()
    U1, V1, W1 = _construct_2x1x3_rank6()
    
    # Part 1: j in {0,1}
    U[0:4, :11] = U2
    # V2 is (6,11). m=2, p=3.
    # V2[j*3+k]. j=0,1.
    # In 2x3x3, V index j*3+k. j=0,1.
    V[0*3+0:2*3+3, :11] = V2 # Rows 0..5
    V[0:6, :11] = V2
    
    # W2 is (6,11). p=3, n=2.
    # W2[k*2+i].
    # In 2x3x3, W index k*2+i.
    W[0:6, :11] = W2
    
    # Part 2: j=2
    U[4:6, 11:17] = U1
    # V1 is (3,6). m=1, p=3.
    # V1[j*3+k]. j=0.
    # In 2x3x3, V index 2*3+k = 6+k.
    V[6:9, 11:17] = V1
    
    # W1 is (6,6). p=3, n=2.
    # W1[k*2+i].
    # In 2x3x3, W index k*2+i.
    W[0:6, 11:17] += W1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2 and 3.
    # Part 1: 2x2x6 (split p=6 into 3+3)
    # Part 2: 2x3x6 (split p=6 into 3+3)
    
    # Get base decompositions
    U_223, V_223, W_223 = _construct_2x2x3_rank11()
    U_233, V_233, W_233 = _construct_2x3x3_rank17()
    
    R_226 = 22
    R_236 = 34
    TotalR = R_226 + R_236
    
    U = np.zeros((n*m, TotalR), dtype=float)
    V = np.zeros((m*p, TotalR), dtype=float)
    W = np.zeros((p*n, TotalR), dtype=float)
    
    # Fill 2x2x6 part (cols 0-21)
    # 2x2x6 is two 2x2x3.
    # k=0,1,2 and k=3,4,5.
    
    # First 2x2x3 (k=0..2)
    U[0:4, 0:11] = U_223
    # V_223 is (6,11). j*3+k.
    V[0:6, 0:11] = V_223
    W[0:6, 0:11] = W_223
    
    # Second 2x2x3 (k=3..5)
    U[0:4, 11:22] = U_223
    # V index j*6+k. k=3,4,5.
    # V_223 col j*3+k' where k'=0,1,2.
    # Target index j*6 + (k'+3).
    for j in range(2):
        for kp in range(3):
            V[j*6 + kp + 3, 11:22] = V_223[j*3 + kp, :]
    # W index k*2+i. k=3,4,5.
    # W_223 index k'*2+i where k'=0,1,2.
    # Target index (k'+3)*2 + i.
    for kp in range(3):
        for i in range(2):
            W[(kp+3)*2 + i, 11:22] = W_223[kp*2 + i, :]
            
    # Fill 2x3x6 part (cols 22-55)
    # 2x3x6 is two 2x3x3.
    
    # First 2x3x3 (k=0..2)
    U[4:6, 22:39] = U_233 # U_233 is (6,17). Rows 4,5 correspond to j=2?
    # Wait. In 2x3x6, A is 2x5.
    # Part 1 (2x2x6) used A[0:4]. (i=0,1; j=0,1)
    # Part 2 (2x3x6) uses A[0:6]. (i=0,1; j=0,1,2)
    # So U[0:6, 22:39] = U_233.
    U[0:6, 22:39] = U_233
    V[0:9, 22:39] = V_233
    W[0:6, 22:39] = W_233
    
    # Second 2x3x3 (k=3..5)
    U[0:6, 39:56] = U_233
    # V index j*6+k. k=3,4,5.
    # V_233 index j*3+k'.
    for j in range(3):
        for kp in range(3):
            V[j*6 + kp + 3, 39:56] = V_233[j*3 + kp, :]
    # W index k*2+i. k=3,4,5.
    for kp in range(3):
        for i in range(2):
            W[(kp+3)*2 + i, 39:56] = W_233[kp*2 + i, :]
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"].shape[1])
