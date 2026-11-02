# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy: Decompose M_{2,3,6} as M_{2,3,2} \otimes M_{1,1,3}.
    M_{2,3,2} is decomposed using Strassen (rank 7) for the 2x2x2 core 
    and schoolbook (rank 4) for the remaining j=2 part. Total rank 11.
    M_{1,1,3} is schoolbook rank 3.
    Total rank = 11 * 3 = 33.
    """
    n, m, p = 2, 3, 6
    
    # 1. Construct M_{2,3,2} decomposition (Rank 11)
    # Part A: M_{2,2,2} Strassen (Rank 7)
    # Part B: j=2 part (Rank 4)
    
    U_A = np.zeros((2*3, 11)) # n*m = 6
    V_A = np.zeros((3*2, 11)) # m*p = 6
    W_A = np.zeros((2*2, 11)) # p*n = 4
    
    # Strassen for M_{2,2,2}
    # Indices i,j in {0,1}
    # P1 = (A00+A11)(B00+B11)
    U_A[0, 0] = 1; U_A[3, 0] = 1
    V_A[0, 0] = 1; V_A[3, 0] = 1
    W_A[0, 0] = 1; W_A[1, 0] = 1; W_A[2, 0] = 1; W_A[3, 0] = -1 # C00=P1..., C01, C10, C11
    # Wait, W_A indices are k*n+i.
    # C00 (k=0,i=0): idx 0. C01 (k=1,i=0): idx 1. C10 (k=0,i=1): idx 2. C11 (k=1,i=1): idx 3.
    # P1 contributes to C00(+), C01(0), C10(0), C11(-).
    W_A[0, 0] = 1
    W_A[3, 0] = -1
    
    # P2 = (A10+A11)B00
    U_A[3, 1] = 1; U_A[2, 1] = 1 # A10 is idx 2? No. A is 2x3. i=1,j=0 -> idx 1*3+0=3. i=1,j=1 -> idx 1*3+1=4.
    # Correction: U_A index is i*m+j.
    # i=0,j=0 -> 0. i=0,j=1 -> 1.
    # i=1,j=0 -> 3. i=1,j=1 -> 4.
    U_A[3, 1] = 1; U_A[4, 1] = 1
    V_A[0, 1] = 1 # B00 (j=0,k=0) -> idx 0
    # C10(+), C11(-)
    W_A[2, 1] = 1
    W_A[3, 1] = -1
    
    # P3 = A00(B01-B11)
    U_A[0, 2] = 1
    V_A[1, 2] = 1; V_A[3, 2] = -1 # B01 (j=0,k=1) idx 1. B11 (j=1,k=1) idx 3.
    # C01(+), C11(+)
    W_A[1, 2] = 1
    W_A[3, 2] = 1
    
    # P4 = A11(B10-B00)
    U_A[4, 3] = 1
    V_A[2, 3] = 1; V_A[0, 3] = -1 # B10 (j=1,k=0) idx 2. B00 idx 0.
    # C00(+), C10(+)
    W_A[0, 3] = 1
    W_A[2, 3] = 1
    
    # P5 = (A00+A01)B11
    U_A[0, 4] = 1; U_A[1, 4] = 1
    V_A[3, 4] = 1
    # C00(-), C01(+)
    W_A[0, 4] = -1
    W_A[1, 4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    U_A[3, 5] = 1; U_A[0, 5] = -1
    V_A[0, 5] = 1; V_A[1, 5] = 1
    # C11(+)
    W_A[3, 5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U_A[1, 6] = 1; U_A[4, 6] = -1
    V_A[2, 6] = 1; V_A[3, 6] = 1
    # C00(+)
    W_A[0, 6] = 1

    # Part B: j=2 part (Rank 4)
    # C_{i,k} += A_{i,2} B_{2,k}
    # i in {0,1}, k in {0,1}
    # A_{i,2} idx: i=0->2, i=1->5.
    # B_{2,k} idx: j=2. k=0->4, k=1->5.
    
    # t=7: i=0, k=0
    U_A[2, 7] = 1
    V_A[4, 7] = 1
    W_A[0, 7] = 1
    
    # t=8: i=0, k=1
    U_A[2, 8] = 1
    V_A[5, 8] = 1
    W_A[1, 8] = 1
    
    # t=9: i=1, k=0
    U_A[5, 9] = 1
    V_A[4, 9] = 1
    W_A[2, 9] = 1
    
    # t=10: i=1, k=1
    U_A[5, 10] = 1
    V_A[5, 10] = 1
    W_A[3, 10] = 1

    # 2. Construct M_{1,1,3} decomposition (Rank 3)
    U_B = np.zeros((1*1, 3)) # n*m = 1
    V_B = np.zeros((1*3, 3)) # m*p = 3
    W_B = np.zeros((3*1, 3)) # p*n = 3
    
    # t=0: k=0
    U_B[0, 0] = 1
    V_B[0, 0] = 1
    W_B[0, 0] = 1
    
    # t=1: k=1
    U_B[0, 1] = 1
    V_B[1, 1] = 1
    W_B[1, 1] = 1
    
    # t=2: k=2
    U_B[0, 2] = 1
    V_B[2, 2] = 1
    W_B[2, 2] = 1

    # 3. Kronecker Product
    rank_A = 11
    rank_B = 3
    rank = rank_A * rank_B
    
    U = np.zeros((6, rank))
    V = np.zeros((18, rank))
    W = np.zeros((12, rank))
    
    for ta in range(rank_A):
        for tb in range(rank_B):
            col = ta * rank_B + tb
            
            # U
            for i in range(2):
                for ip in range(1):
                    # U_A index: i*3 + j? No, U_A is for M_{2,3,2}.
                    # U_A shape (6, 11). Index i*m_A + j_A. m_A=3.
                    # U_B shape (1, 3). Index i_B*m_B + j_B. m_B=1.
                    # Kronecker index: (i, i_B) mapped to i*n_B + i_B?
                    # n=2, n_B=1.
                    # i is 0,1. i_B is 0.
                    # New i is i.
                    # New j is j_A * m_B + j_B = j_A * 1 + 0 = j_A.
                    # So U[ i*3 + j_A, col ] = U_A[i*3+j_A, ta] * U_B[0, tb]
                    U[i*3 + 0, col] = U_A[i*3 + 0, ta] * U_B[0, tb] # Wait, j_A is 0..2.
                    # My U_A loop above only set j=0,1 for Strassen.
                    # I need to loop over all j_A.
            
            # Let's do the Kronecker more systematically.
            # U_A[i_A, j_A, ta] where i_A in 0..1, j_A in 0..2.
            # U_B[i_B, j_B, tb] where i_B=0, j_B=0.
            # U[i_A*1 + i_B, j_A*1 + j_B, ta, tb]
            # i = i_A. j = j_A.
            # U[i*3 + j, col] = U_A[i*3+j, ta] * U_B[0, tb]
            
            for i_A in range(2):
                for j_A in range(3):
                    U[i_A*3 + j_A, col] = U_A[i_A*3 + j_A, ta] * U_B[0, tb]
                    
            # V
            # V_A[j_A, k_A, ta] where j_A in 0..2, k_A in 0..1.
            # V_B[j_B, k_B, tb] where j_B=0, k_B in 0..2.
            # V[j_A*1 + j_B, k_A*3 + k_B, col]
            # j = j_A. k = k_A*3 + k_B.
            for j_A in range(3):
                for k_A in range(2):
                    for k_B in range(3):
                        V[j_A*1 + 0, k_A*3 + k_B, col] = V_A[j_A*2 + k_A, ta] * V_B[0*3 + k_B, tb]
                        
            # W
            # W_A[k_A, i_A, ta] where k_A in 0..1, i_A in 0..1.
            # W_B[k_B, i_B, tb] where k_B in 0..2, i_B=0.
            # W[k_A*3 + k_B, i_A*1 + i_B, col]
            # k = k_A*3 + k_B. i = i_A.
            for k_A in range(2):
                for i_A in range(2):
                    for k_B in range(3):
                        W[k_A*3 + k_B, i_A*1 + 0, col] = W_A[k_A*2 + i_A, ta] * W_B[k_B*1 + 0, tb]

    # Verify
    T = np.zeros((6, 18, 12))
    for i in range(2):
        for j in range(3):
            for k in range(6):
                T[i*3+j, j*6+k, k*2+i] = 1.0
                
    recon = np.zeros((6, 18, 12))
    for col in range(rank):
        recon += np.einsum('a,b,c->abc', U[:, col], V[:, col], W[:, col])
        
    if np.max(np.abs(recon - T)) < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    rank_fb = 36
    U_fb = np.zeros((6, rank_fb))
    V_fb = np.zeros((18, rank_fb))
    W_fb = np.zeros((12, rank_fb))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(6):
                U_fb[i*3+j, col] = 1
                V_fb[j*6+k, col] = 1
                W_fb[k*2+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
