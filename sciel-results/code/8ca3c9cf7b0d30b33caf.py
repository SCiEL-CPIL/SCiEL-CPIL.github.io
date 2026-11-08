import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # ------------------------------------------------------------------
    # Base decomposition: 2x2x2 (Strassen, rank 7)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # ------------------------------------------------------------------
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # r=0: M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # C00 += M1, C11 -= M1
        # W row: k*n+i. C00 is (i=0,k=0)->row 0. C11 is (i=1,k=1)->row 3.
        # So W[0,0]=1, W[3,0]=-1?
        # Let's check: C[i,k] = sum A[i,j] B[j,k].
        # C00 = A00*B00 + A01*B10.
        # C11 = A10*B01 + A11*B11.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # C00 += M1 => adds A00B00 (good), A00B11 (bad), A11B00 (bad), A11B11 (bad for C00).
        # C11 -= M1 => subtracts A00B00 (bad), A00B11 (bad), A11B00 (bad), A11B11 (good).
        # So we need to add/subtract other Ms to cancel the bad terms.
        # Standard Strassen:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 + M7
        
        # Let's define the 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # r=0: M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        
        # r=1: M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # r=2: M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = -1
        
        # r=3: M4
        U[3, 3] = 1
        V[1, 3] = 1; V[2, 3] = -1
        W[2, 3] = 1; W[0, 3] = -1
        
        # r=4: M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # r=5: M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[2, 5] = 1
        
        # r=6: M7
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W

    # ------------------------------------------------------------------
    # Base decomposition: 2x1x2 (Outer product, rank 2)
    # A is 2x1, B is 1x2, C is 2x2.
    # C[i,k] = A[i,0] * B[0,k].
    # Rank 2:
    # r=0: A00 * B00 -> C00
    # r=1: A00 * B01 -> C01
    # Wait, A has 2 rows.
    # C00 = A00*B00
    # C01 = A00*B01
    # C10 = A10*B00
    # C11 = A10*B01
    # This is rank 4? No.
    # U[i,0,r], V[0,k,r], W[k,0,i,r]? No, W is (p*n, rank) = (4, rank).
    # Indices: a=(i,0) for A (size 2), b=(0,k) for B (size 2), c=(i',k') for C (size 4).
    # T = sum U[i,r] V[k,r] W[i',k',r].
    # We want C[i,k] = A[i] B[k].
    # This is the outer product of vector A and vector B.
    # Rank of outer product tensor is 1? No, it's a matrix.
    # The tensor is T_{i, k, i', k'} = delta_{i,i'} A_i B_k? No.
    # The multiplication is C = A B.
    # The tensor T satisfies C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Here j is only 0. So C_{i,k} = A_{i,0} B_{0,k}.
    # The tensor T has support where i'=i and k'=k.
    # T_{i,0, 0,k, i',k'} = 1 if i=i', k=k'.
    # This is the identity tensor for i and k.
    # Rank is 2*2 = 4?
    # Let's check:
    # U[i,0,r], V[0,k,r], W[i',k',r].
    # If we use r=0..3:
    # r=0: U[0,0,0]=1, V[0,0,0]=1, W[0,0,0]=1 (C00 = A0 B0)
    # r=1: U[0,0,1]=1, V[0,1,1]=1, W[1,0,1]=1 (C01 = A0 B1)
    # r=2: U[1,0,2]=1, V[0,0,2]=1, W[0,1,2]=1 (C10 = A1 B0)
    # r=3: U[1,0,3]=1, V[0,1,3]=1, W[1,1,3]=1 (C11 = A1 B1)
    # So rank 4.
    
    def outer_212():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 0] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[0, 1] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[1, 1] = 1
        return U, V, W

    # ------------------------------------------------------------------
    # Combine 2x2x2 and 2x1x2 to get 2x3x2 (rank 7+4=11)
    # ------------------------------------------------------------------
    def combine_232():
        U1, V1, W1 = strassen_222()
        U2, V2, W2 = outer_212()
        
        rank = 11
        U = np.zeros((6, rank))
        V = np.zeros((6, rank))
        W = np.zeros((4, rank))
        
        # Part 1: j in {0,1} (Strassen)
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*3 + j, r] = U1[i*2 + j, r]
                # V row index: j*2 + k
                for k in range(2):
                    V[j*2 + k, r] = V1[j*2 + k, r]
                # W row index: k*2 + i
                for k in range(2):
                    for i in range(2):
                        W[k*2 + i, r] = W1[k*2 + i, r]
                        
        # Part 2: j=2 (Outer)
        for r in range(4):
            for i in range(2):
                U[i*3 + 2, r+7] = U2[i, r]
            for k in range(2):
                V[2*2 + k, r+7] = V2[k, r]
            for k in range(2):
                for i in range(2):
                    W[k*2 + i, r+7] = W2[k*2 + i, r]
                    
        return U, V, W

    # ------------------------------------------------------------------
    # Combine two 2x3x2 to get 2x3x4 (rank 11+11=22)
    # ------------------------------------------------------------------
    def combine_234():
        U1, V1, W1 = combine_232()
        U2, V2, W2 = combine_232()
        
        rank = 22
        U = np.zeros((6, rank))
        V = np.zeros((12, rank))
        W = np.zeros((8, rank))
        
        # Part 1: k in {0,1}
        for r in range(11):
            for i in range(2):
                for j in range(3):
                    U[i*3 + j, r] = U1[i*3 + j, r]
            for j in range(3):
                for k in range(2):
                    V[j*4 + k, r] = V1[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[k*2 + i, r] = W1[k*2 + i, r]
                    
        # Part 2: k in {2,3}
        for r in range(11):
            for i in range(2):
                for j in range(3):
                    U[i*3 + j, r+11] = U2[i*3 + j, r]
            for j in range(3):
                for k in range(2):
                    V[j*4 + (k+2), r+11] = V2[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[(k+2)*2 + i, r+11] = W2[k*2 + i, r]
                    
        return U, V, W

    # ------------------------------------------------------------------
    # Combine two 2x3x4 to get 2x3x8 (rank 22+22=44)
    # ------------------------------------------------------------------
    def combine_238():
        U1, V1, W1 = combine_234()
        U2, V2, W2 = combine_234()
        
        rank = 44
        U = np.zeros((6, rank))
        V = np.zeros((24, rank))
        W = np.zeros((16, rank))
        
        # Part 1: k in {0..3}
        for r in range(22):
            for i in range(2):
                for j in range(3):
                    U[i*3 + j, r] = U1[i*3 + j, r]
            for j in range(3):
                for k in range(4):
                    V[j*8 + k, r] = V1[j*4 + k, r]
            for k in range(4):
                for i in range(2):
                    W[k*2 + i, r] = W1[k*2 + i, r]
                    
        # Part 2: k in {4..7}
        for r in range(22):
            for i in range(2):
                for j in range(3):
                    U[i*3 + j, r+22] = U2[i*3 + j, r]
            for j in range(3):
                for k in range(4):
                    V[j*8 + (k+4), r+22] = V2[j*4 + k, r]
            for k in range(4):
                for i in range(2):
                    W[(k+4)*2 + i, r+22] = W2[k*2 + i, r]
                    
        return U, V, W

    U, V, W = combine_238()
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
