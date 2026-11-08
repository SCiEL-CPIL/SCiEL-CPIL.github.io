import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Helper: Kronecker product of decompositions
    def kron_decomp(U1, V1, W1, U2, V2, W2):
        return np.kron(U1, U2), np.kron(V1, V2), np.kron(W1, W2)

    # Helper: Strassen for 2x2x2
    def strassen_222():
        # Rank 7
        # U: (2*2, 7) = 4x7
        # V: (2*2, 7) = 4x7
        # W: (2*2, 7) = 4x7
        # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
        # n=2, m=2, p=2
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00 +=, C11 +=
        # C00 is k=0,i=0 -> W[0]. C11 is k=1,i=1 -> W[1*2+1]=W[3].
        
        # M2 = (A00+A11)(B10-B11)
        U[0, 1] = 1; U[3, 1] = 1
        V[2, 1] = 1; V[3, 1] = -1 # B10 is j=1,k=0 -> 2. B11 is j=1,k=1 -> 3.
        W[2, 1] = 1; W[3, 1] = -1 # C10 is k=0,i=1 -> W[1]. C11 is k=1,i=1 -> W[3].
        # Wait. C10: k=0, i=1. W index k*n+i = 0*2+1 = 1.
        # C11: k=1, i=1. W index 1*2+1 = 3.
        # My W assignments above:
        # M1: C00 (W0), C11 (W3). Correct.
        # M2: C10 (W1), C11 (W3). 
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's build carefully.
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1 # C00, C11
        
        # M2: (A00+A11)(B10-B11)
        U[0,1]=1; U[3,1]=1
        V[2,1]=1; V[3,1]=-1
        W[1,1]=1; W[3,1]=-1 # C10, C11
        
        # M3: (A00-A10)B00
        U[0,2]=1; U[2,2]=-1
        V[0,2]=1
        W[0,2]=1; W[1,2]=1 # C00, C01
        
        # M4: (A10+A11)B11
        U[2,3]=1; U[3,3]=1
        V[3,3]=1
        W[1,3]=1; W[3,3]=1 # C10, C11
        
        # M5: (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1 # C00, C01
        
        # M6: (A01-A00)(B00+B10)
        U[1,5]=1; U[0,5]=-1
        V[0,5]=1; V[2,5]=1
        W[3,5]=1 # C11
        
        # M7: (A10-A01)(B01+B11)
        U[2,6]=1; U[1,6]=-1
        V[1,6]=1; V[3,6]=1
        W[0,6]=1; W[3,6]=1 # C00, C11
        
        return U, V, W

    # Helper: Trivial 1x1x2
    def trivial_112():
        # A 1x1, B 1x2, C 1x2
        # C0 = A0 B0
        # C1 = A0 B1
        U = np.array([[1, 1]]) # 1x2
        V = np.array([[1, 0], [0, 1]]) # 2x2
        W = np.array([[1, 0], [0, 1]]) # 2x2 (p*n = 2*1)
        return U, V, W

    # Helper: Trivial 1x2x2
    def trivial_122():
        # A 1x2, B 2x2, C 1x2
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A0 B00 -> C0
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # Term 2: A1 B10 -> C0
        U[1,1]=1; V[2,1]=1; W[0,1]=1
        # Term 3: A0 B01 -> C1
        U[0,2]=1; V[1,2]=1; W[1,2]=1
        # Term 4: A1 B11 -> C1
        U[1,3]=1; V[3,3]=1; W[1,3]=1
        
        return U, V, W

    # 1. Get T_222 and T_112
    U_222, V_222, W_222 = strassen_222()
    U_112, V_112, W_112 = trivial_112()
    
    # 2. T_224 = T_222 x T_112
    U_224, V_224, W_224 = kron_decomp(U_222, V_222, W_222, U_112, V_112, W_112)
    
    # 3. T_124 = two T_122s
    U_122_a, V_122_a, W_122_a = trivial_122()
    U_122_b, V_122_b, W_122_b = trivial_122()
    
    # Stack T_122_a and T_122_b
    # T_124: A 1x2, B 2x4, C 1x4
    # U: 2x8
    U_124 = np.hstack([U_122_a, U_122_b])
    # V: 8x8. B is 2x4. 
    # T_122_a uses B cols 0,1 (first 2x2 block).
    # T_122_b uses B cols 2,3 (second 2x2 block).
    # We need to pad V_122_a to 8x4 and V_122_b to 8x4.
    V_124 = np.zeros((8, 8))
    # V_122 is 4x4 (rows j*p+k, p=2).
    # For first block (cols 0,1 of B_124), indices are 0..3.
    V_124[0:4, 0:4] = V_122_a
    # For second block (cols 2,3 of B_124), indices are 4..7?
    # B index j*p+k. j in 0..1. k in 0..3.
    # If we split B into two 2x2 blocks, 
    # Block 1: k=0,1. Indices: j*4+k? No.
    # Original B for T_122 was 2x2. Index j*2+k.
    # For T_124, B is 2x4. Index j*4+k.
    # If we map T_122_a to k=0,1:
    # V_122_a row j*2+k maps to V_124 row j*4+k.
    for j in range(2):
        for k in range(2):
            V_124[j*4+k, 0:4] = V_122_a[j*2+k, 0:4]
            
    # Map T_122_b to k=2,3:
    # V_122_b row j*2+k maps to V_124 row j*4+(k+2).
    for j in range(2):
        for k in range(2):
            V_124[j*4+(k+2), 4:8] = V_122_b[j*2+k, 0:4]
            
    # W: 4x8. C is 1x4. Index k*n+i. n=1. Index k.
    W_124 = np.hstack([W_122_a, W_122_b])
    # W_122 is 2x4 (p=2, n=1). Rows 0,1.
    # For first block, C cols 0,1. W rows 0,1.
    # For second block, C cols 2,3. W rows 2,3.
    # W_122_a rows map to W_124 rows 0,1.
    # W_122_b rows map to W_124 rows 2,3.
    W_124[0:2, 0:4] = W_122_a
    W_124[2:4, 4:8] = W_122_b
    
    # 4. T_324 = T_224 + T_124
    # A 3x2. Top 2x2 (rows 0,1), Bottom 1x2 (row 2).
    # B 2x4. Shared.
    # C 3x4. Top 2x4 (rows 0,1), Bottom 1x4 (row 2).
    
    # U_224 is 4x14 (n=2,m=2).
    # U_124 is 2x8 (n=1,m=2).
    # U_324 is 6x22.
    U_324 = np.zeros((6, 22))
    U_324[0:4, 0:14] = U_224
    U_324[4:6, 14:22] = U_124
    
    # V_224 is 8x14 (m=2,p=4).
    # V_124 is 8x8.
    # V_324 is 8x22.
    V_324 = np.hstack([V_224, V_124])
    
    # W_224 is 8x14 (p=4,n=2). Rows k*n+i.
    # W_124 is 4x8.
    # W_324 is 12x22.
    W_324 = np.zeros((12, 22))
    # Top part (rows 0,1 of C): W_224 rows 0..7.
    W_324[0:8, 0:14] = W_224
    # Bottom part (row 2 of C): W_124 rows 0..3 map to W_324 rows 8..11.
    W_324[8:12, 14:22] = W_124
    
    # 5. T_122 for the outer product
    U_B, V_B, W_B = trivial_122()
    
    # 6. T_348 = T_324 x T_122
    U_348, V_348, W_348 = kron_decomp(U_324, V_324, W_324, U_B, V_B, W_B)
    
    return {"U": U_348.tolist(), "V": V_348.tolist(), "W": W_348.tolist()}
# EVOLVE-BLOCK-END
