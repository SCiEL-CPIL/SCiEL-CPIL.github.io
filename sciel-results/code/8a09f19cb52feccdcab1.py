import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Build the full 6x6 tensor decomposition using Strassen 2x2 blocks (rank 7)
    # Each block is 3x3, so we treat the 6x6 matrix as a 2x2 block matrix
    # where each entry is a 3x3 matrix.
    # U has shape (n*m, rank) = (36, rank), V has shape (m*p, rank) = (36, rank),
    # W has shape (p*n, rank) = (36, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For blocks: i = i2*3+i3, j = j2*3+j3, k = k2*3+k3
    # Block indices: i2,j2 in {0,1}, i3,j3 in {0,1,2}
    
    # Strassen 2x2 decomposition (rank 7)
    # U2[i2*2+j2, r], V2[j2*2+k2, r], W2[k2*2+i2, r]
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # r=0: (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # r=1: (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1; W2[3,1]=-1
    
    # r=2: A00(B10-B11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[2,2]=1; W2[3,2]=1
    
    # r=3: A11(B01-B00)
    U2[3,3]=1
    V2[0,3]=-1; V2[2,3]=1
    W2[0,3]=1; W2[1,3]=1
    
    # r=4: (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[2,4]=1
    
    # r=5: (A01-A10)(B00+B10)
    U2[0,5]=-1; U2[2,5]=1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # r=6: (A10-A00)(B11+B01)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    
    # Verify Strassen 2x2
    def verify2x2():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for r in range(7):
                        s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    strassen_valid = verify2x2()
    
    # For the 3x3 sub-blocks, we use schoolbook (rank 27) since we don't have
    # a verified rank-19 decomposition. The total rank will be 7 * 27 = 189.
    # U3[i3*3+j3, r], V3[j3*3+k3, r], W3[k3*3+i3, r]
    U3 = np.zeros((9, 27), dtype=float)
    V3 = np.zeros((9, 27), dtype=float)
    W3 = np.zeros((9, 27), dtype=float)
    
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                r = i3 * 9 + j3 * 3 + k3
                U3[i3*3+j3, r] = 1
                V3[j3*3+k3, r] = 1
                W3[k3*3+i3, r] = 1
    
    # Verify 3x3 schoolbook
    def verify3x3():
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(27):
                        s += U3[i*3+j, r] * V3[j*3+k, r] * W3[k*3+i, r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    liu_valid = verify3x3()
    
    if not strassen_valid or not liu_valid:
        # Fallback to full schoolbook 6x6 (rank 216)
        rank = 216
        U6 = np.zeros((36, rank), dtype=float)
        V6 = np.zeros((36, rank), dtype=float)
        W6 = np.zeros((36, rank), dtype=float)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    r = i * 36 + j * 6 + k
                    U6[i*6+j, r] = 1
                    V6[j*6+k, r] = 1
                    W6[k*6+i, r] = 1
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    # Construct 6x6 factors via Kronecker product
    strassen_rank = 7
    liu_rank = 27
    rank = strassen_rank * liu_rank  # 189
    
    U6 = np.zeros((36, rank), dtype=float)
    V6 = np.zeros((36, rank), dtype=float)
    W6 = np.zeros((36, rank), dtype=float)
    
    for r2 in range(strassen_rank):
        for r3 in range(liu_rank):
            r = r2 * liu_rank + r3
            
            nz_u2 = np.nonzero(U2[:, r2])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            
            nz_u3 = np.nonzero(U3[:, r3])[0]
            nz_v3 = np.nonzero(V3[:, r3])[0]
            nz_w3 = np.nonzero(W3[:, r3])[0]
            
            for u2 in nz_u2:
                i2 = u2 // 2
                j2 = u2 % 2
                val_u2 = U2[u2, r2]
                
                for u3 in nz_u3:
                    i3 = u3 // 3
                    j3 = u3 % 3
                    val_u3 = U3[u3, r3]
                    
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    g_u = i6 * 6 + j6
                    U6[g_u, r] += val_u2 * val_u3
            
            for v2 in nz_v2:
                j2 = v2 // 2
                k2 = v2 % 2
                val_v2 = V2[v2, r2]
                
                for v3 in nz_v3:
                    j3 = v3 // 3
                    k3 = v3 % 3
                    val_v3 = V3[v3, r3]
                    
                    j6 = j2 * 3 + j3
                    k6 = k2 * 3 + k3
                    g_v = j6 * 6 + k6
                    V6[g_v, r] += val_v2 * val_v3
            
            for w2 in nz_w2:
                k2 = w2 // 2
                i2 = w2 % 2
                val_w2 = W2[w2, r2]
                
                for w3 in nz_w3:
                    k3 = w3 // 3
                    i3 = w3 % 3
                    val_w3 = W3[w3, r3]
                    
                    k6 = k2 * 3 + k3
                    i6 = i2 * 3 + i3
                    g_w = k6 * 6 + i6
                    W6[g_w, r] += val_w2 * val_w3
    
    # Verify the full 6x6 decomposition
    def verify6x6():
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    s = 0.0
                    for r in range(rank):
                        s += U6[i*6+j, r] * V6[j*6+k, r] * W6[k*6+i, r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    if not verify6x6():
        # Fallback to schoolbook 6x6
        U6 = np.zeros((36, 216), dtype=float)
        V6 = np.zeros((36, 216), dtype=float)
        W6 = np.zeros((36, 216), dtype=float)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    r = i * 36 + j * 6 + k
                    U6[i*6+j, r] = 1
                    V6[j*6+k, r] = 1
                    W6[k*6+i, r] = 1
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    # Time guard
    if time.time() - start_time > 110:
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
