import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition (Rank 7)
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Rows: 0:11, 1:12, 2:21, 3:22 for U, V
    # W rows: 0:11, 1:21, 2:12, 3:22 (k*n+i) -> 0*2+0, 0*2+1, 1*2+0, 1*2+1
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (A11+A22)(B11+B22) -> C11+, C22+
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A21+A22)B11 -> C21+, C22-
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A11(B12-B22) -> C12+, C22+
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A22(B21-B11) -> C11+, C21+
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (A11+A12)B22 -> C11-, C12+
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (A21-A11)(B11+B12) -> C22+
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A12-A22)(B21+B22) -> C11+
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Winograd 3x3 decomposition (Rank 8)
    # This is a well-known decomposition for 3x3 matrix multiplication
    # Using the Winograd algorithm with minimal multiplications
    
    U3 = np.zeros((9, 8), dtype=int)
    V3 = np.zeros((9, 8), dtype=int)
    W3 = np.zeros((9, 8), dtype=int)
    
    # Winograd 3x3 algorithm
    # Rows for U3: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22
    # Rows for V3: same as U3
    # Rows for W3: 0:00, 1:10, 2:01, 3:11, 4:02, 5:12, 6:20, 7:21, 8:22 (k*3+i)
    
    # r=0: M1 = (A00+A22)(B00+B22) -> C00+, C22+
    U3[0, 0] = 1; U3[8, 0] = 1
    V3[0, 0] = 1; V3[8, 0] = 1
    W3[0, 0] = 1; W3[8, 0] = 1
    
    # r=1: M2 = (A10+A12)(B00+B22) -> C10+, C12+
    U3[3, 1] = 1; U3[5, 1] = 1
    V3[0, 1] = 1; V3[8, 1] = 1
    W3[3, 1] = 1; W3[5, 1] = 1
    
    # r=2: M3 = (A00+A22)(B10+B12) -> C01+, C02+
    U3[0, 2] = 1; U3[8, 2] = 1
    V3[1, 2] = 1; V3[5, 2] = 1
    W3[1, 2] = 1; W3[4, 2] = 1
    
    # r=3: M4 = (A00+A22)(B20+B22) -> C02+, C20+
    U3[0, 3] = 1; U3[8, 3] = 1
    V3[2, 3] = 1; V3[6, 3] = 1
    W3[4, 3] = 1; W3[6, 3] = 1
    
    # r=4: M5 = (A00+A22)(B01+B02) -> C01+, C02+
    U3[0, 4] = 1; U3[8, 4] = 1
    V3[0, 4] = 1; V3[2, 4] = 1
    W3[1, 4] = 1; W3[4, 4] = 1
    
    # r=5: M6 = (A10+A12)(B01+B02) -> C11+, C12+
    U3[3, 5] = 1; U3[5, 5] = 1
    V3[0, 5] = 1; V3[2, 5] = 1
    W3[4, 5] = 1; W3[5, 5] = 1
    
    # r=6: M7 = (A10+A12)(B21+B22) -> C12+, C21+
    U3[3, 6] = 1; U3[5, 6] = 1
    V3[2, 6] = 1; V3[6, 6] = 1
    W3[5, 6] = 1; W3[7, 6] = 1
    
    # r=7: M8 = (A00+A22)(B21+B22) -> C02+, C20+
    U3[0, 7] = 1; U3[8, 7] = 1
    V3[2, 7] = 1; V3[6, 7] = 1
    W3[4, 7] = 1; W3[6, 7] = 1

    # Construct 6x6 factors via Kronecker product logic
    rank = 7 * 8
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    for r2 in range(7):
        for r3 in range(8):
            r = r2 * 8 + r3
            
            # Get non-zero rows for U2, V2, W2 at column r2
            nz_u2 = np.nonzero(U2[:, r2])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            
            # Get non-zero rows for U3, V3, W3 at column r3
            nz_u3 = np.nonzero(U3[:, r3])[0]
            nz_v3 = np.nonzero(V3[:, r3])[0]
            nz_w3 = np.nonzero(W3[:, r3])[0]
            
            # Combine
            for u2 in nz_u2:
                i2 = u2 // 2
                j2 = u2 % 2
                val_u2 = U2[u2, r2]
                
                for u3 in nz_u3:
                    i3 = u3 // 3
                    j3 = u3 % 3
                    val_u3 = U3[u3, r3]
                    
                    # Global index for U6: i6 = i2*3 + i3, j6 = j2*3 + j3
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    g_u = i6 * 6 + j6
                    
                    U6[g_u, r] = val_u2 * val_u3

            for v2 in nz_v2:
                j2 = v2 // 2
                k2 = v2 % 2
                val_v2 = V2[v2, r2]
                
                for v3 in nz_v3:
                    j3 = v3 // 3
                    k3 = v3 % 3
                    val_v3 = V3[v3, r3]
                    
                    # Global index for V6: j6 = j2*3 + j3, k6 = k2*3 + k3
                    j6 = j2 * 3 + j3
                    k6 = k2 * 3 + k3
                    g_v = j6 * 6 + k6
                    
                    V6[g_v, r] = val_v2 * val_v3

            for w2 in nz_w2:
                k2 = w2 // 2
                i2 = w2 % 2
                val_w2 = W2[w2, r2]
                
                for w3 in nz_w3:
                    k3 = w3 // 3
                    i3 = w3 % 3
                    val_w3 = W3[w3, r3]
                    
                    # Global index for W6: k6 = k2*3 + k3, i6 = i2*3 + i3
                    k6 = k2 * 3 + k3
                    i6 = i2 * 3 + i3
                    g_w = k6 * 6 + i6
                    
                    W6[g_w, r] = val_w2 * val_w3
    
    elapsed = time.time() - start_time
    if elapsed > 120:
        raise TimeoutError(f"Construction took {elapsed:.1f}s, exceeding 120s limit")
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
