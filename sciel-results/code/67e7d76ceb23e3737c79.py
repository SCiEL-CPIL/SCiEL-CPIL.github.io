import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Tensor product of Strassen 2x2 (Rank 7) and Schoolbook 3x3 (Rank 27)
    # Total Rank = 7 * 27 = 189
    # This decomposes the 6x6 multiplication into 7 block multiplications of 3x3 matrices.
    
    # 1. Define Strassen 2x2 decomposition (Rank 7)
    # U2, V2, W2 have shape (4, 7) where row index is i*2 + j for U/V, and k*2 + i for W?
    # In Strassen, for 2x2:
    # U rows: (0,0), (1,0), (0,1), (1,1) -> indices 0, 1, 2, 3
    # V rows: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
    # W rows: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
    # Wait, the standard flattening for W[k*n+i] with n=2:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # So W index is k*2 + i.
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen M1: (A00 + A11) * (B00 + B11)
    # U: A00 (0,0)=1, A11 (1,1)=1 -> idx 0:1, idx 3:1
    # V: B00 (0,0)=1, B11 (1,1)=1 -> idx 0:1, idx 3:1
    # W: C00 += 1, C01 += 1, C10 += 1, C11 += 1? 
    # M1 contributes to:
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M5 + M6 - M7
    
    # Let's assign coefficients based on standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Col 0 (M1):
    # U: A00(0), A11(3) -> 1, 1
    # V: B00(0), B11(3) -> 1, 1
    # W: C00(0)+1, C01(1)-1? No.
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M5 + M6 - M7
    
    # So W coeffs for M1:
    # C00: +1 (idx 0)
    # C01: 0
    # C10: 0
    # C11: +1 (idx 3)
    # Wait, M1 is subtracted in C01? No.
    # Let's re-verify Strassen formulas:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 - P4 - P6 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P5 + P6 - P7
    
    # W matrix for M1 (P1):
    # C00(0): 1
    # C01(1): 0
    # C10(2): 0
    # C11(3): 1
    W2[0, 0] = 1; W2[3, 0] = 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    
    # W for M2 (P2):
    # C10(2): 1
    W2[2, 1] = 1
    # U: (A10+A11) -> A10(idx 2), A11(idx 3)
    U2[2, 1] = 1; U2[3, 1] = 1
    # V: B00(idx 0)
    V2[0, 1] = 1
    
    # W for M3 (P3):
    # C01(1): 1
    W2[1, 2] = 1
    # U: A00(idx 0)
    U2[0, 2] = 1
    # V: (B10-B11) -> B10(idx 2), B11(idx 3)
    V2[2, 2] = 1; V2[3, 2] = -1
    
    # W for M4 (P4):
    # C00(0): -1
    # C10(2): 1
    W2[0, 3] = -1; W2[2, 3] = 1
    # U: A11(idx 3)
    U2[3, 3] = 1
    # V: (B01-B00) -> B01(idx 1), B00(idx 0)
    V2[1, 3] = 1; V2[0, 3] = -1
    
    # W for M5 (P5):
    # C01(1): 1
    # C11(3): -1
    W2[1, 4] = 1; W2[3, 4] = -1
    # U: (A00+A01) -> A00(idx 0), A01(idx 1)
    U2[0, 4] = 1; U2[1, 4] = 1
    # V: B11(idx 3)
    V2[3, 4] = 1
    
    # W for M6 (P6):
    # C00(0): -1
    # C11(3): 1
    W2[0, 5] = -1; W2[3, 5] = 1
    # U: (A10-A00) -> A10(idx 2), A00(idx 0)
    U2[2, 5] = 1; U2[0, 5] = -1
    # V: (B00+B01) -> B00(idx 0), B01(idx 1)
    V2[0, 5] = 1; V2[1, 5] = 1
    
    # W for M7 (P7):
    # C00(0): 1
    # C11(3): -1
    W2[0, 6] = 1; W2[3, 6] = -1
    # U: (A01-A11) -> A01(idx 1), A11(idx 3)
    U2[1, 6] = 1; U2[3, 6] = -1
    # V: (B10+B11) -> B10(idx 2), B11(idx 3)
    V2[2, 6] = 1; V2[3, 6] = 1

    # 2. Define Schoolbook 3x3 decomposition (Rank 27)
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    # Schoolbook: C[i,k] = sum_j A[i,j] B[j,k]
    # Each product A[i,j] * B[j,k] is a rank 1 term.
    # There are 3*3*3 = 27 terms.
    # Index r corresponds to i, j, k where r = i*9 + j*3 + k.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                # U: A[i,j] -> row idx i*3 + j
                U3[i * 3 + j, r] = 1
                # V: B[j,k] -> row idx j*3 + k
                V3[j * 3 + k, r] = 1
                # W: C[i,k] -> row idx i*3 + k (Note: Task says W[k*n+i]. Here n=3. So k*3+i? 
                # Let's check the task convention again.
                # "output factor follows the official trace convention W[k*n+i,r]"
                # In the 3x3 sub-problem, the output is a 3x3 matrix.
                # The global W index is k_global * 6 + i_global.
                # In the decomposition loop, we map local indices to global.
                # Local W index for 3x3: If we follow the same convention W[k*n+i] with n=3, 
                # it should be k*3 + i.
                # My previous code used `k3 = w3 // 3; i3 = w3 % 3`, which implies index = k*3+i.
                # So I should set W3[k*3 + i, r] = 1.
                W3[k * 3 + i, r] = 1

    # 3. Construct the 6x6x6 matrices
    rank = 7 * 27
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    # Precompute non-zeros for efficiency
    nz_u2 = {r: np.nonzero(U2[:, r])[0] for r in range(7)}
    nz_v2 = {r: np.nonzero(V2[:, r])[0] for r in range(7)}
    nz_w2 = {r: np.nonzero(W2[:, r])[0] for r in range(7)}
    nz_u3 = {r: np.nonzero(U3[:, r])[0] for r in range(27)}
    nz_v3 = {r: np.nonzero(V3[:, r])[0] for r in range(27)}
    nz_w3 = {r: np.nonzero(W3[:, r])[0] for r in range(27)}
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            
            # Process U
            for u2_idx in nz_u2[r2]:
                i2 = u2_idx // 2
                j2 = u2_idx % 2
                val_u2 = U2[u2_idx, r2]
                
                for u3_idx in nz_u3[r3]:
                    i3 = u3_idx // 3
                    j3 = u3_idx % 3
                    val_u3 = U3[u3_idx, r3]
                    
                    # Map to global 6x6 indices
                    # Global A is 6x6. Split into 2x2 blocks of 3x3.
                    # Global i = i2 * 3 + i3
                    # Global j = j2 * 3 + j3
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    g_u = i6 * 6 + j6
                    U6[g_u, r] += val_u2 * val_u3
            
            # Process V
            for v2_idx in nz_v2[r2]:
                j2 = v2_idx // 2
                k2 = v2_idx % 2
                val_v2 = V2[v2_idx, r2]
                
                for v3_idx in nz_v3[r3]:
                    j3 = v3_idx // 3
                    k3 = v3_idx % 3
                    val_v3 = V3[v3_idx, r3]
                    
                    j6 = j2 * 3 + j3
                    k6 = k2 * 3 + k3
                    g_v = j6 * 6 + k6
                    V6[g_v, r] += val_v2 * val_v3
            
            # Process W
            for w2_idx in nz_w2[r2]:
                k2 = w2_idx // 2
                i2 = w2_idx % 2
                val_w2 = W2[w2_idx, r2]
                
                for w3_idx in nz_w3[r3]:
                    k3 = w3_idx // 3
                    i3 = w3_idx % 3
                    val_w3 = W3[w3_idx, r3]
                    
                    k6 = k2 * 3 + k3
                    i6 = i2 * 3 + i3
                    g_w = k6 * 6 + i6
                    W6[g_w, r] += val_w2 * val_w3

    # Safety check: ensure we didn't exceed time (though this is O(1) construction)
    if time.time() > deadline:
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
