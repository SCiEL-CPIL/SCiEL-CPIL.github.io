import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: Decompose 6x6 = (3x2) x (2x3) using mixed radix 3 and 2.
    # This allows us to combine a 3x3x3 decomposition with a 2x2x2 decomposition.
    # Rank = R3 * R2.
    
    # 1. Define 2x2x2 Strassen Decomposition
    # Indices: I2 = i0*2 + j0, J2 = j0*2 + k0, K2 = k0*2 + i0
    # i0, j0, k0 in {0, 1}
    # Strassen's 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Output C_ik:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    R2 = 7
    U2 = np.zeros((4, R2), dtype=np.int8)
    V2 = np.zeros((4, R2), dtype=np.int8)
    W2 = np.zeros((4, R2), dtype=np.int8)
    
    # r=0: P1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1  # C00 (k0=0, i0=0)
    W2[3, 0] = 1  # C11 (k0=1, i0=1)
    
    # r=1: P2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1  # C10 (k0=1, i0=0)
    W2[3, 1] = -1 # C11 (k0=1, i0=1)
    
    # r=2: P3
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1  # C01 (k0=0, i0=1)
    W2[3, 2] = 1  # C11 (k0=1, i0=1)
    
    # r=3: P4
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1  # C00 (k0=0, i0=0)
    W2[1, 3] = 1  # C10 (k0=1, i0=0)
    
    # r=4: P5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C00 (k0=0, i0=0)
    W2[2, 4] = 1  # C01 (k0=0, i0=1)
    
    # r=5: P6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1  # C11 (k0=1, i0=1)
    
    # r=6: P7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1  # C00 (k0=0, i0=0)
    
    # 2. Define 3x3x3 Decomposition
    # We can use a known rank-reduced algorithm for 3x3 or just schoolbook rank 27.
    # Best known rank for 3x3 is 19 (Pan's algorithm) or 23 (others).
    # Implementing Pan's algorithm correctly is complex.
    # Let's check if we can implement a simple rank 19 or 20 if available, or stick to 27.
    # Given time constraints, implementing a verified rank 27 is safest.
    # However, "STAGNATION" suggests we need a structural change or improvement.
    # Current score 0.0 means the previous attempt failed.
    # The inspiration says 1.1429 is the seed. Score 1.0 = seed.
    # Seed likely uses Schoolbook 6x6 (Rank 216).
    # Strassen 6x6 via 3x2 decomposition: 27 * 7 = 189.
    # 189 < 216, so this should be an improvement (Score > 1.0).
    # Can we do better?
    # Rank 3x3 is 19 (Pan). 19 * 7 = 133.
    # Or maybe use a 6x6 specific algorithm?
    # Let's try to implement Pan's 19-rank 3x3 algorithm.
    # Reference: "On the multiplication of 3x3 matrices" by Pan.
    # It involves 19 multiplications.
    # The coefficients are integers.
    
    # Let's construct the 3x3 factor matrices for Pan's algorithm.
    # I will define U3, V3, W3 for 3x3x3.
    # Indices: I3 = i1*3 + j1, J3 = j1*3 + k1, K3 = k1*3 + i1.
    # i1, j1, k1 in {0, 1, 2}.
    
    # This is complex to hardcode from memory.
    # Alternative: Use a known rank 20 or 21 algorithm?
    # Actually, there is a well-known rank 19 algorithm.
    # Let's try to find a simpler decomposition first to ensure correctness, then optimize if time permits.
    # Schoolbook 3x3 has rank 27.
    # If I use Schoolbook 3x3, total rank is 189.
    # If I can find a rank 24 3x3 algorithm, total is 168.
    # If I find rank 21, total is 147.
    # If I find rank 19, total is 133.
    
    # Let's implement Schoolbook 3x3 for now to guarantee a valid, improved solution (189 < 216).
    # Then I will try to swap in a lower-rank 3x3 if I can code it reliably.
    # Actually, I can try to use a "block Strassen" on 3x3? No, Strassen is for even dimensions.
    # There is a rank 20 algorithm for 3x3.
    
    # Let's stick to Schoolbook 3x3 (Rank 27) first to ensure a non-zero score.
    # 189 is definitely better than 216.
    
    R3 = 27
    U3 = np.zeros((9, R3), dtype=np.int8)
    V3 = np.zeros((9, R3), dtype=np.int8)
    W3 = np.zeros((9, R3), dtype=np.int8)
    
    col = 0
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                U3[i1*3 + j1, col] = 1
                V3[j1*3 + k1, col] = 1
                W3[k1*3 + i1, col] = 1
                col += 1
                
    # Total Rank
    R_total = R2 * R3
    
    # 3. Combine U2, V2, W2 and U3, V3, W3 into U, V, W for 6x6x6
    # Indices for 6x6:
    # i = 2*i1 + i0, j = 2*j1 + j0
    # I = i*6 + j
    # i1, j1 in {0,1,2}, i0, j0 in {0,1}
    
    # Mapping:
    # i1 = i // 2, i0 = i % 2
    # j1 = j // 2, j0 = j % 2
    
    U = np.zeros((36, R_total), dtype=np.int8)
    V = np.zeros((36, R_total), dtype=np.int8)
    W = np.zeros((36, R_total), dtype=np.int8)
    
    # Iterate over all 6x6 indices and rank indices
    # r = r2 * R3 + r3
    # r2 in 0..6, r3 in 0..26
    
    for i in range(6):
        i1 = i // 2
        i0 = i % 2
        
        for j in range(6):
            j1 = j // 2
            j0 = j % 0
            
            I = i * 6 + j
            
            # U contribution: U2[i0*2+j0, r2] * U3[i1*3+j1, r3]
            I2 = i0 * 2 + j0
            I3 = i1 * 3 + j1
            
            for r2 in range(R2):
                u2_val = U2[I2, r2]
                if u2_val == 0: continue
                for r3 in range(R3):
                    u3_val = U3[I3, r3]
                    if u3_val == 0: continue
                    r = r2 * R3 + r3
                    U[I, r] = u2_val * u3_val

    # V indices: J = j*6 + k
    # j = 2*j1 + j0, k = 2*k1 + k0
    # J2 = j0*2 + k0, J3 = j1*3 + k1
    for j in range(6):
        j1 = j // 2
        j0 = j % 2
        
        for k in range(6):
            k1 = k // 2
            k0 = k % 2
            
            J = j * 6 + k
            J2 = j0 * 2 + k0
            J3 = j1 * 3 + k1
            
            for r2 in range(R2):
                v2_val = V2[J2, r2]
                if v2_val == 0: continue
                for r3 in range(R3):
                    v3_val = V3[J3, r3]
                    if v3_val == 0: continue
                    r = r2 * R3 + r3
                    V[J, r] = v2_val * v3_val

    # W indices: K = k*6 + i
    # k = 2*k1 + k0, i = 2*i1 + i0
    # K2 = k0*2 + i0, K3 = k1*3 + i1
    for k in range(6):
        k1 = k // 2
        k0 = k % 2
        
        for i in range(6):
            i1 = i // 2
            i0 = i % 2
            
            K = k * 6 + i
            K2 = k0 * 2 + i0
            K3 = k1 * 3 + i1
            
            for r2 in range(R2):
                w2_val = W2[K2, r2]
                if w2_val == 0: continue
                for r3 in range(R3):
                    w3_val = W3[K3, r3]
                    if w3_val == 0: continue
                    r = r2 * R3 + r3
                    W[K, r] = w2_val * w3_val

    # Optional: Check time and potentially try to reduce rank if possible?
    # No, 189 is a solid improvement over 216.
    # 189/216 = 0.875. Score = 216/189 = 1.1428...
    # This matches the inspiration score of 1.1429.
    
    # Verification (Optional, skipped for speed in final output)
    # We can verify a small subset to ensure no index errors.
    if time.time() - start_time < 5:
        # Check one element of the tensor decomposition
        # T[i,j,k] = sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
        # Should be 1 for all i,j,k.
        err = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    val = 0
                    for r in range(R_total):
                        val += U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    if val != 1:
                        err += 1
        if err > 0:
            # Fallback to schoolbook 6x6 if something went wrong (should not happen)
            U_fb = np.zeros((36, 216), dtype=np.int8)
            V_fb = np.zeros((36, 216), dtype=np.int8)
            W_fb = np.zeros((36, 216), dtype=np.int8)
            col = 0
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        U_fb[i*6+j, col] = 1
                        V_fb[j*6+k, col] = 1
                        W_fb[k*6+i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
