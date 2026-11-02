import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # Rank 56 is derived from 8 blocks of 2x2x2, each with rank 7 (Strassen)
    rank = 56
    
    # Initialize U, V, W
    # U shape: (n*m, rank) = (4, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (32, 56)
    # Note: The task description says W shape is (p*n, rank).
    # Trace convention W[k*n+i, r] implies index is k*n + i.
    # For p=16, n=2, indices are 0..31.
    # C[i, k] -> W[k*n + i].
    # Wait, standard C is n x p. C[i, k].
    # The tensor index for C is (i, k).
    # The prompt says W[k*n+i, r]. This maps C[i,k] to row k*n+i.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen rank-7 decomposition vectors for 2x2 * 2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    u_vecs = [
        [1, 0, -1, 0],   # M1: (A00 - A10)
        [1, 0, 1, 0],    # M2: (A00 + A10)
        [0, 1, 0, 0],    # M3: A01
        [0, 0, 0, 1],    # M4: A11
        [1, 1, 0, 0],    # M5: (A00 + A01)
        [0, 0, 1, -1],   # M6: (A10 - A11)
        [0, 1, 0, 1]     # M7: (A01 + A11)
    ]
    
    v_vecs = [
        [1, 0, 1, 0],    # M1: (B00 + B10)
        [0, 0, 1, 0],    # M2: B10
        [0, 1, 0, -1],   # M3: (B01 - B00)  -- Wait, standard Strassen M3 is A01 * (B10 - B00)? 
                         # Let's re-verify Strassen vectors.
                         # M1 = (A00-A10)(B00+B10)
                         # M2 = (A00+A10)B10
                         # M3 = A01(B10-B00)
                         # M4 = A11(B00-B10)
                         # M5 = (A00+A01)B00
                         # M6 = (A10-A11)(B00+B11)
                         # M7 = (A01+A11)(B10-B11)
                         # 
                         # My previous v_vecs in the prompt were:
                         # v1: [1, 0, 0, 1] -> B00+B10? No, indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
                         # B00 is idx 0. B10 is idx 2.
                         # So [1, 0, 1, 0] is B00+B10.
                         # The prompt's v_vecs[0] was [1, 0, 0, 1] which is B00+B11.
                         # Let's check the prompt's u_vecs[0]: [1, 0, 0, 1] -> A00+A11.
                         # This looks like a different Strassen variant or an error in the prompt's comment vs code.
                         # I will use the standard Strassen vectors.
    ]
    
    # Correct Standard Strassen Vectors
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:10, 3:11
    
    u_vecs = [
        [1, 0, -1, 0],   # M1: A00 - A10
        [1, 0, 1, 0],    # M2: A00 + A10
        [0, 1, 0, 0],    # M3: A01
        [0, 0, 0, 1],    # M4: A11
        [1, 1, 0, 0],    # M5: A00 + A01
        [0, 0, 1, -1],   # M6: A10 - A11
        [0, 1, 0, 1]     # M7: A01 + A11
    ]
    
    v_vecs = [
        [1, 0, 1, 0],    # M1: B00 + B10
        [0, 0, 1, 0],    # M2: B10
        [0, 1, 0, -1],   # M3: B01 - B00
        [1, 0, -1, 0],   # M4: B00 - B10
        [1, 0, 0, 0],    # M5: B00
        [1, 0, 0, 1],    # M6: B00 + B11
        [0, 0, 1, -1]    # M7: B10 - B11
    ]
    
    w_vecs = [
        [1, 0, 0, 1],    # M1: C00 + C11
        [0, 0, 1, -1],   # M2: C10 - C11
        [0, 1, 0, 1],    # M3: C01 + C11
        [1, 0, 1, 0],    # M4: C00 + C10
        [-1, 1, 0, 0],   # M5: -C00 + C01
        [0, 0, 0, 1],    # M6: C11
        [1, 0, 0, 0]     # M7: C00
    ]
    
    # Verify Strassen correctness for one block
    # C00 = M1 - M2 + M4 + M5 - M7?
    # Let's check W vectors sum for C00 (index 0).
    # w1[0]=1, w2[0]=0, w3[0]=0, w4[0]=1, w5[0]=-1, w6[0]=0, w7[0]=1.
    # Sum: M1 + M4 - M5 + M7.
    # M1 = (A00-A10)(B00+B10)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B00
    # M7 = (A01+A11)(B10-B11)
    # This doesn't look like C00 = A00B00 + A01B10.
    # Let's re-evaluate the W vectors.
    # Standard Strassen:
    # C00 = M1 - M2 + M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M3 + M5 - M6
    
    # Let's check my W vectors against this.
    # C00 (idx 0):
    # w1[0]=1 (M1)
    # w2[0]=0 (M2) -> Should be -1?
    # w3[0]=0 (M3)
    # w4[0]=1 (M4)
    # w5[0]=-1 (M5) -> Should be +1?
    # w6[0]=0 (M6)
    # w7[0]=1 (M7) -> Should be -1?
    
    # My W vectors in the prompt:
    # w1: [1, 0, 0, 1] -> C00, C11
    # w2: [0, 0, 1, -1] -> C10, -C11
    # w3: [0, 1, 0, 1] -> C01, C11
    # w4: [1, 0, 1, 0] -> C00, C10
    # w5: [-1, 1, 0, 0] -> -C00, C01
    # w6: [0, 0, 0, 1] -> C11
    # w7: [1, 0, 0, 0] -> C00
    
    # Let's sum for C00 (idx 0):
    # M1: 1
    # M2: 0
    # M3: 0
    # M4: 1
    # M5: -1
    # M6: 0
    # M7: 1
    # Total: M1 + M4 - M5 + M7.
    
    # Let's expand:
    # M1 = (A00-A10)(B00+B10) = A00B00 + A00B10 - A10B00 - A10B10
    # M4 = A11(B00-B10) = A11B00 - A11B10
    # M5 = (A00+A01)B00 = A00B00 + A01B00
    # M7 = (A01+A11)(B10-B11) = A01B10 - A01B11 + A11B10 - A11B11
    
    # Sum = (A00B00 + A00B10 - A10B00 - A10B10) + (A11B00 - A11B10) - (A00B00 + A01B00) + (A01B10 - A01B11 + A11B10 - A11B11)
    # = A00B00 + A00B10 - A10B00 - A10B10 + A11B00 - A11B10 - A00B00 - A01B00 + A01B10 - A01B11 + A11B10 - A11B11
    # = A00B10 - A10B00 - A10B10 + A11B00 - A01B00 + A01B10 - A01B11 - A11B11
    # This is NOT C00 = A00B00 + A01B10.
    
    # The W vectors in the prompt are WRONG for standard Strassen.
    # However, the prompt says the current program scores 1.1429, which implies it is VALID.
    # This means the U, V, W vectors in the prompt MUST form a valid decomposition, even if they don't match the standard Strassen formula I recalled.
    # Or, the prompt's code is correct and my recall of Strassen W vectors is wrong.
    # Let's check the prompt's U and V vectors again.
    # u1: [1, 0, 0, 1] -> A00 + A11
    # v1: [1, 0, 0, 1] -> B00 + B11
    # M1 = (A00+A11)(B00+B11)
    
    # u2: [0, 0, 1, 1] -> A10 + A11
    # v2: [1, 0, 0, 0] -> B00
    # M2 = (A10+A11)B00
    
    # u3: [1, 0, 0, 0] -> A00
    # v3: [0, 1, 0, -1] -> B01 - B10
    # M3 = A00(B01-B10)
    
    # u4: [0, 0, 0, 1] -> A11
    # v4: [-1, 0, 1, 0] -> B10 - B00
    # M4 = A11(B10-B00)
    
    # u5: [1, 1, 0, 0] -> A00 + A01
    # v5: [0, 0, 0, 1] -> B11
    # M5 = (A00+A01)B11
    
    # u6: [-1, 0, 1, 0] -> A10 - A00
    # v6: [1, 1, 0, 0] -> B00 + B01
    # M6 = (A10-A00)(B00+B01)
    
    # u7: [0, 1, 0, -1] -> A01 - A11
    # v7: [0, 0, 1, 1] -> B10 + B11
    # M7 = (A01-A11)(B10+B11)
    
    # This is a different set of 7 products. It might be a valid Strassen variant.
    # Since the prompt says it works, I will USE THE PROMPT'S VECTORS.
    
    u_vecs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    v_vecs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    w_vecs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    # Fill U, V, W
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # U
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
                    
            # V
            # B indices for block r:
            # B[0, 2r] is V[2r]
            # B[0, 2r+1] is V[2r+1]
            # B[1, 2r] is V[16+2r]
            # B[1, 2r+1] is V[16+2r+1]
            v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_idx[li]][col] = v_vecs[s][li]
                    
            # W
            # C indices for block r (trace convention W[k*n+i, r])
            # C[0, 2r] is W[2r*2 + 0] = W[4r]
            # C[0, 2r+1] is W[(2r+1)*2 + 0] = W[4r+2]
            # C[1, 2r] is W[2r*2 + 1] = W[4r+1]
            # C[1, 2r+1] is W[(2r+1)*2 + 1] = W[4r+3]
            w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_idx[li]][col] = w_vecs[s][li]
                    
    # Convert to lists of lists as per interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
