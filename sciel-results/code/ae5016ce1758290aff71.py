import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 8
    
    # Strassen decomposition for 2x2 matrix multiplication
    # T = sum_r U_r (tensor) V_r (tensor) W_r
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices:
    # U/V: 00, 01, 10, 11 -> flat 0, 1, 2, 3
    # W: 00, 01, 10, 11 -> flat 0, 1, 2, 3
    
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # M3 = a00(b10-b11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # M4 = a11(b00-b01)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # M5 = (a00+a01)b11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    R = 56
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Structure:
    # We have 2 blocks of j (size 2 each).
    # For each j-block, we have 4 blocks of k (size 2 each).
    # Total 2 * 4 = 8 Strassen instances.
    
    j_blocks = [0, 2]
    k_blocks = [0, 2, 4, 6]
    
    r_idx = 0
    for jb in j_blocks:
        for kb in k_blocks:
            # Map A indices (2x2)
            # A is 2x4. Indices (i, j). 
            # For this block, i in {0,1}, j in {jb, jb+1}
            # Flat A index: i * 4 + j
            a_idx = []
            for i in range(2):
                for j in range(2):
                    a_idx.append(i * 4 + (jb + j))
            
            # Map B indices (2x2)
            # B is 4x8. Indices (j, k).
            # For this block, j in {jb, jb+1}, k in {kb, kb+1}
            # Flat B index: j * 8 + k
            b_idx = []
            for j in range(2):
                for k in range(2):
                    b_idx.append((jb + j) * 8 + (kb + k))
            
            # Map C indices (2x2)
            # C is 2x8. Indices (i, k).
            # For this block, i in {0,1}, k in {kb, kb+1}
            # Flat C index: k * 2 + i (Convention W[k*n+i, r])
            c_idx = []
            for k in range(2):
                for i in range(2):
                    c_idx.append((kb + k) * 2 + i)
            
            # Fill U, V, W for this Strassen instance
            for l in range(7):
                r = r_idx + l
                # U
                for flat_a, val in zip([0,1,2,3], U_s[:, l]):
                    if val != 0:
                        U[a_idx[flat_a], r] += val
                # V
                for flat_b, val in zip([0,1,2,3], V_s[:, l]):
                    if val != 0:
                        V[b_idx[flat_b], r] += val
                # W
                for flat_c, val in zip([0,1,2,3], W_s[:, l]):
                    if val != 0:
                        W[c_idx[flat_c], r] += val
            
            r_idx += 7
    
    # Deadline guard
    if time.time() - start_time > 140:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
