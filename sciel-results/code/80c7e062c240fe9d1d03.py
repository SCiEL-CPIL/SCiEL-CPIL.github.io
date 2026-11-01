import numpy as np
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs rank-49 factor matrices for 4x4 matrix multiplication
    using the Kronecker product of 2x2 Strassen factors.
    """
    n, m, p = 4, 4, 4
    
    # 2x2 Strassen Factors
    # U2: maps A (4 entries) to 7 intermediate terms
    # A = [[a, b], [c, d]]
    # U2 indices: 0:a, 1:b, 2:c, 3:d
    U2 = np.array([
        [1, 0, 0, 0],  # m1 = (a+d)(e+h) -> a
        [0, 1, 0, 0],  # m1 = (a+d)(e+h) -> d
        [0, 0, 1, 0],  # m3 = c(a+b) -> c
        [0, 0, 0, 1],  # m4 = d(g+c) -> d
        [1, 1, 0, 0],  # m5 = (a+b)g -> a
        [0, 0, 1, -1], # m6 = h(c+d) -> c, d (wait, Strassen m6 is (c+d)(f-h)? No)
        [0, 1, 0, -1]  # m7 = (a-c)(d-e)? 
    ], dtype=float)
    
    # Let's use the standard Strassen formulas carefully.
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)(h)
    # m6 = (c+d)(g-f)
    # m7 = (a-c)(e+h)
    
    # U (A factors):
    # m1: a, d
    # m2: c, d
    # m3: a
    # m4: d
    # m5: a, b
    # m6: c, d
    # m7: a, c
    
    U2 = np.zeros((4, 7))
    U2[0, 0] = 1; U2[0, 6] = 1 # a in m1, m7
    U2[1, 0] = 1; U2[1, 4] = 1 # b in m1 (no, m1 is a+d), b in m5
    # Correction:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c+d)(g-f)
    # m7 = (a-c)(e+h)
    
    U2[0, :] = [1, 0, 1, 0, 1, 0, 1] # a: m1, m3, m5, m7
    U2[1, :] = [0, 0, 0, 0, 1, 0, 0] # b: m5
    U2[2, :] = [0, 1, 0, 0, 0, 1, -1] # c: m2, m6, m7
    U2[3, :] = [1, 1, 0, 1, 0, 1, 0] # d: m1, m2, m4, m6
    
    V2 = np.zeros((4, 7))
    # e: m1, m7
    # f: m3, m6
    # g: m4, m6
    # h: m1, m5, m7
    V2[0, :] = [1, 0, 0, 0, 0, 0, 1] # e
    V2[1, :] = [0, 0, 1, 0, 0, -1, 0] # f
    V2[2, :] = [0, 0, 0, 1, 0, 1, 0] # g
    V2[3, :] = [1, 0, 0, 0, 1, 0, 1] # h
    
    W2 = np.zeros((4, 7))
    # C indices: 0:c00, 1:c01, 2:c10, 3:c11
    # W index in task: k*n + i. For 2x2, k*2+i.
    # c00 (i=0,k=0) idx 0: m1 + m4 - m5 - m7
    # c01 (i=0,k=1) idx 2: m3 + m5
    # c10 (i=1,k=0) idx 1: m2 + m4
    # c11 (i=1,k=1) idx 3: m1 - m2 + m3 + m6
    
    # Task W shape: (p*n, rank) = (4, 7). Index is k*2+i.
    # k=0,i=0 -> 0
    # k=1,i=0 -> 2
    # k=0,i=1 -> 1
    # k=1,i=1 -> 3
    
    W2[0, :] = [1, 0, 0, 1, -1, 0, -1] # c00
    W2[1, :] = [0, 1, 0, 1, 0, 0, 0]   # c10 (idx 1)
    W2[2, :] = [0, 0, 1, 0, 1, 0, 0]   # c01 (idx 2)
    W2[3, :] = [1, -1, 1, 0, 0, 1, 0]  # c11 (idx 3)
    
    # Construct 4x4 factors using Kronecker Product
    # U4 = U2 ⊗ U2
    # V4 = V2 ⊗ V2
    # W4 = W2 ⊗ W2
    
    # The Kronecker product of U2 (4x7) and U2 (4x7) results in a (16x49) matrix.
    # The row index is i1*4 + i2? No, np.kron(A, B) for A(IxJ), B(KxL) gives IK x JL.
    # Row index r = iA * K + iB. Here K=4. So r = i1 * 4 + i2.
    # This matches the row-major order of A (4x4) where i = i1*2 + i2? 
    # Wait. A is 4x4. Indices 0..15.
    # If we view A as a 2x2 block matrix of 2x2 blocks:
    # A_{block(i1, i2, j1, j2)}
    # Row index in flattened 4x4 is (i1*2 + i2)*4 + (j1*2 + j2).
    # The Kronecker product row index is i1*4 + i2? No.
    # np.kron(U2, U2) row index is i1*4 + i2 where i1 is from first U2, i2 from second.
    # First U2 row index is i1 (0..3, representing 2x2 indices).
    # Second U2 row index is i2 (0..3).
    # So row index is i1*4 + i2.
    # This corresponds to flattening the 2x2 block indices (i1, i2) into a single index.
    # But the actual row index in 4x4 is (i1*2 + i2_row)*4 + (j1*2 + j2_col).
    # This is NOT the same as i1*4 + i2.
    
    # We need to permute the rows of the Kronecker product to match the standard row-major order.
    # Current row order in U4_kron: (i1, i2) where i1, i2 are 0..3.
    # i1 = 2*i1_block + i1_local, i2 = 2*i2_block + i2_local.
    # Standard row index: (2*i1_block + i1_local)*4 + (2*i2_block + i2_local)
    # = 8*i1_block + 4*i1_local + 4*i2_block + i2_local.
    
    # Let's generate the permutation vector.
    perm = np.zeros(16, dtype=int)
    for i1 in range(4):
        for i2 in range(4):
            i1_block = i1 // 2
            i1_local = i1 % 2
            i2_block = i2 // 2
            i2_local = i2 % 2
            std_idx = (i1_block * 2 + i1_local) * 4 + (i2_block * 2 + i2_local)
            kron_idx = i1 * 4 + i2
            perm[kron_idx] = std_idx
            # We want U4[std_idx, :] = U4_kron[kron_idx, :]
            # So we permute rows: U4 = U4_kron[perm_inv]
    
    # Actually, easier:
    # Create an identity matrix and permute it.
    P = np.eye(16)
    P = P[perm, :] # This maps kron_idx to std_idx?
    # P[kron_idx, std_idx] = 1.
    # We want U4[std_idx, :] = U4_kron[kron_idx, :].
    # So U4 = P @ U4_kron? No.
    # U4 = U4_kron[perm, :] if perm[kron_idx] = std_idx?
    # Let's check: U4_kron[perm, :] row i is U4_kron[perm[i], :].
    # We want row std_idx to be U4_kron[kron_idx, :].
    # So perm[std_idx] = kron_idx.
    
    perm_rev = np.zeros(16, dtype=int)
    for i1 in range(4):
        for i2 in range(4):
            i1_block = i1 // 2
            i1_local = i1 % 2
            i2_block = i2 // 2
            i2_local = i2 % 2
            std_idx = (i1_block * 2 + i1_local) * 4 + (i2_block * 2 + i2_local)
            kron_idx = i1 * 4 + i2
            perm_rev[std_idx] = kron_idx
            
    U4 = np.kron(U2, U2)[perm_rev, :]
    V4 = np.kron(V2, V2)[perm_rev, :]
    
    # W permutation
    # W2 indices: k*2 + i.
    # W4 indices: k*4 + i.
    # k = 2*k1 + k2, i = 2*i1 + i2.
    # std_idx = (2*k1 + k2)*4 + (2*i1 + i2) = 8*k1 + 4*k2 + 2*i1 + i2.
    # Kron row index: k1*4 + k2? No.
    # W2 is 4x7. Row index is k*2+i.
    # np.kron(W2, W2) row index is k1*4 + k2 where k1, k2 are row indices of W2 (0..3).
    # k1 = 2*k1_block + k1_local? No, k1 is the row index of the first W2, which represents (k1_block, i1_block, k1_local, i1_local)?
    # No, W2 row index is a single number 0..3 representing (k, i).
    # So kron row index is k1*4 + k2 where k1, k2 are the (k,i) pairs of the two 2x2 blocks.
    # k1 = 2*k1_block + k1_local? No.
    # Let's decompose the 2x2 indices.
    # First W2 row index r1 = 2*k1 + i1.
    # Second W2 row index r2 = 2*k2 + i2.
    # Kron row index = r1 * 4 + r2.
    # We want std_idx = (2*k1 + k2)*4 + (2*i1 + i2).
    
    perm_w = np.zeros(16, dtype=int)
    for r1 in range(4):
        for r2 in range(4):
            k1 = r1 // 2
            i1 = r1 % 2
            k2 = r2 // 2
            i2 = r2 % 2
            std_idx = (2*k1 + k2)*4 + (2*i1 + i2)
            kron_idx = r1 * 4 + r2
            perm_w[std_idx] = kron_idx
            
    W4 = np.kron(W2, W2)[perm_w, :]
    
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

# EVOLVE-BLOCK-END
