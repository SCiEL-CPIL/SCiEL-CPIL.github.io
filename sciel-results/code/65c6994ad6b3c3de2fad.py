import numpy as np

def _kronecker_decomp(d1, d2):
    """
    d1: dict with 'U', 'V', 'W' for T1 (n1, m1, p1)
    d2: dict with 'U', 'V', 'W' for T2 (n2, m2, p2)
    Returns decomposition for T1 x T2 (n1*n2, m1*m2, p1*p2)
    """
    U1, V1, W1 = d1['U'], d1['V'], d1['W']
    U2, V2, W2 = d2['U'], d2['V'], d2['W']
    
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    # The rank is r1 * r2
    # We need to construct U, V, W of size (n1*n2, r1*r2), etc.
    
    # Using Kronecker product of matrices
    # U_comb[i1*n2 + i2, c1*r2 + c2] = U1[i1, c1] * U2[i2, c2]
    # This is exactly the Kronecker product of U1 and U2 if we reshape appropriately?
    # np.kron(U1, U2) has shape (n1*n2, r1*r2).
    # Element (i1*n2+i2, c1*r2+c2) is U1[i1,c1]*U2[i2,c2].
    # Yes.
    
    U_comb = np.kron(U1, U2)
    V_comb = np.kron(V1, V2)
    W_comb = np.kron(W1, W2)
    
    return {'U': U_comb, 'V': V_comb, 'W': W_comb}

def _sum_decomps(d1, d2):
    """
    Sum of two decompositions for the same tensor shape.
    Just concatenate columns.
    """
    return {
        'U': np.hstack([d1['U'], d2['U']]),
        'V': np.hstack([d1['V'], d2['V']]),
        'W': np.hstack([d1['W'], d2['W']])
    }

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {'U': U, 'V': V, 'W': W}

def _strassen_2x2x2():
    # Strassen decomposition for 2x2x2
    # Rank 7
    # U, V, W shapes: (4, 7), (4, 7), (4, 7)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C21 += M2, C22 -= M2
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C12 += M3, C22 += M3
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11 += M4, C21 += M4
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C11 -= M5, C12 += M5
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11 += M7
    
    return {'U': U, 'V': V, 'W': W}

def construct():
    # 1. Get M_{2,2,2} (Rank 7)
    d_222 = _strassen_2x2x2()
    
    # 2. Get M_{1,1,2} (Rank 2)
    d_112 = _schoolbook(1, 1, 2)
    
    # 3. M_{2,2,4} = M_{2,2,2} x M_{1,1,2} (Rank 14)
    d_224 = _kronecker_decomp(d_222, d_112)
    
    # 4. M_{2,2,3} (Rank 12)
    d_223 = _schoolbook(2, 2, 3)
    
    # 5. M_{2,2,7} = M_{2,2,4} + M_{2,2,3} (Rank 26)
    # Note: We are summing tensors of same shape (2,2,7)
    # M_{2,2,4} covers k=0..3
    # M_{2,2,3} covers k=0..2
    # This is NOT a valid sum for M_{2,2,7} because the indices don't align simply as a block diagonal.
    # Wait. M_{2,2,4} is for p=4. M_{2,2,3} is for p=3.
    # We need M_{2,2,7}.
    # We can view M_{2,2,7} as M_{2,2,4} \oplus M_{2,2,3} if we treat the p-index as disjoint.
    # The tensor T_{2,2,7} has p-index 0..6.
    # T_{2,2,4} has p-index 0..3.
    # T_{2,2,3} has p-index 0..2.
    # If we embed T_{2,2,4} into T_{2,2,7} by zero-padding p=4..6, and T_{2,2,3} by zero-padding p=3..6,
    # then their sum is NOT T_{2,2,7}. It's T_{2,2,4} + T_{2,2,3} which has double counting in p=0..2?
    # No, we want to decompose T_{2,2,7}.
    # T_{2,2,7} = T_{2,2,4}^{(p=0..3)} + T_{2,2,3}^{(p=3..6)}?
    # The multiplication is linear in B and C.
    # C_{i,k} = \sum A_{i,j} B_{j,k}.
    # For k=0..3, it's part of the 4-wide multiplication.
    # For k=3..6, it's part of the 3-wide multiplication?
    # Yes, we can split the columns of B and C.
    # B = [B_1 | B_2] where B_1 is 2x4, B_2 is 2x3.
    # C = [C_1 | C_2].
    # C_1 = A B_1 (2x4 mult)
    # C_2 = A B_2 (2x3 mult)
    # So T_{2,2,7} = T_{2,2,4} \oplus T_{2,2,3}.
    # The decomposition is the block sum (horizontal concat of U,V,W).
    
    # We need to ensure the indices in U,V,W correspond to the correct slices.
    # For d_224, the V and W matrices have indices for p=0..3.
    # For d_223, the V and W matrices have indices for p=0..2.
    # When we concatenate, we need to shift the indices of d_223 to start at p=4?
    # No, the Kronecker product and sum logic works on the tensor components.
    # If we just hstack U, V, W, does it work?
    # The tensor defined by the hstack is the sum of the two tensors.
    # T_total = T_1 + T_2.
    # T_1 is supported on p=0..3 (others 0).
    # T_2 is supported on p=0..2 (others 0).
    # This sum is NOT T_{2,2,7}. It has T_{2,2,3} added to the first 3 columns, and T_{2,2,4} added to the first 4.
    # This is wrong.
    
    # We must shift the indices of the second part.
    # For d_223, we need to map its p-index 0..2 to 3..5 of the 7-index space?
    # Or 4..6?
    # If we use 0..3 for first, then 3..6 for second? Overlap at 3.
    # Better: 0..3 and 4..6.
    # So we need to shift the p-index of d_223 by 4.
    
    # How to shift?
    # In V and W, the p-index is involved.
    # V shape (m*p, rank). Index is j*p + k.
    # W shape (p*n, rank). Index is k*n + i.
    
    # We can create a "shifted" version of d_223.
    # But easier: Just construct d_223 for p=3, then manually adjust indices in V and W.
    # Or, use a helper to shift p-index.
    
    # Let's implement a shift function for p-index.
    def _shift_p(d, offset, new_p):
        # d is for (n, m, p_old)
        # new_p is the total p for the target tensor
        # offset is the starting p index
        n, m, p_old = d['U'].shape[0]//m, d['V'].shape[0]//d['U'].shape[0]//m, d['V'].shape[0]//m
        # Actually we know n, m, p_old from context.
        # Let's pass n, m, p_old.
        pass

    # Simpler: Just build d_223 with p=3, then create new V, W with zeros of size (m*7, r) and (7*n, r),
    # and copy the rows into the correct positions.
    
    n, m, p_target = 2, 2, 7
    
    # d_224 is for p=4.
    # V_224 has size (2*4, 14). Rows 0..7 correspond to j=0,1 and k=0..3.
    # We need to place this into V_total of size (2*7, 26).
    # Rows for k=0..3 should be at indices j*7 + k.
    
    # d_223 is for p=3.
    # V_223 has size (2*3, 12). Rows correspond to j=0,1 and k=0..2.
    # We need to place this into V_total, but shifted by 4 in k.
    # So k=0..2 should go to k=4..6.
    
    U_total = np.zeros((n*m, 26))
    V_total = np.zeros((m*p_target, 26))
    W_total = np.zeros((p_target*n, 26))
    
    # Copy d_224
    U_total[:, 0:14] = d_224['U']
    # Shift V
    for j in range(m):
        for k in range(4):
            V_total[j*p_target + k, 0:14] = d_224['V'][j*4 + k, :]
    # Shift W
    for k in range(4):
        for i in range(n):
            W_total[k*n + i, 0:14] = d_224['W'][k*n + i, :]
            
    # Copy d_223 (shifted by 4)
    U_total[:, 14:26] = d_223['U']
    for j in range(m):
        for k in range(3):
            V_total[j*p_target + (k+4), 14:26] = d_223['V'][j*3 + k, :]
    for k in range(3):
        for i in range(n):
            W_total[(k+4)*n + i, 14:26] = d_223['W'][k*n + i, :]
            
    d_227 = {'U': U_total, 'V': V_total, 'W': W_total}
    
    # 6. M_{1,2,1} (Rank 2)
    d_121 = _schoolbook(1, 2, 1)
    
    # 7. M_{2,4,7} = M_{2,2,7} x M_{1,2,1}
    d_final = _kronecker_decomp(d_227, d_121)
    
    return d_final

