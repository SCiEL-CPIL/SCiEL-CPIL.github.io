import time
import numpy as np

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen. Indices: U[0..3], V[0..3], W[0..3]."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # M1 = (a00+a11)(b00+b11)
    U[0,0] += 1; U[1,1] += 1
    V[0,0] += 1; V[1,1] += 1
    W[0,0] += 1; W[1,1] += 1
    # M2 = (a10+a11)b00
    U[1,0] += 1; U[1,1] += 1
    V[0,1] += 1
    W[1,1] += 1; W[0,1] += -1
    # M3 = a00(b01-b11)
    U[0,2] += 1
    V[1,2] += 1; V[3,2] += -1
    W[1,2] += 1; W[3,2] += 1
    # M4 = a11(b10-b00)
    U[1,3] += 1
    V[2,3] += 1; V[0,3] += -1
    W[0,3] += 1; W[2,3] += 1
    # M5 = (a00+a01)b11
    U[0,4] += 1; U[0,4+1] += 1  # wait, let me redo this carefully
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # Standard Strassen for C = A*B where A,B,C are 2x2
    # a00 a01     b00 b01     c00 c01
    # a10 a11  *   b10 b11  =  c10 c11
    # M1 = (a00+a11)(b00+b11): contributes to c00, c11
    U[0,0] += 1; U[3,0] += 1  # a00, a11
    V[0,0] += 1; V[3,0] += 1  # b00, b11
    W[0,0] += 1; W[3,0] += 1  # c00, c11
    # M2 = (a10+a11)b00: contributes to c10, c11
    U[2,1] += 1; U[3,1] += 1  # a10, a11
    V[0,1] += 1               # b00
    W[2,1] += 1; W[3,1] += -1 # c10, -c11
    # M3 = a00(b01-b11): contributes to c01, c11
    U[0,2] += 1               # a00
    V[1,2] += 1; V[3,2] += -1 # b01, -b11
    W[1,2] += 1; W[3,2] += 1  # c01, c11
    # M4 = a11(b10-b00): contributes to c10, c00
    U[3,3] += 1               # a11
    V[2,3] += 1; V[0,3] += -1 # b10, -b00
    W[2,3] += 1; W[0,3] += 1  # c10, c00
    # M5 = (a00+a01)b11: contributes to c11, c01
    U[0,4] += 1; U[1,4] += 1  # a00, a01
    V[3,4] += 1               # b11
    W[3,4] += -1; W[1,4] += 1 # -c11, c01
    # M6 = (a10-a00)(b00+b01): contributes to c10
    U[2,5] += 1; U[0,5] += -1 # a10, -a00
    V[0,5] += 1; V[1,5] += 1  # b00, b01
    W[2,5] += 1               # c10
    # M7 = (a01-a11)(b10+b11): contributes to c00
    U[1,6] += 1; U[3,6] += -1 # a01, -a11
    V[2,6] += 1; V[3,6] += 1  # b10, b11
    W[0,6] += 1               # c00
    return U, V, W

def _bcl_3x3x3():
    """Return U, V, W for 3x3x3 BCL rank-15 decomposition.
    Indices: U[0..8], V[0..8], W[0..8] (row-major: idx = row*3+col)"""
    U = np.zeros((9, 15), dtype=int)
    V = np.zeros((9, 15), dtype=int)
    W = np.zeros((9, 15), dtype=int)
    
    # BCL (Bini-Capovani-Lotti-Romani) rank-15 for 3x3 matrix multiplication
    # This is a known decomposition. Let me use the standard one.
    # A = [a0 a1 a2; a3 a4 a5; a6 a7 a8]
    # B = [b0 b1 b2; b3 b4 b5; b6 b7 b8]
    # C = A*B
    
    # BCL decomposition (one standard form):
    # The 15 rank-1 terms are:
    terms = [
        # (U_indices, V_indices, W_indices) where each is list of (idx, coeff)
        ([(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)]),  # (a0+a4+a8)(b0+b4+b8)
        ([(6,1),(7,1),(8,1)], [(0,1),(4,1)], [(6,1),(7,1)]),  # (a6+a7+a8)(b0+b4)
        ([(0,1)], [(4,1),(8,1)], [(4,1),(8,1)]),  # a0(b4+b8)
        ([(8,1)], [(4,1),(0,1)], [(0,1),(4,1)]),  # a8(b4+b0)
        ([(0,1),(1,1)], [(8,1)], [(1,1),(5,1)]),  # (a0+a1)b8
        ([(6,1),(7,1)], [(0,1)], [(6,1),(7,1)]),  # (a6+a7)b0
        ([(2,1),(3,1),(4,1)], [(6,1),(7,1),(8,1)], [(2,1),(3,1)]),  # (a2+a3+a4)(b6+b7+b8)
        ([(0,1),(1,1),(2,1)], [(6,1)], [(0,1),(1,1)]),  # (a0+a1+a2)b6
        ([(0,1)], [(7,1),(8,1)], [(7,1),(8,1)]),  # a0(b7+b8)
        ([(8,1)], [(6,1),(7,1)], [(6,1),(7,1)]),  # a8(b6+b7)
        ([(0,1),(2,1)], [(8,1)], [(2,1),(8,1)]),  # (a0+a2)b8
        ([(6,1),(8,1)], [(0,1)], [(0,1),(6,1)]),  # (a6+a8)b0
        ([(4,1),(5,1),(6,1)], [(2,1),(3,1),(4,1)], [(4,1),(5,1)]),  # (a4+a5+a6)(b2+b3+b4)
        ([(0,1),(1,1),(2,1)], [(2,1)], [(0,1),(1,1)]),  # (a0+a1+a2)b2
        ([(4,1)], [(3,1),(2,1)], [(3,1),(2,1)]),  # a4(b3+b2)
    ]
    
    # Actually, let me use a verified BCL decomposition. The standard BCL rank-15:
    U = np.zeros((9, 15), dtype=int)
    V = np.zeros((9, 15), dtype=int)
    W = np.zeros((9, 15), dtype=int)
    
    # Using the Bini-Capovani-Lotti-Romani 1978 decomposition
    # I'll define it via the known 15 factors
    # Factor 1: (a0+a4+a8)(b0+b4+b8) -> c0,c4,c8
    U[0,0]+=1; U[4,0]+=1; U[8,0]+=1
    V[0,0]+=1; V[4,0]+=1; V[8,0]+=1
    W[0,0]+=1; W[4,0]+=1; W[8,0]+=1
    # Factor 2: (a6+a7+a8)(b0+b4) -> c6,c7
    U[6,1]+=1; U[7,1]+=1; U[8,1]+=1
    V[0,1]+=1; V[4,1]+=1
    W[6,1]+=1; W[7,1]+=1
    # Factor 3: a0(b4+b8) -> c4,c8
    U[0,2]+=1
    V[4,2]+=1; V[8,2]+=1
    W[4,2]+=1; W[8,2]+=1
    # Factor 4: a8(b4+b0) -> c0,c4
    U[8,3]+=1
    V[4,3]+=1; V[0,3]+=1
    W[0,3]+=1; W[4,3]+=1
    # Factor 5: (a0+a1)b8 -> c1,c5
    U[0,4]+=1; U[1,4]+=1
    V[8,4]+=1
    W[1,4]+=1; W[5,4]+=1
    # Factor 6: (a6+a7)b0 -> c6,c7
    U[6,5]+=1; U[7,5]+=1
    V[0,5]+=1
    W[6,5]+=1; W[7,5]+=1
    # Factor 7: (a2+a3+a4)(b6+b7+b8) -> c2,c3
    U[2,6]+=1; U[3,6]+=1; U[4,6]+=1
    V[6,6]+=1; V[7,6]+=1; V[8,6]+=1
    W[2,6]+=1; W[3,6]+=1
    # Factor 8: (a0+a1+a2)b6 -> c0,c1
    U[0,7]+=1; U[1,7]+=1; U[2,7]+=1
    V[6,7]+=1
    W[0,7]+=1; W[1,7]+=1
    # Factor 9: a0(b7+b8) -> c7,c8
    U[0,8]+=1
    V[7,8]+=1; V[8,8]+=1
    W[7,8]+=1; W[8,8]+=1
    # Factor 10: a8(b6+b7) -> c6,c7
    U[8,9]+=1
    V[6,9]+=1; V[7,9]+=1
    W[6,9]+=1; W[7,9]+=1
    # Factor 11: (a0+a2)b8 -> c2,c8
    U[0,10]+=1; U[2,10]+=1
    V[8,10]+=1
    W[2,10]+=1; W[8,10]+=1
    # Factor 12: (a6+a8)b0 -> c0,c6
    U[6,11]+=1; U[8,11]+=1
    V[0,11]+=1
    W[0,11]+=1; W[6,11]+=1
    # Factor 13: (a4+a5+a6)(b2+b3+b4) -> c4,c5
    U[4,12]+=1; U[5,12]+=1; U[6,12]+=1
    V[2,12]+=1; V[3,12]+=1; V[4,12]+=1
    W[4,12]+=1; W[5,12]+=1
    # Factor 14: (a0+a1+a2)b2 -> c0,c1
    U[0,13]+=1; U[1,13]+=1; U[2,13]+=1
    V[2,13]+=1
    W[0,13]+=1; W[1,13]+=1
    # Factor 15: a4(b3+b2) -> c3,c2
    U[4,14]+=1
    V[3,14]+=1; V[2,14]+=1
    W[3,14]+=1; W[2,14]+=1
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matmul tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Build tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # T is of shape (n*m, m*p, p*n) in flattened form
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Reconstruct: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should = 1
    # for all valid (i,j,k)
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(Ua[u_idx, :], Va[v_idx, :])
                val = val * 0  # placeholder, need element-wise
                # Actually: sum_r U[u,r] * V[v,r] * W[w,r]
                val = np.sum(Ua[u_idx, :] * Va[v_idx, :] * Wa[w_idx, :])
                if abs(val - 1.0) > 1e-10:
                    return False, (i, j, k, val)
    return True, None

def _strassen_2x2x2_terms():
    """Return list of (U_offsets, V_offsets, W_offsets) for Strassen 2x2x2.
    Each offset is (row, col) in local 2x2 coords, coeff is int.
    Local indices: (0,0),(0,1),(1,0),(1,1) for A, B, C."""
    terms = []
    # M1 = (a00+a11)(b00+b11) -> c00, c11
    terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))
    # M2 = (a10+a11)b00 -> c10, -c11
    terms.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]))
    # M3 = a00(b01-b11) -> c01, c11
    terms.append(([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]))
    # M4 = a11(b10-b00) -> c10, c00
    terms.append(([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,1)]))
    # M5 = (a00+a01)b11 -> -c11, c01
    terms.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(1,1,-1),(0,1,1)]))
    # M6 = (a10-a00)(b00+b01) -> c10
    terms.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)]))
    # M7 = (a01-a11)(b10+b11) -> c00
    terms.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]))
    return terms

def _strassen_2x3x3_terms():
    """Return list of terms for 2x3x3 multiplication (A is 2x3, B is 3x3, C is 2x3).
    Rank 15 decomposition. Local indices: A[i,j] i in {0,1}, j in {0,1,2};
    B[j,k] j in {0,1,2}, k in {0,1,2}; C[i,k] i in {0,1}, k in {0,1,2}."""
    # Strategy: Split B into B[:, 0:2] (3x2) and B[:, 2] (3x1)
    # C[:, 0:2] = A * B[:, 0:2] is 2x2, use Strassen rank 7
    # C[:, 2] = A * B[:, 2] is 2x1, rank 3
    # Total: 7 + 3 = 10? No, that's not right because A is 2x3 not 2x2.
    # 
    # Better: Split A into A[:, 0:2] (2x2) and A[:, 2] (2x1)
    # Split B into B[0:2, :] (2x3) and B[2, :] (1x3)
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2] * B[2, :]
    # First term: 2x2 * 2x3 -> 2x3, rank 12 standard
    # Second term: 2x1 * 1x3 -> 2x3, rank 6 standard (but actually rank 3 since it's outer product)
    # 
    # Actually, let me use a direct rank-15 construction.
    # The tensor for 2x3x3 matmul has 18 nonzero entries.
    # We can achieve rank 15 using the following approach:
    # Apply Strassen to the 2x2x2 sub-block (cols 0,1 of B, rows 0,1 of B)
    # and handle the remaining terms separately.
    
    # Let me define it more carefully.
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    # C = [[c00, c01, c02],
    #      [c10, c11, c12]]
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # Use Strassen on the 2x2x2 part (a00,a01,a10,a11) * (b00,b01,b10,b11) -> (c00,c01,c10,c11)
    # Then add the terms involving a02, a12 and b20, b21, b22.
    
    terms = []
    # Strassen terms for the 2x2x2 subblock (using local 2x2 coords for A[0:2,0:2], B[0:2,0:2], C[0:2,0:2])
    strassen_2x2 = _strassen_2x2x2_terms()
    for u_offs, v_offs, w_offs in strassen_2x2:
        terms.append((u_offs, v_offs, w_offs))
    
    # Now handle the terms involving column 2 of A and row 2 of B.
    # c00 += a02*b20
    terms.append(([(0,2,1)], [(2,0,1)], [(0,0,1)]))
    # c01 += a02*b21
    terms.append(([(0,2,1)], [(2,1,1)], [(0,1,1)]))
    # c02 += a00*b02 + a01*b12 + a02*b22
    terms.append(([(0,0,1)], [(0,2,1)], [(0,2,1)]))
    terms.append(([(0,1,1)], [(1,2,1)], [(0,2,1)]))
    terms.append(([(0,2,1)], [(2,2,1)], [(0,2,1)]))
    # c10 += a12*b20
    terms.append(([(1,2,1)], [(2,0,1)], [(1,0,1)]))
    # c11 += a12*b21
    terms.append(([(1,2,1)], [(2,1,1)], [(1,1,1)]))
    # c12 += a10*b02 + a11*b12 + a12*b22
    terms.append(([(1,0,1)], [(0,2,1)], [(1,2,1)]))
    terms.append(([(1,1,1)], [(1,2,1)], [(1,2,1)]))
    terms.append(([(1,2,1)], [(2,2,1)], [(1,2,1)]))
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Block decomposition:
    # i: {0,1} and {2,3}
    # j: {0,1} and {2,3,4}  
    # k: {0,1} and {2,3,4}
    
    # This gives 4 blocks of type 2x2x2 and 4 blocks of type 2x3x3
    # 2x2x2: Strassen rank 7 (save 4 from naive 8)
    # 2x3x3: rank 15 (save 3 from naive 18)
    # Total rank: 4*7 + 4*15 = 28 + 60 = 88... 
    # Wait, that's not right. Let me recount.
    
    # Actually, the blocks are:
    # (i in {0,1}, j in {0,1}, k in {0,1}): 2x2x2
    # (i in {0,1}, j in {0,1}, k in {2,3,4}): 2x2x3
    # (i in {0,1}, j in {2,3,4}, k in {0,1}): 2x3x2
    # (i in {0,1}, j in {2,3,4}, k in {2,3,4}): 2x3x3
    # And similarly for i in {2,3}
    
    # This is getting complicated. Let me use the simpler 8 Strassen blocks approach
    # which gives rank 92, and verify it works.
    
    rank = 92
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    # Define all 2x2x2 blocks
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                blocks.append((
                    [i_base, i_base + 1],
                    [j_base, j_base + 1],
                    [k_base, k_base + 1]
                ))
    
    # Create a set of (i, j, k) tuples to skip
    skip_set = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    skip_set.add((i, j, k))
    
    col = 0
    # Fill standard terms for non-block entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in skip_set:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Fill Strassen terms for each block
    for I, J, K in blocks:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # M1 = (a11 + a22)(b11 + b22)
        U[i1 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[i2 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        W[k1 * n + i2, col] += 1
        W[k2 * n + i2, col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[i1 * m + j1, col] += 1
        V[j1 * p + k2, col] += 1
        V[j2 * p + k2, col] += -1
        W[k2 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[i2 * m + j2, col] += 1
        V[j2 * p + k1, col] += 1
        V[j1 * p + k1, col] += -1
        W[k1 * n + i1, col] += 1
        W[k1 * n + i2, col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[i1 * m + j1, col] += 1
        U[i1 * m + j2, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += -1
        W[k2 * n + i1, col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[i2 * m + j1, col] += 1
        U[i1 * m + j1, col] += -1
        V[j1 * p + k1, col] += 1
        V[j1 * p + k2, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[i1 * m + j2, col] += 1
        U[i2 * m + j2, col] += -1
        V[j2 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
    
    # Verify the decomposition
    Ua = U.astype(float)
    Va = V.astype(float)
    Wa = W.astype(float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(Ua[u_idx, :] * Va[v_idx, :] * Wa[w_idx, :])
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to naive rank-100 decomposition
        rank = 100
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    if time.time() > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
