"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen formulas:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: A00+A11, B00+B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: A10+A11, B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00, B01-B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11, B10-B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: A00+A01, B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: A10-A00, B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: A01-A11, B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    # Strassen terms: for each term t, list of (li, lj, lk, coeff_A, coeff_B)
    # A block at (li, lj), B block at (lj, lk)
    strassen_AB = [
        # M1: (A00+A11)(B00+B11)
        [(0, 0, 0, 1, 1), (1, 1, 1, 1, 1)],
        # M2: (A10+A11)B00
        [(1, 0, 0, 1, 1), (1, 1, 0, 1, 1)],
        # M3: A00(B01-B11)
        [(0, 0, 1, 1, 1), (0, 1, 1, 1, -1)],
        # M4: A11(B10-B00)
        [(1, 1, 0, 1, 1), (1, 0, 0, 1, -1)],
        # M5: (A00+A01)B11
        [(0, 0, 1, 1, 1), (0, 1, 1, 1, 1)],
        # M6: (A10-A00)(B00+B01)
        [(1, 0, 0, 1, 1), (1, 0, 1, 1, 1),
         (0, 0, 0, -1, 1), (0, 1, 0, -1, 1)],
        # M7: (A01-A11)(B10+B11)
        [(0, 1, 0, 1, 1), (0, 1, 1, 1, 1),
         (1, 1, 0, -1, 1), (1, 1, 1, -1, 1)],
    ]
    
    # Strassen C coefficients: for each term t, list of (li, lk, coeff_C)
    strassen_C = [
        # M1 -> C00, C11
        [(0, 0, 1), (1, 1, 1)],
        # M2 -> C10, C11
        [(1, 0, 1), (1, 1, -1)],
        # M3 -> C01, C11
        [(0, 1, 1), (1, 1, 1)],
        # M4 -> C00, C10
        [(0, 0, -1), (1, 0, 1)],
        # M5 -> C00, C01
        [(0, 0, 1), (0, 1, 1)],
        # M6 -> C11
        [(1, 1, 1)],
        # M7 -> C00
        [(0, 0, 1)],
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            
            # Accumulate U and V contributions from all (A_block, B_block) pairs
            # We need to handle multiple contributions to the same (ei, ej) or (ej, ek)
            # So we accumulate into temporary arrays
            
            # U contributions: indexed by (li, lj, ei, ej)
            # V contributions: indexed by (lj, lk, ej, ek)
            # W contributions: indexed by (li, lk, ei, ek)
            
            # For U: for each (li, lj, lk, cA, cB) in strassen_AB[t]:
            #   for ei, ej: U[2*li+ei, 2*lj+ej] += cA * inner_U[ei*2+ej, r]
            # But multiple terms may contribute to the same U position, so accumulate
            
            u_acc = np.zeros((4, 4))
            v_acc = np.zeros((4, 4))
            w_acc = np.zeros((4, 4))
            
            for (li, lj, lk, cA, cB) in strassen_AB[t]:
                for ei in range(2):
                    for ej in range(2):
                        u_acc[2*li+ei, 2*lj+ej] += cA * inner_U[ei*2+ej, r]
                        v_acc[2*lj+ej, 2*lk+ei] += cB * inner_V[ej*2+ei, r]
            
            for (li, lk, cC) in strassen_C[t]:
                for ei in range(2):
                    for ek in range(2):
                        w_acc[2*li+ei, 2*lk+ek] += cC * inner_W[ek*2+ei, r]
            
            # Now write accumulated values
            for gi in range(4):
                for gj in range(4):
                    U[gi*4+gj, col] = u_acc[gi, gj]
                    V[gi*4+gj, col] = v_acc[gi, gj]
                    W[gi*4+gj, col] = w_acc[gi, gj]
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _decomp_4x4x5():
    """
    Decomposition of M_{4,4,5} using one 4x4x4 Strassen block (rank 49) 
    plus one 4x4x1 schoolbook block (rank 16).
    Total rank: 65.
    U: (16, 65), V: (20, 65), W: (20, 65)
    """
    n, m, p = 4, 4, 5
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U444, V444, W444 = _decomp_4x4x4()
    U441, V441, W441 = _schoolbook_nm_p(4, 4, 1)
    
    # Place 4x4x4 part in first 49 columns
    U[:, :49] = U444
    # For 4x4x4 part, k ranges 0..3
    for j in range(m):
        for k in range(4):
            V[j * 5 + k, :49] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * 4 + i, :49] = W444[k * 4 + i, :]
    
    # Place 4x4x1 part in next 16 columns (for k=4)
    col_start = 49
    U[:, col_start:col_start+16] = U441
    for j in range(m):
        V[j * 5 + 4, col_start:col_start+16] = V441[j * 1 + 0, :]
    for i in range(n):
        W[4 * 4 + i, col_start:col_start+16] = W441[0 * 4 + i, :]
    
    return U, V, W

def _decomp_4x4x9_combined():
    """
    Decomposition of M_{4,4,9} by splitting into M_{4,4,5} (rank 65) + M_{4,4,4} (rank 49).
    Total rank: 114.
    U: (16, 114), V: (36, 114), W: (36, 114)
    """
    n, m, p = 4, 4, 9
    rank_5 = 65
    rank_4 = 49
    total_rank = rank_5 + rank_4
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # First part: 4x4x5 covering k=0..4
    U1, V1, W1 = _decomp_4x4x5()
    U[:, :65] = U1
    
    # V1 has shape (20, 65) for m=4, p=5
    # Map k=0..4 to output k=0..4
    for j in range(m):
        for k in range(5):
            V[j * 9 + k, :65] = V1[j * 5 + k, :]
    
    # W1 has shape (20, 65) for p=5, n=4
    # Map k=0..4 to output k=0..4
    for k in range(5):
        for i in range(n):
            W[k * 4 + i, :65] = W1[k * 4 + i, :]
    
    # Second part: 4x4x4 covering k=5..8
    U2, V2, W2 = _decomp_4x4x4()
    U[:, 65:] = U2
    
    # V2 has shape (16, 49) for m=4, p=4
    # Map k=0..3 to output k=5..8
    for j in range(m):
        for k in range(4):
            V[j * 9 + (5 + k), 65:] = V2[j * 4 + k, :]
    
    # W2 has shape (16, 49) for p=4, n=4
    # Map k=0..3 to output k=5..8
    for k in range(4):
        for i in range(n):
            W[(5 + k) * 4 + i, 65:] = W2[k * 4 + i, :]
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, atol=1e-9):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Sum over rank: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1
    # Use einsum for efficiency
    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    
    max_err = np.max(np.abs(result - 1.0))
    return max_err < atol, max_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Primary approach: 4x4x5 + 4x4x4 = rank 65 + 49 = 114
    U, V, W = _decomp_4x4x9_combined()
    
    if time.time() < deadline - 10:
        is_valid, max_err = _verify_decomp(U, V, W, n, m, p)
        if is_valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try alternative: 4x4x4 + 4x4x4 + 4x4x1 = rank 49+49+16 = 114
    if time.time() < deadline - 10:
        n, m, p = 4, 4, 9
        total_rank = 49 + 49 + 16
        U2 = np.zeros((n * m, total_rank))
        V2 = np.zeros((m * p, total_rank))
        W2 = np.zeros((p * n, total_rank))
        
        # Part 1: 4x4x4 for k=0..3
        U444, V444, W444 = _decomp_4x4x4()
        U2[:, :49] = U444
        for j in range(m):
            for k in range(4):
                V2[j * 9 + k, :49] = V444[j * 4 + k, :]
        for k in range(4):
            for i in range(n):
                W2[k * 4 + i, :49] = W444[k * 4 + i, :]
        
        # Part 2: 4x4x4 for k=4..7
        U2[:, 49:98] = U444
        for j in range(m):
            for k in range(4):
                V2[j * 9 + (4 + k), 49:98] = V444[j * 4 + k, :]
        for k in range(4):
            for i in range(n):
                W2[(4 + k) * 4 + i, 49:98] = W444[k * 4 + i, :]
        
        # Part 3: 4x4x1 for k=8
        U441, V441, W441 = _schoolbook_nm_p(4, 4, 1)
        U2[:, 98:] = U441
        for j in range(m):
            V2[j * 9 + 8, 98:] = V441[j * 1 + 0, :]
        for i in range(n):
            W2[8 * 4 + i, 98:] = W441[0 * 4 + i, :]
        
        is_valid2, max_err2 = _verify_decomp(U2, V2, W2, n, m, p)
        if is_valid2:
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Final fallback: schoolbook (rank 144)
    Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
