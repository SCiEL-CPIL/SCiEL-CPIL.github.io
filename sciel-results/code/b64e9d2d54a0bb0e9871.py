"""Tensor decomposition for 4x4 * 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6

    # Build a rank-9 decomposition for 2x2 * 2x3 (A is 2x2, B is 2x3)
    # Using 7 Strassen terms for the 2x2 part and 2 extra for the 3rd column
    # Total rank = 9
    
    def build_2x2x3():
        """Return U (4,9), V (6,9), W (6,9) for 2x2 * 2x3."""
        n2, m2, p2 = 2, 2, 3
        rank = 9
        U = np.zeros((n2*m2, rank))
        V = np.zeros((m2*p2, rank))
        W = np.zeros((p2*n2, rank))
        
        # Strassen for 2x2 * 2x2 (first two columns of B)
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For the 3rd column (B13, B23), use schoolbook:
        # C13 = A11*B13 + A12*B23
        # C23 = A21*B13 + A22*B23
        
        # Strassen terms (columns 0-6)
        # Term 1: M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1  # A11, A22
        V[0, 0] = 1; V[3, 0] = 1  # B11, B22
        W[0, 0] = 1; W[4, 0] = 1  # C11, C22
        
        # Term 2: M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1  # A21, A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1; W[4, 1] = -1  # C21, C22
        
        # Term 3: M3 = A11(B12-B22)
        U[0, 2] = 1  # A11
        V[1, 2] = 1; V[3, 2] = -1  # B12, B22
        W[1, 2] = 1; W[4, 2] = 1  # C12, C22
        
        # Term 4: M4 = A22(B21-B11)
        U[3, 3] = 1  # A22
        V[2, 3] = 1; V[0, 3] = -1  # B21, B11
        W[0, 3] = 1; W[2, 3] = 1  # C11, C21
        
        # Term 5: M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1  # A11, A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1; W[1, 4] = 1  # C11, C12
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1  # A21, A11
        V[0, 5] = 1; V[1, 5] = 1  # B11, B12
        W[4, 5] = 1  # C22
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1  # A12, A22
        V[2, 6] = 1; V[3, 6] = 1  # B21, B22
        W[0, 6] = 1  # C11
        
        # Schoolbook for 3rd column (columns 7-8)
        # C13 = A11*B13 + A12*B23
        # C23 = A21*B13 + A22*B23
        
        # Term 8: A11*B13
        U[0, 7] = 1  # A11
        V[2, 7] = 1  # B13 (row 0, col 2 in B)
        W[4, 7] = 1  # C13 (row 2, col 0 in C)
        
        # Term 9: A12*B23
        U[1, 8] = 1  # A12
        V[5, 8] = 1  # B23 (row 1, col 2 in B)
        W[4, 8] = 1  # C13
        
        # Wait, I need to handle C23 as well. Let me reconsider.
        # Actually, for the 3rd column, I need 4 terms: A11*B13, A12*B23, A21*B13, A22*B23
        # But I only have 2 extra columns. Let me use a different approach.
        
        # Actually, let me just use schoolbook for the entire 2x2 * 2x3: rank = 12
        # But that's worse. Let me think again.
        
        # Better approach: use Strassen for 2x2 * 2x2 (rank 7) and then
        # handle the 3rd column separately with 2 multiplications:
        # C13 = A11*B13 + A12*B23 -> 2 terms
        # C23 = A21*B13 + A22*B23 -> 2 terms
        # Total: 7 + 4 = 11, not 9.
        
        # Let me just use rank 12 (schoolbook) for now, or find a better decomposition.
        # Actually, the minimum rank for 2x2 * 2x3 is 9 (known result).
        
        # Let me try a different decomposition:
        # Use 7 Strassen terms for the 2x2 part, and for the 3rd column:
        # C13 = A11*B13 + A12*B23
        # C23 = A21*B13 + A22*B23
        # This requires 4 additional terms, total 11.
        
        # Alternatively, I can combine: 
        # Let M8 = (A11+A12)B13, M9 = (A21+A22)B23
        # Then C13 = A11*B13 + A12*B23, C23 = A21*B13 + A22*B23
        # This doesn't directly help.
        
        # For now, let me just use the known rank-9 decomposition if I can find it,
        # or fall back to a higher rank.
        
        # Actually, let me just build the full 4x4 * 4x6 using a direct approach.
        # I'll use the fact that 4x4 * 4x6 can be decomposed as:
        # Split A into 2x2 blocks, B into 2x3 blocks.
        # There are 4 2x2 blocks in A and 4 2x3 blocks in B (2x2 grid of 2x3).
        # Each pair uses rank 9 (or 11), total 16 * 9 = 144 or 16 * 11 = 176.
        # That's worse than the current 77.
        
        # Let me stick with the recursive Strassen approach from the original.
        # The original had rank 77 (7 * 11 for the 2x2x3 base case).
        
        # Let me just return the rank-11 decomposition for 2x2 * 2x3.
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Strassen terms 0-6 (as above)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[4, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[4, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[4, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[4, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        # Schoolbook for 3rd column: 4 terms
        # Term 7: A11*B13
        U[0, 7] = 1
        V[2, 7] = 1
        W[4, 7] = 1  # C13
        
        # Term 8: A12*B23
        U[1, 8] = 1
        V[5, 8] = 1
        W[4, 8] = 1  # C13
        
        # Term 9: A21*B13
        U[2, 9] = 1
        V[2, 9] = 1
        W[5, 9] = 1  # C23
        
        # Term 10: A22*B23
        U[3, 10] = 1
        V[5, 10] = 1
        W[5, 10] = 1  # C23
        
        return U, V, W
    
    # Build the full 4x4 * 4x6 decomposition
    # Split into 2x2 blocks: A has 4 blocks (2x2), B has 4 blocks (2x3)
    # Each block pair uses the 2x2 * 2x3 decomposition (rank 11)
    # Total rank = 16 * 11 = 176, which is worse than 77.
    
    # Let me use the recursive Strassen approach from the original code.
    # The original achieved rank 77 = 7 * 11.
    
    # I'll just use the original approach with careful implementation.
    
    def build_decomp(n, m, p):
        """Build tensor decomposition for n x m * m x p matrix multiplication."""
        if n == 1 and m == 1 and p == 1:
            return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
        
        if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        W[k*n + i, col] = 1
                        col += 1
            return U, V, W
        
        # For 2x2 * 2x3, use the special base case
        if n == 2 and m == 2 and p == 3:
            return build_2x2x3()
        
        n2, m2, p2 = n // 2, m // 2, p // 2
        U_sub, V_sub, W_sub = build_decomp(n2, m2, p2)
        rank_sub = U_sub.shape[1]
        total_rank = 7 * rank_sub
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        terms = [
            {'A': [(11, 1), (22, 1)], 'B': [(11, 1), (22, 1)], 'C': [(11, 1), (22, 1)]},
            {'A': [(21, 1), (22, 1)], 'B': [(11, 1)], 'C': [(21, 1), (22, -1)]},
            {'A': [(11, 1)], 'B': [(12, 1), (22, -1)], 'C': [(12, 1), (22, 1)]},
            {'A': [(22, 1)], 'B': [(21, 1), (11, -1)], 'C': [(11, 1), (21, 1)]},
            {'A': [(11, 1), (12, 1)], 'B': [(22, 1)], 'C': [(11, -1), (12, 1)]},
            {'A': [(21, 1), (11, -1)], 'B': [(11, 1), (12, 1)], 'C': [(22, 1)]},
            {'A': [(12, 1), (22, -1)], 'B': [(21, 1), (22, 1)], 'C': [(11, 1)]}
        ]
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            
            for u_idx in range(n2 * m2):
                u_val = U_sub[u_idx]
                if np.all(u_val == 0):
                    continue
                i = u_idx // m2
                j = u_idx % m2
                for block, coeff in term['A']:
                    if block == 11:
                        g_u = i * m + j
                    elif block == 12:
                        g_u = i * m + (j + m2)
                    elif block == 21:
                        g_u = (i + n2) * m + j
                    elif block == 22:
                        g_u = (i + n2) * m + (j + m2)
                    else:
                        continue
                    U[g_u, base_col:base_col+rank_sub] += u_val * coeff
            
            for v_idx in range(m2 * p2):
                v_val = V_sub[v_idx]
                if np.all(v_val == 0):
                    continue
                j = v_idx // p2
                k = v_idx % p2
                for block, coeff in term['B']:
                    if block == 11:
                        g_v = j * p + k
                    elif block == 12:
                        g_v = j * p + (k + p2)
                    elif block == 21:
                        g_v = (j + m2) * p + k
                    elif block == 22:
                        g_v = (j + m2) * p + (k + p2)
                    else:
                        continue
                    V[g_v, base_col:base_col+rank_sub] += v_val * coeff
            
            for w_idx in range(p2 * n2):
                w_val = W_sub[w_idx]
                if np.all(w_val == 0):
                    continue
                k = w_idx // n2
                i = w_idx % n2
                for block, coeff in term['C']:
                    if block == 11:
                        g_w = k * n + i
                    elif block == 12:
                        g_w = (k + p2) * n + i
                    elif block == 21:
                        g_w = k * n + (i + n2)
                    elif block == 22:
                        g_w = (k + p2) * n + (i + n2)
                    else:
                        continue
                    W[g_w, base_col:base_col+rank_sub] += w_val * coeff
        
        return U, V, W
    
    U, V, W = build_decomp(n, m, p)
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    elapsed = time.time() - t0
    if elapsed > 140:
        # Fallback to schoolbook if we're running out of time
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
