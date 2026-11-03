"""Tensor decomposition for 4x4 * 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    t0 = time.time()
    DEADLINE = 120.0
    
    n, m, p = 4, 4, 6
    
    def build_strassen(n, m, p):
        """Build Strassen decomposition for even dimensions n, m, p.
        Returns U (n*m, rank), V (m*p, rank), W (p*n, rank) as numpy arrays.
        """
        if n == 2 and m == 2 and p == 2:
            # Base case: 2x2 * 2x2 Strassen
            # U: 4x7, V: 4x7, W: 4x7
            # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # Strassen 7 terms for 2x2*2x2:
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
            
            # For 2x2, blocks are single elements
            # A11=A[0,0]=U idx 0, A12=A[0,1]=U idx 1, A21=A[1,0]=U idx 2, A22=A[1,1]=U idx 3
            # B11=B[0,0]=V idx 0, B12=B[0,1]=V idx 1, B21=B[1,0]=V idx 2, B22=B[1,1]=V idx 3
            # C11=C[0,0]=W idx 0, C12=C[0,1]=W idx 1, C21=C[1,0]=W idx 2, C22=C[1,1]=W idx 3
            
            terms = [
                # (A_indices, coeffs, B_indices, coeffs, C_indices, coeffs)
                ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),  # M1
                ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),        # M2
                ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),        # M3
                ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),        # M4
                ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),        # M5
                ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),        # M6
                ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),        # M7
            ]
            
            for t_idx, (a_idx, a_c, b_idx, b_c, c_idx, c_c) in enumerate(terms):
                for ai, ac in zip(a_idx, a_c):
                    U[ai, t_idx] += ac
                for bi, bc in zip(b_idx, b_c):
                    V[bi, t_idx] += bc
                for ci, cc in zip(c_idx, c_c):
                    W[ci, t_idx] += cc
            
            return U, V, W
        
        # Recursive case: split into halves
        n2, m2, p2 = n // 2, m // 2, p // 2
        U_sub, V_sub, W_sub = build_strassen(n2, m2, p2)
        rank_sub = U_sub.shape[1]
        total_rank = 7 * rank_sub
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Strassen terms with block structure
        # Blocks: 11, 12, 21, 22
        terms = [
            {'A': [(11, 1), (22, 1)], 'B': [(11, 1), (22, 1)], 'C': [(11, 1), (22, 1)]},
            {'A': [(21, 1), (22, 1)], 'B': [(11, 1)], 'C': [(21, 1), (22, -1)]},
            {'A': [(11, 1)], 'B': [(12, 1), (22, -1)], 'C': [(12, 1), (22, 1)]},
            {'A': [(22, 1)], 'B': [(21, 1), (11, -1)], 'C': [(11, 1), (21, 1)]},
            {'A': [(11, 1), (12, 1)], 'B': [(22, 1)], 'C': [(11, -1), (12, 1)]},
            {'A': [(21, 1), (11, -1)], 'B': [(11, 1), (12, 1)], 'C': [(22, 1)]},
            {'A': [(12, 1), (22, -1)], 'B': [(21, 1), (22, 1)], 'C': [(11, 1)]},
        ]
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            
            for c in range(rank_sub):
                # Process U entries
                for u_idx in range(n2 * m2):
                    u_val = U_sub[u_idx, c]
                    if u_val == 0:
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
                        U[g_u, base_col + c] += u_val * coeff
                
                # Process V entries
                for v_idx in range(m2 * p2):
                    v_val = V_sub[v_idx, c]
                    if v_val == 0:
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
                        V[g_v, base_col + c] += v_val * coeff
                
                # Process W entries
                for w_idx in range(p2 * n2):
                    w_val = W_sub[w_idx, c]
                    if w_val == 0:
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
                        W[g_w, base_col + c] += w_val * coeff
        
        return U, V, W
    
    def build_schoolbook(n, m, p):
        """Build schoolbook decomposition for n x m * m x p.
        Returns U (n*m, rank), V (m*p, rank), W (p*n, rank) as numpy arrays.
        """
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
        return U, V, W
    
    def verify(U, V, W, n, m, p):
        """Verify the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Split 4x4 * 4x6 into 4x4 * 4x4 and 4x4 * 4x2
    # 4x4 * 4x4: Strassen rank = 49
    # 4x4 * 4x2: Schoolbook rank = 32
    # Total rank = 49 + 32 = 81
    
    # Alternative: Use Strassen for the whole thing
    # 4x4 * 4x6: Split into 2x2 blocks
    # n=4, m=4, p=6 -> n2=2, m2=2, p2=3
    # But p2=3 is odd, so we can't directly use Strassen recursively
    
    # Let's try: 4x4 * 4x6
    # Split p=6 into p1=4 and p2=2
    # Part 1: 4x4 * 4x4 -> Strassen rank 49
    # Part 2: 4x4 * 4x2 -> Schoolbook rank 32
    # Total: 81
    
    U1, V1, W1 = build_strassen(4, 4, 4)
    U2, V2, W2 = build_schoolbook(4, 4, 2)
    
    # Concatenate horizontally
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    rank = U.shape[1]
    
    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = build_schoolbook(n, m, p)
        rank = U.shape[1]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
