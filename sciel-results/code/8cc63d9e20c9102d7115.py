"""Matrix multiplication 4x4x7 - Strassen 4x4x4 + 4x(2x2x3) decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    def make_schoolbook(n, m, p):
        """Return schoolbook decomposition as (U, V, W) lists."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W form a valid tensor decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None

    def strassen_2x2_terms():
        """Return 7 rank-1 terms for 2x2x2 matrix multiplication.
        U: 4-dim indexed by (i,j) -> index i*2+j
        V: 4-dim indexed by (j,k) -> index j*2+k
        W: 4-dim indexed by (k,i) -> index k*2+i
        """
        terms = []
        
        # Term 1: A00*B00 -> C00
        U = np.zeros(4); U[0] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[0] = 1
        terms.append((U, V, W))
        
        # Term 2: A01*B11 -> C01
        U = np.zeros(4); U[1] = 1
        V = np.zeros(4); V[3] = 1
        W = np.zeros(4); W[2] = 1
        terms.append((U, V, W))
        
        # Term 3: A10*B00 -> C10
        U = np.zeros(4); U[2] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[1] = 1
        terms.append((U, V, W))
        
        # Term 4: A11*B10 -> C11
        U = np.zeros(4); U[3] = 1
        V = np.zeros(4); V[2] = 1
        W = np.zeros(4); W[3] = 1
        terms.append((U, V, W))
        
        # Term 5: (A00+A10)*B10 -> C01 + C11
        U = np.zeros(4); U[0] = 1; U[2] = 1
        V = np.zeros(4); V[2] = 1
        W = np.zeros(4); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        
        # Term 6: (A01+A11)*B00 -> C00 + C10
        U = np.zeros(4); U[1] = 1; U[3] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[0] = 1; W[1] = 1
        terms.append((U, V, W))
        
        # Term 7: (A00-A11)*(B01+B11) -> C01 + C11
        U = np.zeros(4); U[0] = 1; U[3] = -1
        V = np.zeros(4); V[1] = 1; V[3] = 1
        W = np.zeros(4); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        
        return terms

    def verify_2x2(terms):
        """Verify 2x2x2 decomposition."""
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for (U, V, W) in terms:
                        total += U[i*2+j] * V[j*2+k] * W[k*2+i]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x3_terms():
        """Return 11 rank-1 terms for 2x2x3 matrix multiplication.
        A is 2x2, B is 2x3, C is 2x3.
        U: 4-dim indexed by (i,j) -> i*2+j
        V: 6-dim indexed by (j,k) -> j*3+k
        W: 6-dim indexed by (k,i) -> k*2+i
        
        Strategy: Split B into B[:,0:2] (2x2) and B[:,2] (2x1)
        C = A @ B[:,0:2] + A @ B[:,2]
        First part: 2x2x2 Strassen (rank 7)
        Second part: 2x2x1 schoolbook (rank 4)
        Total: rank 11
        """
        terms = []
        
        # Part 1: 2x2x2 Strassen for B[:,0:2]
        # V index for B[j, k] where k=0,1: j*3 + k
        # W index for C[k, i] where k=0,1: k*2 + i
        s2 = strassen_2x2_terms()
        for (U2, V2, W2) in s2:
            U4 = np.zeros(4)
            V6 = np.zeros(6)
            W6 = np.zeros(6)
            
            for idx in range(4):
                U4[idx] = U2[idx]
            
            for idx in range(4):
                j, k = idx // 2, idx % 2
                V6[j * 3 + k] = V2[idx]
            
            for idx in range(4):
                k, i = idx // 2, idx % 2
                W6[k * 2 + i] = W2[idx]
            
            if np.any(U4 != 0) and np.any(V6 != 0) and np.any(W6 != 0):
                terms.append((U4, V6, W6))
        
        # Part 2: 2x2x1 schoolbook for B[:,2]
        # A[i,j] * B[j,2] -> C[i,2]
        # U index: i*2+j
        # V index: j*3+2
        # W index: 2*2+i = 4+i
        for i in range(2):
            for j in range(2):
                U4 = np.zeros(4); U4[i*2+j] = 1
                V6 = np.zeros(6); V6[j*3+2] = 1
                W6 = np.zeros(6); W6[4+i] = 1
                terms.append((U4, V6, W6))
        
        return terms

    def verify_2x2x3(terms):
        """Verify 2x2x3 decomposition."""
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    total = 0
                    for (U, V, W) in terms:
                        total += U[i*2+j] * V[j*3+k] * W[k*2+i]
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None

    def build_strassen_4x4x4():
        """Build rank-1 terms for 4x4x4 using Strassen."""
        
        def expand_with_signs(U2, V2, W2, A_factors, B_factors, C_factors):
            """Expand a 2x2x2 rank-1 term to 4x4x4 with signs."""
            U4 = np.zeros(16)
            V4 = np.zeros(16)
            W4 = np.zeros(16)
            
            for (block, sign) in A_factors:
                bi, bj = block
                for i in range(2):
                    for j in range(2):
                        U4[(2*bi + i)*4 + (2*bj + j)] += sign * U2[i*2 + j]
            
            for (block, sign) in B_factors:
                bi, bj = block
                for i in range(2):
                    for j in range(2):
                        V4[(2*bi + i)*4 + (2*bj + j)] += sign * V2[i*2 + j]
            
            for (block, sign) in C_factors:
                bi, bj = block
                for i in range(2):
                    for j in range(2):
                        W4[(2*bj + j)*4 + (2*bi + i)] += sign * W2[i*2 + j]
            
            return U4, V4, W4
        
        strassen_defs = [
            {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
            {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((1,0), 1), ((1,1), 1)]},
            {'A': [((0,0), 1)], 'B': [((1,0), 1), ((1,1), -1)], 'C': [((0,1), 1)]},
            {'A': [((1,1), 1)], 'B': [((0,1), 1), ((0,0), -1)], 'C': [((0,0), 1), ((1,0), 1)]},
            {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,1), 1), ((1,1), 1)]},
            {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,0), 1)]},
            {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        ]
        
        s2 = strassen_2x2_terms()
        terms = []
        for prod_def in strassen_defs:
            for (U2, V2, W2) in s2:
                U4, V4, W4 = expand_with_signs(
                    U2, V2, W2,
                    prod_def['A'], prod_def['B'], prod_def['C']
                )
                if np.any(U4 != 0) and np.any(V4 != 0) and np.any(W4 != 0):
                    terms.append((U4, V4, W4))
        
        return terms

    def verify_4x4x4(terms):
        """Verify 4x4x4 decomposition."""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0
                    for (U, V, W) in terms:
                        total += U[i*4+j] * V[j*4+k] * W[k*4+i]
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None

    def build_2x2x3_in_4x4x3(row_block, col_block):
        """Build 2x2x3 terms embedded in 4x4x3 space.
        row_block: 0 or 1 (which 2-row block of A and C)
        col_block: 0 or 1 (which 2-col block of A and B)
        The 3 columns of B/C are columns 4,5,6 (global) but locally 0,1,2.
        
        U: 16-dim, index (i,j) -> i*4+j, only rows in [2*row_block, 2*row_block+1] and cols in [2*col_block, 2*col_block+1]
        V: 28-dim, index (j,k) -> j*7+k, only rows in [2*col_block, 2*col_block+1] and cols 4,5,6
        W: 28-dim, index (k,i) -> k*4+i, only cols 4,5,6 and rows in [2*row_block, 2*row_block+1]
        """
        base_terms = strassen_2x2x3_terms()
        terms = []
        
        for (U2, V2, W2) in base_terms:
            U16 = np.zeros(16)
            V28 = np.zeros(28)
            W28 = np.zeros(28)
            
            # Embed U: local (i,j) -> global (2*row_block+i, 2*col_block+j)
            for i in range(2):
                for j in range(2):
                    U16[(2*row_block + i)*4 + (2*col_block + j)] = U2[i*2+j]
            
            # Embed V: local (j,k) -> global (2*col_block+j, 4+k)
            for j in range(2):
                for k in range(3):
                    V28[(2*col_block + j)*7 + (4 + k)] = V2[j*3+k]
            
            # Embed W: local (k,i) -> global (4+k, 2*row_block+i)
            for k in range(3):
                for i in range(2):
                    W28[(4 + k)*4 + (2*row_block + i)] = W2[k*2+i]
            
            if np.any(U16 != 0) and np.any(V28 != 0) and np.any(W28 != 0):
                terms.append((U16, V28, W28))
        
        return terms

    # Verify Strassen 2x2x2
    s2_terms = strassen_2x2_terms()
    s2_valid = verify_2x2(s2_terms)
    
    if not s2_valid:
        U, V, W, rank = make_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Verify 2x2x3
    s23_terms = strassen_2x2x3_terms()
    s23_valid, _ = verify_2x2x3(s23_terms)
    
    if not s23_valid:
        U, V, W, rank = make_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Build 4x4x4 Strassen decomposition
    strassen_4x4_terms = build_strassen_4x4x4()
    s4_valid, s4_err = verify_4x4x4(strassen_4x4_terms)

    if not s4_valid:
        U, V, W, rank = make_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Count ranks
    rank_4x4x4 = len(strassen_4x4_terms)  # should be 49
    rank_2x2x3 = len(s23_terms)  # should be 11
    rank_4x4x3 = 4 * rank_2x2x3  # 4 blocks * 11 = 44
    
    R = rank_4x4x4 + rank_4x4x3  # 49 + 44 = 93

    U = np.zeros((n * m, R))  # (16, 93)
    V = np.zeros((m * p, R))  # (28, 93)
    W = np.zeros((p * n, R))  # (28, 93)

    # Part 1: Strassen 4x4x4 (columns 0-3 of B and C)
    for idx, (U4, V4, W4) in enumerate(strassen_4x4_terms):
        if idx >= rank_4x4x4:
            break
        U[:16, idx] = U4
        V[:16, idx] = V4
        W[:16, idx] = W4

    # Part 2: Four 2x2x3 blocks for 4x4x3 (columns 4-6 of B and C)
    col_offset = rank_4x4x4
    for row_block in range(2):
        for col_block in range(2):
            block_terms = build_2x2x3_in_4x4x3(row_block, col_block)
            for (U16, V28, W28) in block_terms:
                if col_offset >= R:
                    break
                U[:, col_offset] = U16
                V[:, col_offset] = V28
                W[:, col_offset] = W28
                col_offset += 1

    # Verify full decomposition
    valid_full, err_full = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)

    if not valid_full:
        # Fall back to schoolbook
        U, V, W, rank = make_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
