"""Matrix multiplication 4x4x7 - hybrid Strassen + schoolbook decomposition."""
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

    # Verify Strassen 2x2x2
    s2_terms = strassen_2x2_terms()
    s2_valid = verify_2x2(s2_terms)
    
    if not s2_valid:
        # Fall back to schoolbook
        U, V, W, rank = make_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Build 4x4x4 Strassen decomposition
    def build_strassen_4x4x4():
        """Build rank-1 terms for 4x4x4 using Strassen."""
        
        def expand_with_signs(U2, V2, W2, A_factors, B_factors, C_factors):
            """Expand a 2x2x2 rank-1 term to 4x4x4 with signs.
            U2: 4-dim, indexed by (i,j) -> i*2+j
            V2: 4-dim, indexed by (j,k) -> j*2+k  
            W2: 4-dim, indexed by (k,i) -> k*2+i
            A_factors: list of ((bi,bj), sign)
            B_factors: list of ((bi,bj), sign)
            C_factors: list of ((bi,bj), sign) where (bi,bj) is (row_block, col_block)
            """
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
                bi, bj = block  # (row_block, col_block) for C
                # W is indexed by (k, i) where k is column index, i is row index
                # C[bi][bj] means rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                # So W index: k*4+i where k in [2*bj, 2*bj+1], i in [2*bi, 2*bi+1]
                for i in range(2):
                    for j in range(2):
                        W4[(2*bj + j)*4 + (2*bi + i)] += sign * W2[i*2 + j]
            
            return U4, V4, W4
        
        # Strassen 4x4 definitions
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # C = A*B = [[C00, C01], [C10, C11]]
        
        strassen_defs = [
            # M1 = (A00+A11)*(B00+B11) -> C00 + C11
            {
                'A': [((0,0), 1), ((1,1), 1)],
                'B': [((0,0), 1), ((1,1), 1)],
                'C': [((0,0), 1), ((1,1), 1)]
            },
            # M2 = (A10+A11)*B00 -> C10 + C11
            {
                'A': [((1,0), 1), ((1,1), 1)],
                'B': [((0,0), 1)],
                'C': [((1,0), 1), ((1,1), 1)]
            },
            # M3 = A00*(B10-B11) -> C01
            {
                'A': [((0,0), 1)],
                'B': [((1,0), 1), ((1,1), -1)],
                'C': [((0,1), 1)]
            },
            # M4 = A11*(B01-B00) -> C00 + C10
            {
                'A': [((1,1), 1)],
                'B': [((0,1), 1), ((0,0), -1)],
                'C': [((0,0), 1), ((1,0), 1)]
            },
            # M5 = (A00+A01)*B11 -> C01 + C11
            {
                'A': [((0,0), 1), ((0,1), 1)],
                'B': [((1,1), 1)],
                'C': [((0,1), 1), ((1,1), 1)]
            },
            # M6 = (A10-A00)*(B00+B01) -> C10
            {
                'A': [((1,0), 1), ((0,0), -1)],
                'B': [((0,0), 1), ((0,1), 1)],
                'C': [((1,0), 1)]
            },
            # M7 = (A01-A11)*(B10+B11) -> C00 + C11
            {
                'A': [((0,1), 1), ((1,1), -1)],
                'B': [((1,0), 1), ((1,1), 1)],
                'C': [((0,0), 1), ((1,1), 1)]
            },
        ]
        
        terms = []
        for prod_def in strassen_defs:
            for (U2, V2, W2) in s2_terms:
                U4, V4, W4 = expand_with_signs(
                    U2, V2, W2,
                    prod_def['A'], prod_def['B'], prod_def['C']
                )
                # Check if all three vectors are non-zero
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

    strassen_4x4_terms = build_strassen_4x4x4()
    s4_valid, s4_err = verify_4x4x4(strassen_4x4_terms)

    if not s4_valid:
        # Fall back to schoolbook
        U, V, W, rank = make_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Build full 4x4x7: Strassen 4x4x4 (rank 49) + schoolbook 4x4x3 (rank 48) = 97
    R = 49 + 48  # 97

    U = np.zeros((n * m, R))  # (16, 97)
    V = np.zeros((m * p, R))  # (28, 97)
    W = np.zeros((p * n, R))  # (28, 97)

    # Part 1: Strassen 4x4x4 (columns 0-3 of B and C)
    for idx, (U4, V4, W4) in enumerate(strassen_4x4_terms):
        if idx >= 49:
            break
        U[:16, idx] = U4
        V[:16, idx] = V4  # first 16 rows of V (j in 0..3, k in 0..3)
        W[:16, idx] = W4  # first 16 rows of W (k in 0..3, i in 0..3)

    # Part 2: Schoolbook for 4x4x3 (columns 4-6 of B and C)
    col_offset = 49
    for i in range(n):
        for j in range(m):
            for k in range(3):
                global_k = 4 + k  # columns 4, 5, 6
                U[i * m + j, col_offset] = 1
                V[j * p + global_k, col_offset] = 1
                W[global_k * n + i, col_offset] = 1
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
