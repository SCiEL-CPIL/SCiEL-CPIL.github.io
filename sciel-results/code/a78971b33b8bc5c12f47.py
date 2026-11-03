import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_factors():
    """Return list of (u, v, w) for 2x2 x 2x2 = 2x2, rank 7.
    u, v, w are lists of 4 elements (flattened 2x2)."""
    factors = []
    # s1 = (a00+a11)(b00+b11) -> c00, c11
    factors.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # s2 = (a10+a11)b00 -> c10
    factors.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # s3 = a00(b01-b11) -> c01
    factors.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # s4 = a11(b10-b00) -> c00
    factors.append(([0,0,0,1], [0,-1,1,0], [1,0,0,0]))
    # s5 = (a00+a01)b11 -> c00, c01
    factors.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # s6 = (a10-a00)(b00+b01) -> c11
    factors.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # s7 = (a01-a11)(b10+b11) -> c00, c11
    factors.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1]))
    return factors

def _strassen_4x4_factors():
    """Return list of (u, v, w) for 4x4 x 4x4 = 4x4, rank 49.
    u, v, w are lists of 16 elements (flattened 4x4, index = row*4+col)."""
    s2 = _strassen_2x2_factors()
    
    # Block indices in 4x4 flattened:
    # A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
    # Same for B and C.
    
    # Strassen at block level:
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    # M2 = (A21+A22)B11 -> C21
    # M3 = A11(B12-B22) -> C12
    # M4 = A22(B21-B11) -> C11
    # M5 = (A11+A12)B22 -> C11, C12
    # M6 = (A21-A11)(B11+B12) -> C22
    # M7 = (A12-A22)(B21+B22) -> C11, C22
    
    # Each Mi is a 2x2 matrix product, decomposed via s2.
    
    # For each block combination, I need to map the 2x2 Strassen factors
    # to the 4x4 indices.
    
    def block_to_4x4(block_indices, vec_2x2):
        """Map a 2x2 vector (4 elements) to 4x4 (16 elements) using block_indices."""
        result = [0]*16
        for i, idx in enumerate(block_indices):
            result[idx] = vec_2x2[i]
        return result
    
    A11_idx = [0,1,4,5]
    A12_idx = [2,3,6,7]
    A21_idx = [8,9,12,13]
    A22_idx = [10,11,14,15]
    B11_idx = [0,1,4,5]
    B12_idx = [2,3,6,7]
    B21_idx = [8,9,12,13]
    B22_idx = [10,11,14,15]
    C11_idx = [0,1,4,5]
    C12_idx = [2,3,6,7]
    C21_idx = [8,9,12,13]
    C22_idx = [10,11,14,15]
    
    factors = []
    
    # M1 = (A11+A22)(B11+B22)
    # u for A11+A22: sum of block vectors
    for u2, v2, w2 in s2:
        u4 = block_to_4x4(A11_idx, u2)
        for i, idx in enumerate(A22_idx):
            u4[idx] += u2[i]
        v4 = block_to_4x4(B11_idx, v2)
        for i, idx in enumerate(B22_idx):
            v4[idx] += v2[i]
        w4 = [0]*16
        for i, idx in enumerate(C11_idx):
            w4[idx] += w2[i]
        for i, idx in enumerate(C22_idx):
            w4[idx] += w2[i]
        factors.append((u4, v4, w4))
    
    # M2 = (A21+A22)B11
    for u2, v2, w2 in s2:
        u4 = block_to_4x4(A21_idx, u2)
        for i, idx in enumerate(A22_idx):
            u4[idx] += u2[i]
        v4 = block_to_4x4(B11_idx, v2)
        w4 = block_to_4x4(C21_idx, w2)
        factors.append((u4, v4, w4))
    
    # M3 = A11(B12-B22)
    for u2, v2, w2 in s2:
        u4 = block_to_4x4(A11_idx, u2)
        v4 = block_to_4x4(B12_idx, v2)
        for i, idx in enumerate(B22_idx):
            v4[idx] -= v2[i]
        w4 = block_to_4x4(C12_idx, w2)
        factors.append((u4, v4, w4))
    
    # M4 = A22(B21-B11)
    for u2, v2, w2 in s2:
        u4 = block_to_4x4(A22_idx, u2)
        v4 = block_to_4x4(B21_idx, v2)
        for i, idx in enumerate(B11_idx):
            v4[idx] -= v2[i]
        w4 = block_to_4x4(C11_idx, w2)
        factors.append((u4, v4, w4))
    
    # M5 = (A11+A12)B22
    for u2, v2, w2 in s2:
        u4 = block_to_4x4(A11_idx, u2)
        for i, idx in enumerate(A12_idx):
            u4[idx] += u2[i]
        v4 = block_to_4x4(B22_idx, v2)
        w4 = block_to_4x4(C11_idx, w2)
        for i, idx in enumerate(C12_idx):
            w4[idx] += w2[i]
        factors.append((u4, v4, w4))
    
    # M6 = (A21-A11)(B11+B12)
    for u2, v2, w2 in s2:
        u4 = block_to_4x4(A21_idx, u2)
        for i, idx in enumerate(A11_idx):
            u4[idx] -= u2[i]
        v4 = block_to_4x4(B11_idx, v2)
        for i, idx in enumerate(B12_idx):
            v4[idx] += v2[i]
        w4 = block_to_4x4(C22_idx, w2)
        factors.append((u4, v4, w4))
    
    # M7 = (A12-A22)(B21+B22)
    for u2, v2, w2 in s2:
        u4 = block_to_4x4(A12_idx, u2)
        for i, idx in enumerate(A22_idx):
            u4[idx] -= u2[i]
        v4 = block_to_4x4(B21_idx, v2)
        for i, idx in enumerate(B22_idx):
            v4[idx] += v2[i]
        w4 = block_to_4x4(C11_idx, w2)
        for i, idx in enumerate(C22_idx):
            w4[idx] += w2[i]
        factors.append((u4, v4, w4))
    
    return factors

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k' else 0
    # Check: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] == T[...]
    for i in range(n):
        for j in range(m):
            for jp in range(m*p):
                for kn in range(p*n):
                    j_prime = jp // p
                    k = jp % p
                    k_prime = kn // n
                    i_prime = kn % n
                    expected = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                    actual = sum(U[i*m+j][c] * V[jp][c] * W[kn][c] for c in range(len(U[0])))
                    if abs(actual - expected) > 1e-9:
                        return False, (i, j, j_prime, k, k_prime, i_prime, expected, actual)
    return True, None

def _construct_114():
    """Construct the 114-rank decomposition for 4x4 x 4x9."""
    n, m, p = 4, 4, 9
    
    # Split B (4x9) into B4 (4x4) and B5 (4x5)
    # B4: cols 0-3, B5: cols 4-8
    # C = A*B4 + A*B5
    # A*B4: 4x4 x 4x4, rank 49
    # A*B5: 4x4 x 4x5, split into B4' (4x4, cols 4-7) and B1 (4x1, col 8)
    # A*B4': 4x4 x 4x4, rank 49
    # A*B1: 4x4 x 4x1, rank 16
    # Total: 49 + 49 + 16 = 114
    
    factors = []
    
    # Part 1: A * B4 (cols 0-3 of B)
    s4_factors = _strassen_4x4_factors()
    for u4, v4, w4 in s4_factors:
        # u4: 16 elements for A (4x4)
        # v4: 16 elements for B4 (4x4), but B4 is cols 0-3 of B (4x9)
        # In the full B (4x9), B4[j,k] is at index j*9+k for k in 0..3
        # v4 is indexed as j*4+k for j in 0..3, k in 0..3
        # Need to map to j*9+k
        v_full = [0]*(m*p)  # 36 elements
        for j in range(4):
            for k in range(4):
                v_full[j*p+k] = v4[j*4+k]
        # w4: 16 elements for C (4x4), but C is 4x9
        # C[i,k] for k in 0..3 is at index k*4+i in W
        # w4 is indexed as k*4+i for k in 0..3, i in 0..3
        # This is already in the right format for W (p*n = 9*4 = 36, but we only use k=0..3)
        w_full = [0]*(p*n)  # 36 elements
        for k in range(4):
            for i in range(4):
                w_full[k*n+i] = w4[k*4+i]
        factors.append((u4, v_full, w_full))
    
    # Part 2: A * B4' (cols 4-7 of B)
    for u4, v4, w4 in s4_factors:
        v_full = [0]*(m*p)
        for j in range(4):
            for k in range(4):
                v_full[j*p+(k+4)] = v4[j*4+k]
        w_full = [0]*(p*n)
        for k in range(4):
            for i in range(4):
                w_full[(k+4)*n+i] = w4[k*4+i]
        factors.append((u4, v_full, w_full))
    
    # Part 3: A * B1 (col 8 of B)
    # 4x4 x 4x1: C[i,8] = sum_j A[i,j] * B[j,8]
    # Rank 16: for each (i,j), one factor
    for i in range(4):
        for j in range(4):
            u_full = [0]*16
            u_full[i*m+j] = 1
            v_full = [0]*(m*p)
            v_full[j*p+8] = 1
            w_full = [0]*(p*n)
            w_full[8*n+i] = 1
            factors.append((u_full, v_full, w_full))
    
    rank = len(factors)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(factors):
        for i in range(n*m):
            U[i][c] = u[i]
        for i in range(m*p):
            V[i][c] = v[i]
        for i in range(p*n):
            W[i][c] = w[i]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Try the 114-rank construction
    result = _construct_114()
    
    # Verify
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    ok, err = _verify_decomposition(U, V, W, n, m, p)
    
    if not ok:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return result
