"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s/150s wall-clock limits

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36

    # Build the schoolbook decomposition as a fallback
    schoolbook_rank = n * m * p  # 216
    
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Try Block-Strassen: partition 6x6 into 3x3 blocks of 2x2 submatrices
    # Each 2x2 block multiplication uses Strassen's rank-7 algorithm
    # Total rank = 3 * 3 * 3 * 7 = 189
    
    strassen_rank = 3 * 3 * 3 * 7  # 189
    
    if strassen_rank < best_rank:
        success = _build_block_strassen(n, m, p, deadline)
        if success is not None:
            U_found, V_found, W_found = success
            best_rank = strassen_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found

    # Convert to lists for the output format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}


def _build_block_strassen(n, m, p, deadline):
    """Build a block-Strassen decomposition for 6x6x6 matrix multiplication.
    
    Partition into 3x3 blocks of 2x2 submatrices.
    For each triple (bi, bk, bj), apply Strassen's rank-7 decomposition
    to the 2x2 block product A[bi,bk] * B[bk,bj] -> C[bi,bj].
    Total: 3 * 3 * 3 * 7 = 189 terms.
    
    The key insight: this is a tensor product of the 3x3 schoolbook 
    (rank 9, one term per (bi,bk,bj)) and the 2x2 Strassen (rank 7).
    """
    nm, mp, pn = n * m, m * p, p * n
    num_blocks = 3  # 6 / 2 = 3 blocks per dimension
    block_size = 2
    total_rank = num_blocks ** 3 * 7  # 189
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # Using the standard formulation:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each term s, we have:
    #   A_s = linear combination of a00,a01,a10,a11
    #   B_s = linear combination of b00,b01,b10,b11
    #   C_s = linear combination of c00,c01,c10,c11
    # such that sum_s A_s * B_s = sum_s C_s (as bilinear forms)
    
    # A coefficients: [a00, a01, a10, a11]
    # B coefficients: [b00, b01, b10, b11]
    # C coefficients: [c00, c01, c10, c11]
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1 = (a00+a11)(b00+b11), contributes to C00 and C11
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2 = (a10+a11)b00, contributes to C10 and -C11
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        # M3 = a00(b10-b11), contributes to C01 and C11
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),
        # M4 = a11(b01-b00), contributes to C00 and C10
        ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),
        # M5 = (a00+a01)b11, contributes to -C00 and C01
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        # M6 = (a10-a00)(b00+b01), contributes to C11
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),
        # M7 = (a01-a11)(b10+b11), contributes to C00
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]
    
    col = 0
    for bi in range(num_blocks):
        for bk in range(num_blocks):
            for bj in range(num_blocks):
                for s in range(7):
                    a_coeff, b_coeff, c_coeff = strassen_terms[s]
                    
                    # A factor: A[bi*bs+r, bk*bs+sc] for r,sc in {0,1}
                    # Index in a_coeff: r*2 + sc
                    for r in range(block_size):
                        for sc in range(block_size):
                            val = a_coeff[r * block_size + sc]
                            if val != 0:
                                row_a = bi * block_size + r
                                col_a = bk * block_size + sc
                                idx_a = row_a * m + col_a
                                U[idx_a, col] = val
                    
                    # B factor: B[bk*bs+r, bj*bs+sc] for r,sc in {0,1}
                    # Index in b_coeff: r*2 + sc
                    for r in range(block_size):
                        for sc in range(block_size):
                            val = b_coeff[r * block_size + sc]
                            if val != 0:
                                row_b = bk * block_size + r
                                col_b = bj * block_size + sc
                                idx_b = row_b * p + col_b
                                V[idx_b, col] = val
                    
                    # C factor: C[bi*bs+r, bj*bs+sc] for r,sc in {0,1}
                    # W index convention: W[k*n+i] where k=col, i=row
                    # So W[idx] where idx = col_C * n + row_C
                    # Index in c_coeff: r*2 + sc (r=row offset, sc=col offset)
                    for r in range(block_size):
                        for sc in range(block_size):
                            val = c_coeff[r * block_size + sc]
                            if val != 0:
                                row_c = bi * block_size + r
                                col_c = bj * block_size + sc
                                idx_w = col_c * n + row_c
                                W[idx_w, col] = val
                    
                    col += 1
    
    # Verify the decomposition
    if not _verify_decomposition(n, m, p, U, V, W):
        return None
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that U, V, W form a valid decomposition.
    
    The matrix multiplication tensor T[i,j,k] = 1 if the indices are consistent
    (i.e., A[i,j] * B[j,k] -> C[i,k]) and 0 otherwise.
    
    In the flattened representation:
    - A index: a = i*m + j
    - B index: b = j*p + k  
    - C index: c = k*n + i  (trace convention W[k*n+i])
    
    We need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    """
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    
    # Check all valid (i,j,k) triples: should give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check a sample of invalid triples: should give 0
    # Use deterministic sampling for reproducibility
    np.random.seed(42)
    num_checks = 3000
    for _ in range(num_checks):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        
        # Decode indices
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k1 = b % p
        k2 = c // n
        i2 = c % n
        
        # Skip if this is a valid triple
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    
    return True
# EVOLVE-BLOCK-END
