"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 150s wall-clock limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36

    # Build the schoolbook decomposition as a fallback (rank 216)
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
    # Total rank = 3*3*3*7 = 189
    strassen_rank = 189

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
    Each 2x2 matrix multiply uses Strassen's 7 scalar products.
    Total rank: 3*3*3*7 = 189.
    
    For each (bi, bj, bk) and Strassen term s:
    - U factor: linear combo of 4 elements in A[bi,bk]
    - V factor: linear combo of 4 elements in B[bk,bj]  
    - W factor: linear combo of 4 elements in C[bi,bj]
    """
    nm, mp, pn = n * m, m * p, p * n
    total_rank = 189

    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))

    # Strassen's 7 products for 2x2 matrix multiplication:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each term: (A_coeffs[4], B_coeffs[4], C_coeffs[4])
    # Coeffs indexed as [00, 01, 10, 11]
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a10+a11)b00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = a00(b10-b11)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # M4 = a11(b01-b00)
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5 = (a00+a01)b11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (a10-a00)(b00+b01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (a01-a11)(b10+b11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for s in range(7):
                    a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
                    
                    # A[bi,bk] occupies rows bi*2..bi*2+1, cols bk*2..bk*2+1 in A
                    # In U (shape 36 x rank), index = row * m + col = (bi*2+dr)*6 + (bk*2+dc)
                    for dr in range(2):
                        for dc in range(2):
                            idx = dr * 2 + dc  # 0,1,2,3 for [00,01,10,11]
                            u_idx = (bi * 2 + dr) * 6 + (bk * 2 + dc)
                            U[u_idx, col] = a_coeffs[idx]
                    
                    # B[bk,bj] occupies rows bk*2..bk*2+1, cols bj*2..bj*2+1 in B
                    # In V (shape 36 x rank), index = row * p + col = (bk*2+dr)*6 + (bj*2+dc)
                    for dr in range(2):
                        for dc in range(2):
                            idx = dr * 2 + dc
                            v_idx = (bk * 2 + dr) * 6 + (bj * 2 + dc)
                            V[v_idx, col] = b_coeffs[idx]
                    
                    # C[bi,bj] occupies rows bj*2..bj*2+1, cols bi*2..bi*2+1 in C
                    # In W (shape 36 x rank), index = row * n + col = (bj*2+dr)*6 + (bi*2+dc)
                    for dr in range(2):
                        for dc in range(2):
                            idx = dr * 2 + dc
                            w_idx = (bj * 2 + dr) * 6 + (bi * 2 + dc)
                            W[w_idx, col] = c_coeffs[idx]
                    
                    col += 1

    # Verify the decomposition
    if not _verify_decomposition(n, m, p, U, V, W):
        return None

    return U, V, W


def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that U, V, W form a valid decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    
    # Check all 216 tensor entries that should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check all entries that should be 0 (sample)
    np.random.seed(123)
    for _ in range(1000):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k1 = b % p
        k2 = c // n
        i2 = c % n
        
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    
    return True
# EVOLVE-BLOCK-END
