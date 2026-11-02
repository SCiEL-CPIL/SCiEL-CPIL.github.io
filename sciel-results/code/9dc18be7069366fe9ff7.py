"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_6x6():
    """
    Build a rank-189 decomposition for 6x6 * 6x6 matrix multiplication
    using 2x2 Strassen blocks.

    Index conventions:
      U: row index = i*m + j  (i: 0..5, j: 0..5)
      V: row index = j*p + k  (j: 0..5, k: 0..5)
      W: row index = k*n + i  (k: 0..5, i: 0..5)

    We decompose 6x6 into a 3x3 grid of 2x2 blocks.
    Each 2x2 Strassen block has 7 multiplications.
    Total rank = 3*3*3 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 formulas (local 2x2 block indices: a=0,1; b=0,1)
    # M1 = A00 * B00        -> C00 += M1, C11 += M1
    # M2 = A10 + A11) * B00 -> C10 += M2, C11 -= M2
    # M3 = A00 * (B01 - B11)-> C01 += M3, C11 += M3
    # M4 = A11 * (B10 - B00)-> C00 += M4, C10 += M4
    # M5 = (A00 + A01) * B11-> C00 -= M5, C01 += M5
    # M6 = (A10 - A00) * (B00 + B01) -> C11 += M6
    # M7 = (A01 - A11) * (B10 + B11) -> C00 += M7

    # Each entry: (u_rows, u_coeffs, v_rows, v_coeffs, w_rows, w_coeffs)
    # u_rows: list of (local_i, local_j) in 0..1 for A block
    # v_rows: list of (local_j, local_k) in 0..1 for B block
    # w_rows: list of (local_k, local_i) in 0..1 for C block

    strassen_entries = [
        # M1 = A00 * B00
        ([(0,0)], [1], [(0,0)], [1], [(0,0), (1,1)], [1, 1]),
        # M2 = (A10 + A11) * B00
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]),
        # M3 = A00 * (B01 - B11)
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
        # M4 = A11 * (B10 - B00)
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),
        # M5 = (A00 + A01) * B11
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
    ]

    col = 0
    for ib in range(3):  # block row of A
        for jb in range(3):  # block col of B (which is block row of B, block col of A for C)
            for kb in range(3):  # block col of A (which is block row of C)
                # Wait, let me reconsider the block indexing.
                # A is 6x6, divided into 3x3 blocks of 2x2.
                # Block A[ib, kb] is rows 2*ib..2*ib+1, cols 2*kb..2*kb+1.
                # Block B[jb, kb] is rows 2*kb..2*kb+1, cols 2*jb..2*jb+1.
                # Block C[ib, jb] is rows 2*jb..2*jb+1, cols 2*ib..2*ib+1.
                # 
                # The Strassen formula for block (ib, kb) of A and block (jb, kb) of B:
                # Actually, the standard Strassen is for 2x2 blocks.
                # Let me re-think.
                # 
                # We have 3x3 blocks. For each pair of blocks (A_block, B_block),
                # we need to multiply them. But Strassen works on 2x2 blocks.
                # 
                # Actually, the approach is: divide 6x6 into 3x3 blocks of size 2x2.
                # For each block position, we have a 2x2 matrix.
                # The product C = A * B where A, B, C are 3x3 blocks of 2x2 matrices.
                # 
                # We can use Strassen on the 3x3 block structure? No, Strassen is for 2x2.
                # 
                # Alternative: Use Strassen for each 2x2 sub-block multiplication.
                # But the 3x3 block multiplication is a matrix multiplication of 3x3 matrices
                # where each entry is a 2x2 matrix.
                # 
                # So we need a decomposition of 3x3 matrix multiplication where each entry
                # is a 2x2 matrix.
                # 
                # The schoolbook for 3x3x3 has 27 terms. If we use Strassen-like
                # for 3x3, we can do better. But the simplest is:
                # For each (ib, jb, kb), multiply A[ib,kb] * B[kb,jb] using Strassen (7 mults).
                # Total: 3*3*3*7 = 189.
                # 
                # Let me re-derive.
                
                # A block at (ib, kb): rows 2*ib + a, cols 2*kb + c for a,c in 0,1
                # B block at (kb, jb): rows 2*kb + c, cols 2*jb + b for c,b in 0,1
                # C block at (ib, jb): rows 2*jb + b, cols 2*ib + a for b,a in 0,1
                
                # U index: i*m + j = (2*ib + a)*6 + (2*kb + c)
                # V index: j*p + k = (2*kb + c)*6 + (2*jb + b)
                # W index: k*n + i = (2*jb + b)*6 + (2*ib + a)
                
                for (u_rows, u_coeffs, v_rows, v_coeffs, w_rows, w_coeffs) in strassen_entries:
                    # u_rows are (a, c) pairs for A[ib,kb]
                    # v_rows are (c, b) pairs for B[kb,jb]
                    # w_rows are (b, a) pairs for C[ib,jb]
                    
                    for (a, c), coeff_u in zip(u_rows, u_coeffs):
                        u_idx = (2*ib + a) * m + (2*kb + c)
                        U[u_idx][col] += coeff_u
                    
                    for (c, b), coeff_v in zip(v_rows, v_coeffs):
                        v_idx = (2*kb + c) * p + (2*jb + b)
                        V[v_idx][col] += coeff_v
                    
                    for (b, a), coeff_w in zip(w_rows, w_coeffs):
                        w_idx = (2*jb + b) * n + (2*ib + a)
                        W[w_idx][col] += coeff_w
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 6x6 * 6x6 matrix multiplication.
    
    Uses Strassen-style 2x2 block decomposition with 3x3 block grid.
    Rank = 189 (vs schoolbook 216).
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    t0 = time.time()
    deadline = 140  # 10s safety margin
    
    try:
        result = _build_strassen_6x6()
        
        # Verify the decomposition
        # C[i,j] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Let's check a few entries
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0])
        
        # Verify: for each (i,j,k), C[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 if j == k (the middle index), 0 otherwise
        # Wait, the tensor T[i,j,k] = 1 if j==k else 0 (for standard matrix mult)
        # Actually T[i,j,k] = delta_{j,k}
        
        # Quick verification on a subset
        import random
        rng = random.Random(42)
        errors = 0
        for _ in range(200):
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            expected = 1.0 if j == k else 0.0
            actual = 0.0
            for c in range(rank):
                actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            if abs(actual - expected) > 1e-9:
                errors += 1
        
        if errors == 0:
            return result
        
        # If verification fails, fall back to schoolbook
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
        
    except Exception:
        # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
