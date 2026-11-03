"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p, num_trials=5, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for _ in range(num_trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                Wval = W_np[k * n + i]
                for j in range(m):
                    Uval = U_np[i * m + j]
                    Vval = V_np[j * p + k]
                    s += np.dot(np.multiply(Uval, Vval), Wval)
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

def _build_strassen_tiled(n, m, p):
    """
    Build a Strassen-tiled decomposition for 4x4 * 4x6.
    
    Split A (4x4) into 2x2 blocks of size 2x2:
      A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
      A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    
    Split B (4x6) into 2x3 blocks of size 2x2:
      B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B02 (rows 0-1, cols 4-5)
      B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3), B12 (rows 2-3, cols 4-5)
    
    C (4x6) blocks:
      C0j = A00*B0j + A01*B1j  for j=0,1,2
      C1j = A10*B0j + A11*B1j  for j=0,1,2
    
    For each column j, we have a 2x2 "matrix" product:
      [C0j]   [A00  A01]   [B0j]
      [C1j] = [A10  A11] * [B1j]
    
    Strassen for 2x2 matrix mult uses 7 products:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B11)
      M4 = A22(B22-B12)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    In our block notation (A11=A00, A12=A01, A21=A10, A22=A11; B11=B0j, B12=B1j... wait)
    
    Let me re-index. The "2x2 matrix" is:
      Row 0: [A00, A01]
      Row 1: [A10, A11]
    
    So in Strassen notation:
      A11 = A00, A12 = A01, A21 = A10, A22 = A11
    
    B matrix (2x1 in block sense, but each "entry" is 2x2):
      B11 = B0j, B21 = B1j
    
    Wait, Strassen is for 2x2 * 2x2. Here we have 2x2 * 2x1 (in block sense).
    Actually B is 4x6, split into 2 rows of 2 and 3 columns of 2. So for a fixed column j,
    B has 2 block-rows (each 2x2). So it's 2x2 * 2x1 in the block sense? No...
    
    Let me re-examine. A is 4x4, split into 2x2 blocks. B is 4x6, split into 2x3 blocks.
    C is 4x6, split into 2x3 blocks.
    
    C[i_block, j_block] = sum over middle block l: A[i_block, l] * B[l, j_block]
    
    For i_block=0: C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j] = A00*B0j + A01*B1j
    For i_block=1: C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j] = A10*B0j + A11*B1j
    
    So for fixed j, we're computing:
      [C0j]   [A00  A01]   [B0j]
      [C1j] = [A10  A11] * [B1j]
    
    This is a 2x2 matrix times a 2x1 matrix (in the block sense). But Strassen is for 2x2*2x2.
    
    Hmm, we can't directly apply Strassen here because B is not 2x2 in the block sense for a fixed column.
    
    Alternative: treat the whole thing differently. Let me think of it as:
    We need to compute C = A @ B where A is 4x4 and B is 4x6.
    
    Another approach: use the fact that the rank of the matrix multiplication tensor for 4x4*4x6
    can be bounded by considering it as related to 4x4*4x4 and 4x4*4x2.
    
    Actually, let me just try a different tiling. What if I use Strassen on the 4x4*4x4 part
    and schoolbook for the extra 2 columns?
    
    4x4*4x4: Strassen gives rank 7^2 = 49 (two levels of Strassen)
    4x4*4x2: schoolbook gives 4*4*2 = 32
    Total: 49 + 32 = 81
    
    That's better than 96! Let me construct this.
    
    For 4x4*4x4 with Strassen (rank 49):
    Level 1: split into 2x2 blocks of 2x2. 7 Strassen products, each 2x2*2x2.
    Level 2: each 2x2*2x2 uses 7 Strassen products of 1x1*1x1.
    Total: 7*7 = 49 rank-1 terms.
    
    For 4x4*4x2 with schoolbook (rank 32):
    4*4*2 = 32 rank-1 terms.
    
    Combined: 49 + 32 = 81.
    """
    n, m, p = 4, 4, 6
    rank_total = 81
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B), using Strassen rank 49
    # Strassen for 2x2: 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4, we apply Strassen twice.
    # Level 1: A is 2x2 blocks of 2x2, B is 2x2 blocks of 2x2.
    # The 7 Strassen "products" at level 1 are each 2x2*2x2 matrix multiplications.
    # Each of those is further decomposed with Strassen into 7 scalar products.
    
    # Let me define the Strassen factors for 2x2*2x2.
    # For matrices X (2x2) and Y (2x2):
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y21-Y11)
    # M4 = X22(Y22-Y12)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a product of two linear combinations.
    # For the rank-1 decomposition, each M contributes 1 rank-1 term if we can
    # express it as a product of three vectors. But M = (linear combo of X entries) * (linear combo of Y entries).
    # This is a rank-1 term in the tensor sense: U * V * W where U picks the X entries, V picks the Y entries,
    # and W contributes to the output.
    
    # Actually, for the full tensor decomposition, each scalar multiplication x_ij * y_jk contributes
    # a rank-1 term. Strassen reduces the number of such scalar multiplications.
    
    # Let me just build the schoolbook for 4x4*4x4 first to verify, then try Strassen.
    
    # Schoolbook for 4x4*4x4: rank 64
    # For i in 0..3, j in 0..3, k in 0..3:
    #   U[i*4+j, col] = 1
    #   V[j*4+k, col] = 1
    #   W[k*4+i, col] = 1
    
    # Let me try a simpler approach: just use schoolbook for the whole thing but with
    # a smarter search for lower rank.
    
    # Actually, let me just return schoolbook for now and add a search.
    # The Strassen construction is complex; let me verify the simpler approach first.
    
    # For now, return schoolbook
    return None  # Signal to use schoolbook

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    # Start with schoolbook as baseline
    baseline = _build_schoolbook(n, m, p)
    best = baseline
    best_rank = n * m * p
    
    # Try Strassen-tiled approach
    strassen = _build_strassen_tiled(n, m, p)
    if strassen is not None:
        if _verify_decomposition(strassen["U"], strassen["V"], strassen["W"], n, m, p):
            rank = len(strassen["U"][0])
            if rank < best_rank:
                best = strassen
                best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
