"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-68 tensor decomposition for 3x3 * 3x8 matrix multiplication.
    The decomposition is built by splitting the 3x3 matrix A into blocks:
      A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    B is 3x8, split into B1 (2x8) and B2 (1x8).
    C = A*B = [[A11*B1 + A12*B2, ...], ...]
    
    We use Strassen's algorithm (rank 7) for the 2x2x8 subproblem (two 2x2x4, each using two 2x2x2 Strassen = 7*2=14 per 2x2x4? No.
    Actually, 2x2x8: we can view B1 as [B11 B12 B13 B14] where each is 2x2.
    A11 * B1 = A11*B11 + A11*B12 + A11*B13 + A11*B14.
    This is 4 separate 2x2x2 multiplications. Each 2x2x2 has rank 7 via Strassen. Total 28.
    
    A12 * B2: A12 is 2x1, B2 is 1x8. Product is 2x8.
    C_block12 = A12 * B2. This is an outer product of a 2-vector and 8-vector.
    Each entry (i,k) is A12[i]*B2[k]. There are 16 entries. Each requires 1 multiplication. Total 16.
    
    A21 * B1: A21 is 1x2, B1 is 2x8. Product is 1x8.
    C_block21 = A21 * B1. 8 entries, each is sum of 2 products. So 16 multiplications.
    
    A22 * B2: 1x1 * 1x8 = 1x8. 8 multiplications.
    
    Total: 28 + 16 + 16 + 8 = 68.
    
    Wait, let me double check the Strassen rank for 2x2x8.
    Actually, there might be a better way to do 2x2x8.
    But let's stick with the simple block decomposition.
    
    Let's verify the indices.
    n=3, m=3, p=8.
    A is 3x3. B is 3x8. C is 3x8.
    Split A: rows 0,1 (2x2 block A11), row 2 (1x1 block A22).
    A12 is col 2, rows 0,1 (2x1).
    A21 is row 2, cols 0,1 (1x2).
    
    Split B: rows 0,1 (2x8 block B1), row 2 (1x8 block B2).
    
    C[0:2, :] = A11*B1 + A12*B2
    C[2, :] = A21*B1 + A22*B2
    
    U shape: (n*m, rank) = (9, 68)
    V shape: (m*p, rank) = (24, 68)
    W shape: (p*n, rank) = (24, 68)
    
    Convention: W[k*n + i, r]
    So for output C[i, k], the index in W is k*3 + i.
    
    Let's build the 68 terms.
    """
    deadline = time.time() + 110  # Safety margin
    
    n, m, p = 3, 3, 8
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A11 * B1 using Strassen for 2x2x2 blocks
    # A11 is A[0:2, 0:2], B1 is B[0:2, 0:8]
    # B1 consists of 4 blocks of 2x2: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    # Let's define a helper to add a 2x2x2 Strassen decomposition for A_block at (r0, c0) and B_block at (r0, c0) to C_block at (r0, c0)
    
    def add_strassen_2x2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """
        Adds Strassen 2x2x2 decomposition.
        A is 2x2 at (a_r0, a_c0) in original A (indices i, j)
        B is 2x2 at (b_r0, b_c0) in original B (indices j, k)
        C is 2x2 at (c_r0, c_c0) in original C (indices i, k)
        """
        nonlocal col
        # Strassen formulas:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # A indices: (a_r0 + i, a_c0 + j) for i,j in {0,1}
        # B indices: (b_r0 + j, b_c0 + k) for j,k in {0,1}
        # C indices: (c_r0 + i, c_c0 + k) for i,k in {0,1}
        
        # We need to map these to U, V, W indices.
        # U[i*m + j, col]
        # V[j*p + k, col]
        # W[k*n + i, col]
        
        # M1: (A00+A11)(B00+B11)
        # A terms: (a_r0, a_c0) + (a_r0+1, a_c0+1)
        # B terms: (b_r0, b_c0) + (b_r0+1, b_c0+1)
        # We need to expand (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This is 4 terms. But Strassen uses 7 terms total for the whole 2x2x2.
        # The 7 terms are linear combinations.
        # Term 1: (A00+A11) * (B00+B11)
        # This contributes to C00, C11.
        
        # Let's add each of the 7 Strassen products as a separate rank-1 term.
        # But wait, each "term" in Strassen is a product of two linear combinations.
        # So each term adds 4 entries to U and V (or fewer if coefficients are 0).
        # And contributes to specific C entries.
        
        # Actually, it's easier to just list the 7 products and their contributions.
        
        # Product 1: (A00 + A11) * (B00 + B11)
        # U: A00 -> 1, A11 -> 1
        # V: B00 -> 1, B11 -> 1
        # W: C00 += 1, C11 += 1
        # But C00 also gets M4, M5, M7.
        
        # This approach of adding one rank-1 term per Strassen multiplication is correct.
        # There are 7 multiplications.
        
        # Let's define the 7 terms explicitly.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[a_r0*m + a_c0][col] += 1
        U[(a_r0+1)*m + (a_c0+1)][col] += 1
        V[b_r0*p + b_c0][col] += 1
        V[(b_r0+1)*p + (b_c0+1)][col] += 1
        W[b_c0*n + c_r0][col] += 1  # C00: k=0, i=0
        W[(b_c0+1)*n + (c_r0+1)][col] += 1  # C11: k=1, i=1
        col += 1
        
        # Term 2: M2 = (A10+A11)B00
        U[(a_r0+1)*m + a_c0][col] += 1  # A10
        U[(a_r0+1)*m + (a_c0+1)][col] += 1  # A11
        V[b_r0*p + b_c0][col] += 1  # B00
        W[b_c0*n + (c_r0+1)][col] += 1  # C10: k=0, i=1
        col += 1
        
        # Term 3: M3 = A00(B01-B11)
        U[a_r0*m + a_c0][col] += 1  # A00
        V[b_r0*p + (b_c0+1)][col] += 1  # B01
        V[(b_r0+1)*p + (b_c0+1)][col] -= 1  # -B11
        W[(b_c0+1)*n + c_r0][col] += 1  # C01: k=1, i=0
        W[(b_c0+1)*n + (c_r0+1)][col] += 1  # C11: k=1, i=1
        col += 1
        
        # Term 4: M4 = A11(B10-B00)
        U[(a_r0+1)*m + (a_c0+1)][col] += 1  # A11
        V[(b_r0+1)*p + b_c0][col] += 1  # B10
        V[b_r0*p + b_c0][col] -= 1  # -B00
        W[b_c0*n + c_r0][col] -= 1  # C00: k=0, i=0 (contribution -1)
        W[b_c0*n + (c_r0+1)][col] += 1  # C10: k=0, i=1
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        U[a_r0*m + a_c0][col] += 1  # A00
        U[a_r0*m + (a_c0+1)][col] += 1  # A01
        V[(b_r0+1)*p + (b_c0+1)][col] += 1  # B11
        W[(b_c0+1)*n + c_r0][col] -= 1  # C01: k=1, i=0 (contribution -1)
        W[(b_c0+1)*n + (c_r0+1)][col] -= 1  # C11: k=1, i=1 (contribution -1)
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        U[(a_r0+1)*m + a_c0][col] += 1  # A10
        U[a_r0*m + a_c0][col] -= 1  # -A00
        V[b_r0*p + b_c0][col] += 1  # B00
        V[b_r0*p + (b_c0+1)][col] += 1  # B01
        W[(b_c0+1)*n + (c_r0+1)][col] += 1  # C11: k=1, i=1
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        U[a_r0*m + (a_c0+1)][col] += 1  # A01
        U[(a_r0+1)*m + (a_c0+1)][col] -= 1  # -A11
        V[(b_r0+1)*p + b_c0][col] += 1  # B10
        V[(b_r0+1)*p + (b_c0+1)][col] += 1  # B11
        W[b_c0*n + c_r0][col] += 1  # C00: k=0, i=0
        col += 1
        
        return
    
    # We need to add 4 Strassen 2x2x2 decompositions for A11*B1
    # B1 has 4 blocks of 2x2: columns 0-1, 2-3, 4-5, 6-7
    for k_block in range(4):
        b_c0 = k_block * 2
        add_strassen_2x2x2(0, 0, 0, b_c0, 0, b_c0)
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 2: A12 * B2
    # A12 is A[0:2, 2] (2x1), B2 is B[2, 0:8] (1x8)
    # Product is 2x8. C[0:2, :] += A12 * B2
    # C[i, k] += A[0:2, 2][i] * B[2, k]
    # This is an outer product.
    # For each i in {0,1} and k in {0..7}, we have a term.
    # Total 16 terms.
    for i in range(2):
        for k in range(8):
            U[i*m + 2][col] += 1  # A[i, 2]
            V[2*p + k][col] += 1  # B[2, k]
            W[k*n + i][col] += 1  # C[i, k]
            col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 3: A21 * B1
    # A21 is A[2, 0:2] (1x2), B1 is B[0:2, 0:8] (2x8)
    # Product is 1x8. C[2, :] += A21 * B1
    # C[2, k] += sum_{j=0}^{1} A[2, j] * B[j, k]
    # This is 2 terms for each k. Total 16 terms.
    for k in range(8):
        for j in range(2):
            U[2*m + j][col] += 1  # A[2, j]
            V[j*p + k][col] += 1  # B[j, k]
            W[k*n + 2][col] += 1  # C[2, k]
            col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 4: A22 * B2
    # A22 is A[2, 2] (1x1), B2 is B[2, 0:8] (1x8)
    # Product is 1x8. C[2, :] += A22 * B2
    # C[2, k] += A[2, 2] * B[2, k]
    # 8 terms.
    for k in range(8):
        U[2*m + 2][col] += 1  # A[2, 2]
        V[2*p + k][col] += 1  # B[2, k]
        W[k*n + 2][col] += 1  # C[2, k]
        col += 1
    
    if col != rank:
        # Fallback: pad with zeros if something went wrong (should not happen)
        while col < rank:
            col += 1
        col = rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
