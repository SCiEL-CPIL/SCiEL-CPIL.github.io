"""Rewritten program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # We'll use a block Strassen approach.
    # A is 4x4, B is 4x8, C is 4x8.
    # Partition A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Partition B into 2x8 blocks (rows): B = [[B0], [B1]] where B0 is 2x8, B1 is 2x8
    # Partition C into 2x8 blocks (rows): C = [[C0], [C1]] where C0 is 2x8, C1 is 2x8
    #
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    #
    # For each 2x2 * 2x8 multiplication, we can use Strassen-like approach.
    # But 2x8 is not square. Let's think differently.
    #
    # Actually, let's use a different block structure.
    # A is 4x4, B is 4x8.
    # We can view this as:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each Bij is 2x2
    # C = [[C00, C01, C02, C03], [C10, C11, C12, C13]] where each Cij is 2x2
    #
    # Cij = sum_k Aik * Bkj for k in {0,1}
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C03 = A00*B03 + A01*B13
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    # C13 = A10*B03 + A11*B13
    #
    # Each 2x2 * 2x2 multiplication can be done with 7 multiplications (Strassen).
    # We have 8 such products (A00*B00, A00*B01, ..., A11*B13).
    # Total: 8 * 7 = 56 multiplications.
    
    # Let's implement this.
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)
    
    # Index conventions:
    # U: index i*m + j for A[i,j]
    # V: index j*p + k for B[j,k]
    # W: index k*n + i for C[i,k]
    
    # Strassen 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M4
    
    # For each block product Aij * Bkl (where i,j,k,l are block indices),
    # we add 7 rank-1 terms.
    
    # Block indices:
    # A blocks: (ai, aj) where ai, aj in {0, 1}
    #   A[ai, aj] covers A rows 2*ai..2*ai+1, cols 2*aj..2*aj+1
    # B blocks: (bi, bk) where bi, bk in {0, 1, 2, 3}
    #   B[bi, bk] covers B rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    # C blocks: (ci, ck) where ci, ck in {0, 1, 2, 3}
    #   C[ci, ck] covers C rows 2*ci..2*ci+1, cols 2*ck..2*ck+1
    
    # C[ci, ck] = sum_{bi} A[ci, bi] * B[bi, ck]
    # So for ci in {0,1}, ck in {0,1,2,3}, bi in {0,1}:
    #   C[ci, ck] += A[ci, bi] * B[bi, ck]
    
    # Total block products: 2 * 4 * 2 = 16? No, wait.
    # ci in {0,1}, ck in {0,1,2,3}, bi in {0,1}
    # That's 2 * 4 * 2 = 16 block products, each 2x2*2x2.
    # Each needs 7 mults, total 16*7 = 112. That's worse than 128? No, 112 < 128.
    
    # Wait, let me recount.
    # A is 4x4, so 2x2 blocks: 2 row blocks, 2 col blocks -> 4 blocks
    # B is 4x8, so 2x2 blocks: 2 row blocks, 4 col blocks -> 8 blocks
    # C is 4x8, so 2x2 blocks: 2 row blocks, 4 col blocks -> 8 blocks
    #
    # C[ci, ck] = sum_{bi=0}^{1} A[ci, bi] * B[bi, ck]
    # ci in {0,1}, ck in {0,1,2,3}, bi in {0,1}
    # Number of terms: 2 * 4 * 2 = 16
    # Each term is a 2x2 * 2x2 multiplication, done with 7 mults.
    # Total: 16 * 7 = 112.
    
    # Let's implement this.
    
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ci in range(2):  # C row block
        for ck in range(4):  # C col block
            for bi in range(2):  # B row block (sum index)
                # A block: (ci, bi) -> rows 2*ci..2*ci+1, cols 2*bi..2*bi+1
                # B block: (bi, ck) -> rows 2*bi..2*bi+1, cols 2*ck..2*ck+1
                # C block: (ci, ck) -> rows 2*ci..2*ci+1, cols 2*ck..2*ck+1
                
                # A indices
                a00 = (2*ci) * m + (2*bi)
                a01 = (2*ci) * m + (2*bi + 1)
                a10 = (2*ci + 1) * m + (2*bi)
                a11 = (2*ci + 1) * m + (2*bi + 1)
                
                # B indices
                b00 = (2*bi) * p + (2*ck)
                b01 = (2*bi) * p + (2*ck + 1)
                b10 = (2*bi + 1) * p + (2*ck)
                b11 = (2*bi + 1) * p + (2*ck + 1)
                
                # C indices
                c00 = (2*ck) * n + (2*ci)
                c01 = (2*ck + 1) * n + (2*ci)
                c10 = (2*ck) * n + (2*ci + 1)
                c11 = (2*ck + 1) * n + (2*ci + 1)
                
                # Strassen terms:
                # M1 = (a+d)(e+h)
                U[a00][col] = 1; V[b00][col] = 1; W[c00][col] = 1
                U[a00][col] += 0  # a
                U[a11][col] = 1; V[b00][col] += 0
                # Actually, let me be more careful.
                # M1 = (a00 + a11) * (b00 + b11)
                # This expands to: a00*b00 + a00*b11 + a11*b00 + a11*b11
                # But in tensor decomposition, each rank-1 term is a product of linear forms.
                # So M1 corresponds to: U has a00=1, a11=1; V has b00=1, b11=1; W has c00=1, c01=1, c10=1, c11=1 with appropriate signs.
                
                # Let me redo this properly.
                # For each Strassen multiplication, we have 7 rank-1 terms.
                # Each term is defined by:
                # - U vector: coefficients of A entries
                # - V vector: coefficients of B entries
                # - W vector: coefficients of C entries (with signs)
                
                # M1 = (a00 + a11) * (b00 + b11)
                # Contributes to: C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
                # C00 = M1 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M4
                
                # So M1 contributes to C00 with +1 and C11 with +1.
                
                # Term 1: M1
                U[a00][col] = 1
                U[a11][col] = 1
                V[b00][col] = 1
                V[b11][col] = 1
                W[c00][col] = 1
                W[c11][col] = 1
                col += 1
                
                # Term 2: M2 = (c+d)(e) = (a10 + a11) * b00
                U[a10][col] = 1
                U[a11][col] = 1
                V[b00][col] = 1
                W[c10][col] = 1
                W[c11][col] = -1
                col += 1
                
                # Term 3: M3 = a(f-h) = a00 * (b01 - b11)
                U[a00][col] = 1
                V[b01][col] = 1
                V[b11][col] = -1
                W[c01][col] = 1
                W[c11][col] = 1
                col += 1
                
                # Term 4: M4 = d(g-e) = a11 * (b10 - b00)
                U[a11][col] = 1
                V[b10][col] = 1
                V[b00][col] = -1
                W[c10][col] = 1
                W[c11][col] = 1
                col += 1
                
                # Term 5: M5 = (a+b)(h) = (a00 + a01) * b11
                U[a00][col] = 1
                U[a01][col] = 1
                V[b11][col] = 1
                W[c00][col] = -1
                W[c01][col] = 1
                col += 1
                
                # Term 6: M6 = (c-a)(e+f) = (a10 - a00) * (b00 + b01)
                U[a10][col] = 1
                U[a00][col] = -1
                V[b00][col] = 1
                V[b01][col] = 1
                W[c10][col] = 1
                col += 1
                
                # Term 7: M7 = (b-d)(g+h) = (a01 - a11) * (b10 + b11)
                U[a01][col] = 1
                U[a11][col] = -1
                V[b10][col] = 1
                V[b11][col] = 1
                W[c00][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
