"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-65 decomposition for 4x4 * 4x5.
    
    Decomposes C = A*B into:
    - C[:, 0:4] = A[:,:4] * B[:,:4]  (4x4 * 4x4, rank 49 via Strassen-like)
    - C[:, 4:5] = A[:,:4] * B[:,4:5] (4x4 * 4x1, rank 16 via schoolbook)
    
    Total rank = 65.
    
    For the 4x4*4x4 part, we use a Strassen-like decomposition.
    For the 4x4*4x1 part, we use schoolbook.
    
    The output W follows the convention W[k*n+i, r] where n=4, so W has shape (5*4, rank) = (20, rank).
    Wait, p=5, n=4, so W shape is (p*n, rank) = (20, rank).
    Row index k*n + i where k in 0..p-1=0..4, i in 0..n-1=0..3.
    This corresponds to C[i, k].
    """
    n, m, p = 4, 4, 5  # A is n x m, B is m x p, C is n x p
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U row index: i*m + j for A[i,j]
    # V row index: j*p + k for B[j,k]
    # W row index: k*n + i for C[i,k]
    
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Part 1: C[:, 0:4] = A[:,:4] * B[:,:4] using Strassen for 4x4*4x4
    # We use a Strassen-like decomposition for 4x4 matrices.
    # The standard Strassen for 2x2 has rank 7.
    # For 4x4, we can view as 2x2 block matrices of 2x2 blocks.
    # Strassen on the 2x2 block structure gives 7 "block multiplications" of 2x2 matrices.
    # Each 2x2*2x2 can be done with schoolbook (rank 4) or Strassen (rank 7).
    # If we use schoolbook for the inner 2x2s: 7 * 4 = 28 rank for the block structure.
    # But that's not right. Let me think more carefully.
    
    # Actually, for a 4x4 * 4x4 multiplication, we can use a direct rank-49 decomposition
    # that's essentially schoolbook but we need to get it to 49.
    # Schoolbook for 4x4*4x4 is rank 64.
    
    # Let me use a known rank-49 algorithm for 4x4 matrix multiplication.
    # One approach: use Strassen's algorithm which gives O(n^2.807) but for fixed size...
    
    # Actually, let me reconsider. The minimal rank for 4x4*4x4 is not known to be 49.
    # Schoolbook is 64. Strassen gives 7^log2(4) = 7^2 = 49 if we can apply it correctly.
    
    # Strassen's algorithm for 4x4:
    # Split A, B into 2x2 blocks of 2x2 submatrices.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    
    # Strassen computes:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Then:
    # C11 = M1 + M4 - M2 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # Each Mj is a 2x2 matrix product. We need to decompose each 2x2*2x2 into rank terms.
    # For 2x2*2x2, schoolbook is rank 4. Strassen is rank 7.
    # Using schoolbook for the inner products: 7 * 4 = 28 rank terms.
    # But wait, the Mj are 2x2 products, and we need to express them as linear combinations.
    
    # Let's use schoolbook for the inner 2x2 products. Each 2x2*2x2 has 4 rank-1 terms.
    # So total 7 * 4 = 28 rank terms for the 4x4*4x4 part.
    # Plus 16 for the 4x4*4x1 part = 44 total. But the task says rank 65 in the previous attempt.
    
    # Hmm, let me just implement a clean Strassen-based decomposition.
    
    # For simplicity and correctness, let me use a known rank-49 decomposition.
    # I'll construct it by applying Strassen recursively.
    
    # Define helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff):
        """Add coeff to U[u_idx], V[v_idx], W[w_idx] for current column."""
        pass  # We'll handle this differently
    
    # Let me build the decomposition explicitly.
    # I'll use column index to track rank terms.
    
    col = 0
    
    # For the 4x4*4x4 part (columns 0-3 of C), use Strassen with schoolbook inner products.
    # A[i,j] for i,j in 0..3. B[j,k] for j,k in 0..3. C[i,k] for i,k in 0..3.
    
    # Strassen blocks:
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Each Mj is a 2x2 matrix. We decompose each 2x2*2x2 product into 4 rank-1 terms (schoolbook).
    
    # For a 2x2 matrix product X*Y where X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]:
    # X*Y = [[x00*y00, x00*y01], [x10*y00, x10*y01]] + [[x01*y10, x01*y11], [x11*y10, x11*y11]]
    # Wait, that's not right. Schoolbook:
    # (X*Y)[i,j] = sum_k X[i,k] * Y[k,j]
    # Rank-1 terms: for each k, X[:,k] * Y[k,:]
    
    # Let me define a function to add schoolbook terms for a 2x2*2x2 product.
    # The 2x2 X corresponds to some linear combination of A entries.
    # The 2x2 Y corresponds to some linear combination of B entries.
    # The result contributes to some linear combination of C entries.
    
    # This is getting complex. Let me just enumerate all terms explicitly.
    
    # I'll create the 7 M matrices as linear forms, then decompose each into rank-1 terms.
    
    # Let me use a different approach: directly build the rank-49 decomposition.
    
    # For 4x4*4x4, there exists a rank-49 algorithm. Let me construct it.
    
    # Actually, let me just use the straightforward approach:
    # For the 4x4*4x4 part, use 49 rank-1 terms from a known decomposition.
    # For the 4x4*4x1 part, use 16 rank-1 terms from schoolbook.
    
    # I'll construct the 49-term decomposition for 4x4*4x4 using Strassen.
    
    # Helper: add a term where U[row_a] gets coeff_a, V[row_b] gets coeff_b, W[row_c] gets coeff_c
    # row_a = i*m + j for A[i,j]
    # row_b = j*p + k for B[j,k]  
    # row_c = k*n + i for C[i,k]
    
    # Let me define the 7 Strassen products and decompose each into schoolbook 2x2 terms.
    
    def a_idx(i, j):
        return i * m + j
    
    def b_idx(j, k):
        return j * p + k
    
    def c_idx(k, i):
        return k * n + i
    
    # M1 = (A11 + A22) * (B11 + B22)
    # A11+A22 has entries:
    # (A11+A22)[0,0] = A[0,0] + A[2,2]
    # (A11+A22)[0,1] = A[0,1] + A[2,3]
    # (A11+A22)[1,0] = A[1,0] + A[3,2]
    # (A11+A22)[1,1] = A[1,1] + A[3,3]
    # B11+B22 has entries:
    # (B11+B22)[0,0] = B[0,0] + B[2,2]
    # (B11+B22)[0,1] = B[0,1] + B[2,3]
    # (B11+B22)[1,0] = B[1,0] + B[3,2]
    # (B11+B22)[1,1] = B[1,1] + B[3,3]
    
    # Schoolbook for 2x2*2x2: 4 terms, one for each "inner" index k=0,1
    # Term k=0: X[:,0] * Y[0,:]
    # Term k=1: X[:,1] * Y[1,:]
    
    # For M1, k=0 term:
    # U part: A[0,0]+A[2,2] at col 0, A[1,0]+A[3,2] at col 1 (for X[:,0])
    # V part: B[0,0]+B[2,2] at col 0, B[0,1]+B[2,3] at col 1 (for Y[0,:])
    # W part: C[0,0], C[0,1] from row 0 of result; C[1,0], C[1,1] from row 1
    # Result[0,0] = X[0,0]*Y[0,0] = (A[0,0]+A[2,2])*(B[0,0]+B[2,2]) -> contributes to C[0,0]
    # Result[0,1] = X[0,0]*Y[0,1] = (A[0,0]+A[2,2])*(B[0,1]+B[2,3]) -> contributes to C[0,1]
    # Result[1,0] = X[1,0]*Y[0,0] = (A[1,0]+A[3,2])*(B[0,0]+B[2,2]) -> contributes to C[1,0]
    # Result[1,1] = X[1,0]*Y[0,1] = (A[1,0]+A[3,2])*(B[0,1]+B[2,3]) -> contributes to C[1,1]
    
    # But these contribute to C11 (top-left 2x2 block of C).
    # C11 = M1 + M4 - M2 + M6
    # So M1 contributes +1 to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # This is getting very detailed. Let me just write out all 49 terms.
    
    # Given the complexity, let me use a simpler but still improved approach.
    # I'll use rank 49 for the 4x4*4x4 part with a specific known decomposition.
    
    # For now, let me just output a valid decomposition with rank 65 using a mix.
    # Actually, let me just do schoolbook for everything: rank 80.
    # But that's the same as the seed. I need something better.
    
    # Let me implement Strassen properly for 4x4*4x4 to get rank 49.
    
    # I'll define the 7 M products and their contributions.
    
    # M1 = (A11+A22)(B11+B22) -> contributes +1 to C11
    # M2 = (A21+A22)B11 -> contributes -1 to C11, +1 to C21
    # M3 = A11(B12-B22) -> contributes +1 to C12, -1 to C22
    # M4 = A22(B21-B11) -> contributes +1 to C11, +1 to C21
    # M5 = (A11+A12)B22 -> contributes +1 to C12, +1 to C22
    # M6 = (A21-A11)(B11+B12) -> contributes +1 to C11
    # M7 = (A12-A22)(B21+B22) -> contributes +1 to C22
    
    # Each Mj is a 2x2 product, decomposed into 4 schoolbook terms.
    # Total: 7 * 4 = 28 terms for the 4x4*4x4 part.
    # Plus 16 for 4x4*4x1 = 44 total.
    
    # But wait, is 28 correct? Each 2x2*2x2 schoolbook has 4 rank-1 terms.
    # 7 * 4 = 28. Yes.
    
    # So total rank = 28 + 16 = 44.
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 7 Strassen products
    # Each product is (X, Y) where X and Y are 2x2 linear forms.
    # X[i,j] = sum of coeff * A[ai,aj]
    # Y[i,j] = sum of coeff * B[bi,bj]
    # Result R = X*Y is 2x2, R[i,j] = sum_k X[i,k]*Y[k,j]
    # R contributes to C blocks with given signs.
    
    # I'll represent each 2x2 matrix as a dict of (i,j) -> list of (index, coeff)
    # where index is the flat index in A or B.
    
    def make_2x2_from_A(coeffs):
        """coeffs is a 4x4 list of lists, entries are (a_idx, coeff) or just coeff for single A entry."""
        pass
    
    # Let me just hardcode the 28 terms for the 4x4*4x4 part.
    
    # For each of the 7 M products, and for each of the 4 schoolbook terms (k=0,1 for inner index),
    # I add one rank-1 term.
    
    # M1 = (A11+A22)(B11+B22), contributes +1 to C11
    # X = A11+A22:
    #   X[0,0] = A[0,0] + A[2,2]
    #   X[0,1] = A[0,1] + A[2,3]
    #   X[1,0] = A[1,0] + A[3,2]
    #   X[1,1] = A[1,1] + A[3,3]
    # Y = B11+B22:
    #   Y[0,0] = B[0,0] + B[2,2]
    #   Y[0,1] = B[0,1] + B[2,3]
    #   Y[1,0] = B[1,0] + B[3,2]
    #   Y[1,1] = B[1,1] + B[3,3]
    
    # Schoolbook term k=0: X[:,0] * Y[0,:]
    # U: A[0,0]+A[2,2] at row a_idx(0,0), A[1,0]+A[3,2] at row a_idx(1,0)
    # V: B[0,0]+B[2,2] at row b_idx(0,0), B[0,1]+B[2,3] at row b_idx(0,1)
    # W: C[0,0] at row c_idx(0,0), C[0,1] at row c_idx(1,0), C[1,0] at row c_idx(0,1), C[1,1] at row c_idx(1,1)
    # All with coeff +1
    
    U[a_idx(0,0)][col] += 1.0
    U[a_idx(2,2)][col] += 1.0
    U[a_idx(1,0)][col] += 1.0
    U[a_idx(3,2)][col] += 1.0
    V[b_idx(0,0)][col] += 1.0
    V[b_idx(2,2)][col] += 1.0
    V[b_idx(0,1)][col] += 1.0
    V[b_idx(2,3)][col] += 1.0
    W[c_idx(0,0)][col] += 1.0
    W[c_idx(1,0)][col] += 1.0
    W[c_idx(0,1)][col] += 1.0
    W[c_idx(1,1)][col] += 1.0
    col += 1
    
    # Schoolbook term k=1: X[:,1] * Y[1,:]
    # U: A[0,1]+A[2,3] at row a_idx(0,1), A[1,1]+A[3,3] at row a_idx(1,1)
    # V: B[1,0]+B[3,2] at row b_idx(1,0), B[1,1]+B[3,3] at row b_idx(1,1)
    # W: same as above
    U[a_idx(0,1)][col] += 1.0
    U[a_idx(2,3)][col] += 1.0
    U[a_idx(1,1)][col] += 1.0
    U[a_idx(3,3)][col] += 1.0
    V[b_idx(1,0)][col] += 1.0
    V[b_idx(3,2)][col] += 1.0
    V[b_idx(1,1)][col] += 1.0
    V[b_idx(3,3)][col] += 1.0
    W[c_idx(0,0)][col] += 1.0
    W[c_idx(1,0)][col] += 1.0
    W[c_idx(0,1)][col] += 1.0
    W[c_idx(1,1)][col] += 1.0
    col += 1
    
    # M2 = (A21+A22)B11, contributes -1 to C11, +1 to C21
    # X = A21+A22:
    #   X[0,0] = A[2,0] + A[2,2]
    #   X[0,1] = A[2,1] + A[2,3]
    #   X[1,0] = A[3,0] + A[3,2]
    #   X[1,1] = A[3,1] + A[3,3]
    # Y = B11:
    #   Y[0,0] = B[0,0]
    #   Y[0,1] = B[0,1]
    #   Y[1,0] = B[1,0]
    #   Y[1,1] = B[1,1]
    
    # Term k=0: X[:,0] * Y[0,:]
    U[a_idx(2,0)][col] += 1.0
    U[a_idx(2,2)][col] += 1.0
    U[a_idx(3,0)][col] += 1.0
    U[a_idx(3,2)][col] += 1.0
    V[b_idx(0,0)][col] += 1.0
    V[b_idx(0,1)][col] += 1.0
    # C21: rows 2,3 of C, cols 0,1. C[2,0], C[2,1], C[3,0], C[3,1]
    # But M2 contributes to C11 with -1 and C21 with +1.
    # The result of X*Y is a 2x2 matrix that adds to C21 (bottom-left block).
    W[c_idx(0,2)][col] += 1.0
    W[c_idx(1,2)][col] += 1.0
    W[c_idx(0,3)][col] += 1.0
    W[c_idx(1,3)][col] += 1.0
    # And -1 to C11
    W[c_idx(0,0)][col] += -1.0
    W[c_idx(1,0)][col] += -1.0
    W[c_idx(0,1)][col] += -1.0
    W[c_idx(1,1)][col] += -1.0
    col += 1
    
    # Term k=1: X[:,1] * Y[1,:]
    U[a_idx(2,1)][col] += 1.0
    U[a_idx(2,3)][col] += 1.0
    U[a_idx(3,1)][col] += 1.0
    U[a_idx(3,3)][col] += 1.0
    V[b_idx(1,0)][col] += 1.0
    V[b_idx(1,1)][col] += 1.0
    W[c_idx(0,2)][col] += 1.0
    W[c_idx(1,2)][col] += 1.0
    W[c_idx(0,3)][col] += 1.0
    W[c_idx(1,3)][col] += 1.0
    W[c_idx(0,0)][col] += -1.0
    W[c_idx(1,0)][col] += -1.0
    W[c_idx(0,1)][col] += -1.0
    W[c_idx(1,1)][col] += -1.0
    col += 1
    
    # M3 = A11(B12-B22), contributes +1 to C12, -1 to C22
    # X = A11:
    #   X[0,0] = A[0,0], X[0,1] = A[0,1]
    #   X[1,0] = A[1,0], X[1,1] = A[1,1]
    # Y = B12-B22:
    #   Y[0,0] = B[0,2] - B[2,2]
    #   Y[0,1] = B[0,3] - B[2,3]
    #   Y[1,0] = B[1,2] - B[3,2]
    #   Y[1,1] = B[1,3] - B[3,3]
    
    # Term k=0: X[:,0] * Y[0,:]
    U[a_idx(0,0)][col] += 1.0
    U[a_idx(1,0)][col] += 1.0
    V[b_idx(0,2)][col] += 1.0
    V[b_idx(2,2)][col] += -1.0
    V[b_idx(0,3)][col] += 1.0
    V[b_idx(2,3)][col] += -1.0
    # C12: rows 0,1 of C, cols 2,3
    W[c_idx(2,0)][col] += 1.0
    W[c_idx(3,0)][col] += 1.0
    W[c_idx(2,1)][col] += 1.0
    W[c_idx(3,1)][col] += 1.0
    # C22: rows 2,3 of C, cols 2,3, with -1
    W[c_idx(2,2)][col] += -1.0
    W[c_idx(3,2)][col] += -1.0
    W[c_idx(2,3)][col] += -1.0
    W[c_idx(3,3)][col] += -1.0
    col += 1
    
    # Term k=1: X[:,1] * Y[1,:]
    U[a_idx(0,1)][col] += 1.0
    U[a_idx(1,1)][col] += 1.0
    V[b_idx(1,2)][col] += 1.0
    V[b_idx(3,2)][col] += -1.0
    V[b_idx(1,3)][col] += 1.0
    V[b_idx(3,3)][col] += -1.0
    W[c_idx(2,0)][col] += 1.0
    W[c_idx(3,0)][col] += 1.0
    W[c_idx(2,1)][col] += 1.0
    W[c_idx(3,1)][col] += 1.0
    W[c_idx(2,2)][col] += -1.0
    W[c_idx(3,2)][col] += -1.0
    W[c_idx(2,3)][col] += -1.0
    W[c_idx(3,3)][col] += -1.0
    col += 1
    
    # M4 = A22(B21-B11), contributes +1 to C11, +1 to C21
    # X = A22:
    #   X[0,0] = A[2,2], X[0,1] = A[2,3]
    #   X[1,0] = A[3,2], X[1,1] = A[3,3]
    # Y = B21-B11:
    #   Y[0,0] = B[2,0] - B[0,0]
    #   Y[0,1] = B[2,1] - B[0,1]
    #   Y[1,0] = B[3,0] - B[1,0]
    #   Y[1,1] = B[3,1] - B[1,1]
    
    # Term k=0: X[:,0] * Y[0,:]
    U[a_idx(2,2)][col] += 1.0
    U[a_idx(3,2)][col] += 1.0
    V[b_idx(2,0)][col] += 1.0
    V[b_idx(0,0)][col] += -1.0
    V[b_idx(2,1)][col] += 1.0
    V[b_idx(0,1)][col] += -1.0
    W[c_idx(0,0)][col] += 1.0
    W[c_idx(1,0)][col] += 1.0
    W[c_idx(0,1)][col] += 1.0
    W[c_idx(1,1)][col] += 1.0
    W[c_idx(0,2)][col] += 1.0
    W[c_idx(1,2)][col] += 1.0
    W[c_idx(0,3)][col] += 1.0
    W[c_idx(1,3)][col] += 1.0
    col += 1
    
    # Term k=1: X[:,1] * Y[1,:]
    U[a_idx(2,3)][col] += 1.0
    U[a_idx(3,3)][col] += 1.0
    V[b_idx(3,0)][col] += 1.0
    V[b_idx(1,0)][col] += -1.0
    V[b_idx(3,1)][col] += 1.0
    V[b_idx(1,1)][col] += -1.0
    W[c_idx(0,0)][col] += 1.0
    W[c_idx(1,0)][col] += 1.0
    W[c_idx(0,1)][col] += 1.0
    W[c_idx(1,1)][col] += 1.0
    W[c_idx(0,2)][col] += 1.0
    W[c_idx(1,2)][col] += 1.0
    W[c_idx(0,3)][col] += 1.0
    W[c_idx(1,3)][col] += 1.0
    col += 1
    
    # M5 = (A11+A12)B22, contributes +1 to C12, +1 to C22
    # X = A11+A12:
    #   X[0,0] = A[0,0] + A[0,2]
    #   X[0,1] = A[0,1] + A[0,3]
    #   X[1,0] = A[1,0] + A[1,2]
    #   X[1,1] = A[1,1] + A[1,3]
    # Y = B22:
    #   Y[0,0] = B[2,2], Y[0,1] = B[2,3]
    #   Y[1,0] = B[3,2], Y[1,1] = B[3,3]
    
    # Term k=0: X[:,0] * Y[0,:]
    U[a_idx(0,0)][col] += 1.0
    U[a_idx(0,2)][col] += 1.0
    U[a_idx(1,0)][col] += 1.0
    U[a_idx(1,2)][col] += 1.0
    V[b_idx(2,2)][col] += 1.0
    V[b_idx(2,3)][col] += 1.0
    W[c_idx(2,0)][col] += 1.0
    W[c_idx(3,0)][col] += 1.0
    W[c_idx(2,1)][col] += 1.0
    W[c_idx(3,1)][col] += 1.0
    W[c_idx(2,2)][col] += 1.0
    W[c_idx(3,2)][col] += 1.0
    W[c_idx(2,3)][col] += 1.0
    W[c_idx(3,3)][col] += 1.0
    col += 1
    
    # Term k=1: X[:,1] * Y[1,:]
    U[a_idx(0,1)][col] += 1.0
    U[a_idx(0,3)][col] += 1.0
    U[a_idx(1,1)][col] += 1.0
    U[a_idx(1,3)][col] += 1.0
    V[b_idx(3,2)][col] += 1.0
    V[b_idx(3,3)][col] += 1.0
    W[c_idx(2,0)][col] += 1.0
    W[c_idx(3,0)][col] += 1.0
    W[c_idx(2,1)][col] += 1.0
    W[c_idx(3,1)][col] += 1.0
    W[c_idx(2,2)][col] += 1.0
    W[c_idx(3,2)][col] += 1.0
    W[c_idx(2,3)][col] += 1.0
    W[c_idx(3,3)][col] += 1.0
    col += 1
    
    # M6 = (A21-A11)(B11+B12), contributes +1 to C11
    # X = A21-A11:
    #   X[0,0] = A[2,0] - A[0,0]
    #   X[0,1] = A[2,1] - A[0,1]
    #   X[1,0] = A[3,0] - A[1,0]
    #   X[1,1] = A[3,1] - A[1,1]
    # Y = B11+B12:
    #   Y[0,0] = B[0,0] + B[0,2]
    #   Y[0,1] = B[0,1] + B[0,3]
    #   Y[1,0] = B[1,0] + B[1,2]
    #   Y[1,1] = B[1,1] + B[1,3]
    
    # Term k=0: X[:,0] * Y[0,:]
    U[a_idx(2,0)][col] += 1.0
    U[a_idx(0,0)][col] += -1.0
    U[a_idx(3,0)][col] += 1.0
    U[a_idx(1,0)][col] += -1.0
    V[b_idx(0,0)][col] += 1.0
    V[b_idx(0,2)][col] += 1.0
    V[b_idx(0,1)][col] += 1.0
    V[b_idx(0,3)][col] += 1.0
    W[c_idx(0,0)][col] += 1.0
    W[c_idx(1,0)][col] += 1.0
    W[c_idx(0,1)][col] += 1.0
    W[c_idx(1,1)][col] += 1.0
    col += 1
    
    # Term k=1: X[:,1] * Y[1,:]
    U[a_idx(2,1)][col] += 1.0
    U[a_idx(0,1)][col] += -1.0
    U[a_idx(3,1)][col] += 1.0
    U[a_idx(1,1)][col] += -1.0
    V[b_idx(1,0)][col] += 1.0
    V[b_idx(1,2)][col] += 1.0
    V[b_idx(1,1)][col] += 1.0
    V[b_idx(1,3)][col] += 1.0
    W[c_idx(0,0)][col] += 1.0
    W[c_idx(1,0)][col] += 1.0
    W[c_idx(0,1)][col] += 1.0
    W[c_idx(1,1)][col] += 1.0
    col += 1
    
    # M7 = (A12-A22)(B21+B22), contributes +1 to C22
    # X = A12-A22:
    #   X[0,0] = A[0,2] - A[2,2]
    #   X[0,1] = A[0,3] - A[2,3]
    #   X[1,0] = A[1,2] - A[3,2]
    #   X[1,1] = A[1,3] - A[3,3]
    # Y = B21+B22:
    #   Y[0,0] = B[2,0] + B[2,2]
    #   Y[0,1] = B[2,1] + B[2,3]
    #   Y[1,0] = B[3,0] + B[3,2]
    #   Y[1,1] = B[3,1] + B[3,3]
    
    # Term k=0: X[:,0] * Y[0,:]
    U[a_idx(0,2)][col] += 1.0
    U[a_idx(2,2)][col] += -1.0
    U[a_idx(1,2)][col] += 1.0
    U[a_idx(3,2)][col] += -1.0
    V[b_idx(2,0)][col] += 1.0
    V[b_idx(2,2)][col] += 1.0
    V[b_idx(2,1)][col] += 1.0
    V[b_idx(2,3)][col] += 1.0
    W[c_idx(2,2)][col] += 1.0
    W[c_idx(3,2)][col] += 1.0
    W[c_idx(2,3)][col] += 1.0
    W[c_idx(3,3)][col] += 1.0
    col += 1
    
    # Term k=1: X[:,1] * Y[1,:]
    U[a_idx(0,3)][col] += 1.0
    U[a_idx(2,3)][col] += -1.0
    U[a_idx(1,3)][col] += 1.0
    U[a_idx(3,3)][col] += -1.0
    V[b_idx(3,0)][col] += 1.0
    V[b_idx(3,2)][col] += 1.0
    V[b_idx(3,1)][col] += 1.0
    V[b_idx(3,3)][col] += 1.0
    W[c_idx(2,2)][col] += 1.0
    W[c_idx(3,2)][col] += 1.0
    W[c_idx(2,3)][col] += 1.0
    W[c_idx(3,3)][col] += 1.0
    col += 1
    
    # Now the 4x4*4x1 part: C[:,4] = A[:,:4] * B[:,4]
    # Schoolbook: for i=0..3, j=0..3: C[i,4] += A[i,j] * B[j,4]
    
    for i in range(n):
        for j in range(m):
            U[a_idx(i, j)][col] += 1.0
            V[b_idx(j, 4)][col] += 1.0
            W[c_idx(4, i)][col] += 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
