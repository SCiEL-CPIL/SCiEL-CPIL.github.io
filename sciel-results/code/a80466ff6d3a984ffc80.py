"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130  # 150s budget, 20s safety margin

def _build_2x2_strassen():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    
    Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products for 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C[0,0] = M1 + M4 - M5 + M6
    # C[0,1] = M2 + M5
    # C[1,0] = M3 + M4
    # C[1,1] = M1 - M2 + M3 + M7
    #
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c + d)(f - h)
    # M7 = (a - b)(g + e)
    #
    # For U: row index i*m+j
    # For V: row index j*p+k
    # For W: row index k*n+i
    #
    # Each Strassen product is a sum of tensor terms. Let me work it out.
    
    # Let me use the standard approach: each of the 7 Strassen products
    # corresponds to a bilinear form. For 2x2x2, the schoolbook has 8 terms.
    # Strassen reduces to 7.
    #
    # Product 1: (a+d)(e+h) = ae + ah + de + dh
    #   a = A[0,0] -> U[0*2+0] = U[0]
    #   d = A[1,1] -> U[1*2+1] = U[3]
    #   e = B[0,0] -> V[0*2+0] = V[0]
    #   h = B[1,1] -> V[1*2+1] = V[3]
    #   C[0,0] gets +1, C[0,1] gets +1, C[1,0] gets +1, C[1,1] gets -1
    #   W[0*2+0]=W[0]: +1, W[0*2+1]=W[1]: +1, W[1*2+0]=W[2]: +1, W[1*2+1]=W[3]: -1
    
    # This is getting complex. Let me use a known decomposition.
    # For 2x2x2, Strassen's decomposition in terms of U, V, W:
    
    # I'll use the standard Strassen coefficients.
    # U columns correspond to the 7 Strassen products.
    
    # Product 1: M1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C[0,0]
    W[1, 0] = 1  # C[0,1]
    W[2, 0] = 1  # C[1,0]
    W[3, 0] = -1 # C[1,1]
    
    # Product 2: M2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[0, 1] = -1 # C[0,0] (contributes -M2)
    W[1, 1] = 1  # C[0,1]
    W[2, 1] = 1  # C[1,0]
    W[3, 1] = 1  # C[1,1] (-M2)
    
    # Wait, let me recheck the formulas:
    # C[0,0] = M1 - M2 + M3 + M6
    # C[0,1] = M5 + M2
    # C[1,0] = M4 + M3
    # C[1,1] = M1 - M2 + M3 + M7
    
    # Let me redo this carefully.
    
    # Reset
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h): expands to ae + ah + de + dh
    # Contributions to C: C[0,0]+=M1, C[0,1]+=M1, C[1,0]+=M1, C[1,1]-=M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1
    
    # M2 = (c+d)e: ce + de
    # Contributions: C[0,0]-=M2, C[0,1]+=M2, C[1,0]+=M2, C[1,1]-=M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[0,1]=-1; W[1,1]=1; W[2,1]=1; W[3,1]=-1
    
    # M3 = a(f-h): af - ah
    # Contributions: C[0,0]+=M3, C[1,0]+=M3, C[1,1]+=M3
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[0,2]=1; W[2,2]=1; W[3,2]=1
    
    # M4 = d(g-e): dg - de
    # Contributions: C[1,0]+=M4
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[2,3]=1
    
    # M5 = (a+b)h: ah + bh
    # Contributions: C[0,1]+=M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1
    
    # M6 = (c+d)(f-h): cf - ch + df - dh
    # Contributions: C[0,0]+=M6
    U[2,5]=1; U[3,5]=1
    V[1,5]=1; V[3,5]=-1
    W[0,5]=1
    
    # M7 = (a-b)(g+e): ag + ae - bg - be
    # Contributions: C[1,1]+=M7
    U[0,6]=1; U[1,6]=-1
    V[2,6]=1; V[0,6]=1
    W[3,6]=1
    
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 multiplication tensor using recursive Strassen.
    
    Uses the 2x2 Strassen decomposition recursively.
    Returns U (16, 49), V (16, 49), W (16, 49).
    """
    U2, V2, W2 = _build_2x2_strassen()
    # U2, V2, W2 are 4x7
    
    # For 4x4x4: n=m=p=4
    # Split into 2x2 blocks.
    # A is 4x4, split into blocks A[i1*2+i2, j1*2+j2] where i1,i2,j1,j2 in {0,1}
    # B is 4x4, split into blocks B[j1*2+j2, k1*2+k2]
    # C is 4x4, C[i1*2+i2, k1*2+k2]
    
    # Using Strassen on the block level:
    # 7 block products, each computed using the 2x2 Strassen decomposition.
    # Total rank = 7 * 7 = 49.
    
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block-level Strassen products (same as 2x2 case but for 2x2 blocks):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10+A11)(B01-B11)
    # M7 = (A00-A01)(B10+B00)
    
    # Each block is 2x2, so each block product uses the 2x2 Strassen (rank 7).
    
    # Define block indices:
    # A blocks: A00 = rows 0-1, cols 0-1; A01 = rows 0-1, cols 2-3
    #            A10 = rows 2-3, cols 0-1; A11 = rows 2-3, cols 2-3
    # B blocks: B00 = rows 0-1, cols 0-1; B01 = rows 0-1, cols 2-3
    #            B10 = rows 2-3, cols 0-1; B11 = rows 2-3, cols 2-3
    # C blocks: C00 = rows 0-1, cols 0-1; C01 = rows 0-1, cols 2-3
    #            C10 = rows 2-3, cols 0-1; C11 = rows 2-3, cols 2-3
    
    # For each block-level Strassen product r in 0..6:
    #   It involves certain A-blocks and B-blocks.
    #   For each sub-product s in 0..6 of the 2x2 Strassen:
    #     The entry in U is the product of the corresponding U2 entries for the A-blocks.
    
    # Let me define the block structure more carefully.
    # For the 2x2 Strassen at the block level, we have 7 products.
    # For each product, we need to know which A-blocks and B-blocks are involved
    # and with what signs, and then apply the 2x2 Strassen decomposition within each block.
    
    # Actually, the recursive structure is:
    # The 4x4x4 tensor = sum over 7 block products, each of which is a 2x2x2 tensor
    # (on the block indices) with entries that are 2x2x2 tensors (on the sub-indices).
    
    # So U[i*4+j, c] where c = r*7 + s:
    #   U[i*4+j, r*7+s] = sum over the appropriate sub-indices of U2[...]*U2[...]
    
    # This is getting complicated. Let me use a different approach:
    # Directly construct using the Kronecker product structure.
    
    # The 4x4x4 Strassen decomposition can be seen as:
    # U_4[i*4+j, c] = sum_{i1,j1,i2,j2} U2[i1*2+j1, r] * U2[i2*2+j2, s]
    # where i = i1*2+i2, j = j1*2+j2, c = r*7+s
    # But this isn't quite right either because the Strassen structure at the block level
    # mixes the block indices in a specific way.
    
    # Let me try a cleaner approach: use the known fact that the rank of 4x4x4
    # matrix multiplication is at most 49 (from recursive Strassen), and construct
    # the decomposition explicitly.
    
    # Actually, let me just use the tensor product approach properly.
    # The 2x2 Strassen gives us U2, V2, W2 for multiplying 2x2 matrices.
    # For 4x4, we treat it as 2x2 blocks of 2x2 matrices.
    
    # The block-level Strassen has 7 products. For block product r:
    #   M_r = (sum_{a_blocks} alpha_{r,a} * A_a) * (sum_{b_blocks} beta_{r,b} * B_b)
    # where a and b range over the 4 blocks.
    
    # For each block product r and each sub-index (i2,j2,k2):
    #   The contribution to C[i1*2+i2, k1*2+k2] from block (i1,k1) is:
    #     sum over (j1) of the block product
    
    # I think the cleanest way is to just enumerate.
    
    # Let me define the 7 block products with their A and B block coefficients:
    # Block A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Block B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Block C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # Product 0: (A00+A11)(B00+B11)
    # Product 1: (A10+A11)B00
    # Product 2: A00(B01-B11)
    # Product 3: A11(B10-B00)
    # Product 4: (A00+A01)B11
    # Product 5: (A10+A11)(B01-B11)
    # Product 6: (A00-A01)(B10+B00)
    
    # C block contributions:
    # C00 = P0 - P1 + P2 + P5
    # C01 = P4 + P1
    # C10 = P3 + P2
    # C11 = P0 - P1 + P2 + P6
    
    # For each block product r, the A-blocks involved and their coefficients:
    # P0: A00(+1), A11(+1); B00(+1), B11(+1)
    # P1: A10(+1), A11(+1); B00(+1)
    # P2: A00(+1); B01(+1), B11(-1)
    # P3: A11(+1); B10(+1), B00(-1)
    # P4: A00(+1), A01(+1); B11(+1)
    # P5: A10(+1), A11(+1); B01(+1), B11(-1)
    # P6: A00(+1), A01(-1); B10(+1), B00(+1)
    
    # For each block product, we compute the 2x2 sub-matrix product using U2, V2, W2.
    # The sub-product index s (0..6) corresponds to the 2x2 Strassen sub-product.
    
    # Let me build this step by step.
    
    # For a given block product r, the A-part is a linear combination of A-blocks,
    # and the B-part is a linear combination of B-blocks.
    # The product is:
    #   (sum_a alpha_a * A_a) * (sum_b beta_b * B_b)
    #   = sum_{a,b} alpha_a * beta_b * A_a * B_b
    
    # Each A_a * B_b is a 2x2 * 2x2 multiplication, which can be decomposed
    # using the 2x2 Strassen: U2, V2, W2.
    
    # So for block product r and sub-product s:
    #   U[i*4+j, r*7+s] = sum_{a,b} alpha_{r,a} * beta_{r,b} * 
    #                       (contribution of A_a * B_b to U[i*4+j, s])
    
    # But A_a * B_b means: A-block a times B-block b.
    # A-block a = A[i1*2+i2, j1*2+j2] where (i1,j1) determines the block.
    # Wait, I need to be more careful.
    
    # A-block 0 (A00): rows 0-1, cols 0-1 -> i1=0, j1=0
    # A-block 1 (A01): rows 0-1, cols 2-3 -> i1=0, j1=1
    # A-block 2 (A10): rows 2-3, cols 0-1 -> i1=1, j1=0
    # A-block 3 (A11): rows 2-3, cols 2-3 -> i1=1, j1=1
    
    # Similarly for B-blocks.
    
    # For the 2x2 Strassen decomposition of A_a * B_b:
    # The result is a 2x2 matrix C_sub[i2, k2] (where i2, k2 are sub-indices).
    # C_sub = A_sub * B_sub where A_sub is the 2x2 sub-matrix from A-block a,
    # and B_sub is the 2x2 sub-matrix from B-block b.
    
    # Using the 2x2 Strassen: C_sub[i2, k2] = sum_s U2[i2*2+j2, s] * V2[j2*2+k2, s] * W2[k2*2+i2, s]
    # Wait, that's not right. The decomposition is:
    # C_sub[i2, k2] = sum_s U2[i2*2+j2, s] * V2[j2*2+k2, s] * W2[k2*2+i2, s]
    # No, that's not how it works. The tensor decomposition is:
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # where T[i,j,k] is the coefficient of A[i,?] * B[?,j] * C[?,k]... 
    
    # I'm getting confused with the index conventions. Let me restart with a clearer approach.
    
    # The matrix multiplication tensor T for n x m times m x p gives n x p:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # In the factorization: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For 2x2x2: i,j,k in {0,1}
    # U has shape (4, 7), indexed by (i*2+j, r)
    # V has shape (4, 7), indexed by (j*2+k, r)
    # W has shape (4, 7), indexed by (k*2+i, r)
    
    # For 4x4x4: i,j,k in {0,1,2,3}
    # U has shape (16, 49), indexed by (i*4+j, r)
    # V has shape (16, 49), indexed by (j*4+k, r)
    # W has shape (16, 49), indexed by (k*4+i, r)
    
    # The recursive Strassen: treat 4x4 as 2x2 blocks of 2x2.
    # i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2 where i1,i2,j1,j2,k1,k2 in {0,1}
    
    # The block-level tensor is 2x2x2 (on i1, j1, k1), and for each block,
    # the sub-tensor is 2x2x2 (on i2, j2, k2).
    
    # So: T4[i,j,k] = sum_{r1,r2} U4[i*4+j, r1*7+r2] * V4[j*4+k, r1*7+r2] * W4[k*4+i, r1*7+r2]
    
    # And: U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    #     V4[j*4+k, r1*7+r2] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    #     W4[k*4+i, r1*7+r2] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    # This is the Kronecker product structure!
    
    U2, V2, W2 = _build_2x2_strassen()
    
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            c = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, c] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k, c] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i, c] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
    
    return U, V, W

def _build_4x4x8():
    """Build 4x4x8 multiplication tensor.
    
    Splits the 8 columns of B into two groups of 4.
    Uses two copies of the 4x4x4 Strassen decomposition.
    Total rank = 49 + 49 = 98.
    """
    U4, V4, W4 = _build_4x4x4_strassen()
    
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First half: B columns 0-3, C columns 0-3
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4  # B[0:4, 0:4]
    W[0:16, 0:49] = W4  # C[0:4, 0:4]
    # V[16:32, 0:49] = 0 (already zero)
    # W[16:32, 0:49] = 0 (already zero)
    
    # Second half: B columns 4-7, C columns 4-7
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4  # B[0:4, 4:8] -> rows 16:32 in V (j*p+k where j in 0-3, k in 4-7)
    # Wait, V is indexed by j*p+k where j in 0..m-1=0..3, k in 0..p-1=0..7
    # So j=0, k=4 -> index 0*8+4 = 4
    # j=0, k=5 -> index 5
    # j=0, k=6 -> index 6
    # j=0, k=7 -> index 7
    # j=1, k=4 -> index 1*8+4 = 12
    # etc.
    
    # I need to be more careful with the indexing.
    # V[j*p+k, r] where j in 0..3, k in 0..7
    # For the first 4x4x4: j in 0..3, k in 0..3 -> V indices 0..15
    # For the second 4x4x4: j in 0..3, k in 4..7 -> V indices 4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31
    
    # Let me redo this.
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First copy: handles C[0:4, 0:4] = A * B[:, 0:4]
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second copy: handles C[0:4, 4:8] = A * B[:, 4:8]
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return U, V, W

def construct():
    """Return the tensor decomposition for 4x4x8 matrix multiplication.
    
    Uses recursive Strassen's algorithm:
    - 2x2x2: rank 7
    - 4x4x4: rank 49 (7x7 from Kronecker product)
    - 4x4x8: rank 98 (two copies of 4x4x4)
    
    Returns dict with U, V, W as lists of lists with integer entries.
    """
    U, V, W = _build_4x4x8()
    
    # Convert to lists of lists with integer values
    U_list = [[int(round(val)) for val in row] for row in U]
    V_list = [[int(round(val)) for val in row] for row in V]
    W_list = [[int(round(val)) for val in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
