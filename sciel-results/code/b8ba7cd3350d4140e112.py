"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy: Use Strassen-style decomposition on the block structure.
    Partition A (4x4) and B (4x7) into 2x2 blocks of submatrices:
    A = [[A11, A12], [A21, A22]] where each is 2x2
    B = [[B11, B12, B13], [B21, B22, B23]] where B11,B12,B21,B22 are 2x2 and B13,B23 are 2x1
    
    C = A*B (4x7) = [[C11, C12, C13], [C21, C22, C23]]
    C11 = A11*B11 + A12*B21 (2x2)
    C12 = A11*B12 + A12*B22 (2x2)
    C13 = A11*B13 + A12*B23 (2x1)
    C21 = A21*B11 + A22*B21 (2x2)
    C22 = A21*B12 + A22*B22 (2x2)
    C23 = A21*B13 + A22*B23 (2x1)
    
    For each 2x2 block product, we can use Strassen (rank 7 instead of 8).
    For 2x1 products, use schoolbook (rank 4).
    
    However, a better approach: treat the whole computation as a structured problem.
    We'll build the decomposition by combining Strassen decompositions for each
    pair of 2x2 blocks.
    
    For C11 = A11*B11 + A12*B21:
    This is computing X*Y + Z*W where X=A11, Y=B11, Z=A12, W=B21 (all 2x2).
    We can view this as a (2x2) * (2x2) + (2x2) * (2x2) computation.
    
    A cleaner approach: Use the fact that for 2x2*2x2, Strassen gives rank 7.
    We need to compute 4 such products plus 2 smaller ones.
    
    Let me use a direct approach: build the full rank-r decomposition by
    enumerating all necessary bilinear terms.
    
    Actually, let me use a systematic approach with numpy to verify correctness.
    """
    
    n, m, p = 4, 4, 7
    
    # We'll build the decomposition using Strassen for 2x2*2x2 blocks.
    # 
    # Key insight: C11 = A11*B11 + A12*B21
    # Let's define new variables:
    # For the "top-left" computation (C11, C12, C13, C21, C22, C23):
    # 
    # We can group computations. Let's use a different partition.
    # 
    # Alternative: Use the fact that rank(4x4*4x7) can be computed by
    # treating it as two 2x4*4x7 multiplications or similar.
    # 
    # Let me just build a valid decomposition with reduced rank.
    # 
    # Approach: For each of the 4 "row-pairs" of A and 7 columns of B,
    # we compute 2x4 * 4x7 contributions.
    #
    # Actually, the cleanest approach: 
    # View A as 2 blocks of 2x4, B as 4x7.
    # C_top = A_top * B (2x4 * 4x7 -> 2x7)
    # C_bot = A_bot * B (2x4 * 4x7 -> 2x7)
    #
    # For 2x4 * 4x7: partition B into two 4x2 and one 4x3? No, 7 is odd.
    # B = [B_left (4x3), B_right (4x4)]
    # C_top_left = A_top * B_left (2x4 * 4x3 -> 2x3)
    # C_top_right = A_top * B_right (2x4 * 4x4 -> 2x4)
    #
    # For 2x4 * 4x4: use Strassen-like. Partition A_top into two 2x2:
    # A_top = [A_tl, A_tr] (each 2x2)
    # B_right = [[B_rr1, B_rr2], [B_br1, B_br2]] (each 2x2)
    # C_top_right = A_tl*B_rr1 + A_tr*B_br1 (left 2 cols) + A_tl*B_rr2 + A_tr*B_br2 (right 2 cols)
    #
    # This is getting complex. Let me just use a verified approach.
    #
    # I'll use the following strategy:
    # 1. Build a schoolbook decomposition (rank 112) as baseline
    # 2. Apply Strassen reduction on 2x2*2x2 sub-problems
    #
    # For now, let me construct a rank-70 decomposition using Strassen
    # on the 2x2*2x2 blocks where possible.
    
    # Let me use a cleaner mathematical approach.
    # The tensor T for 4x4*4x7 has entries T[i,j,k] = delta_{j,k} (with appropriate indices)
    # We want T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    
    # Indexing:
    # U has shape (n*m, rank) = (16, rank), indexed by (i,j) where i in [0,4), j in [0,4)
    # V has shape (m*p, rank) = (28, rank), indexed by (j,k) where j in [0,4), k in [0,7)
    # W has shape (p*n, rank) = (28, rank), indexed by (k,i) where k in [0,7), i in [0,4)
    # W[k*n+i, r] = W[k*4+i, r]
    
    # For a rank-1 term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This contributes to C[i,k] when all three are nonzero.
    # For the term to contribute to C[i,k], we need the j index to be consistent:
    # U[i,j] * V[j,k] * W[k,i]
    
    # Schoolbook: for each (i,j,k), U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # This gives rank 4*4*7 = 112.
    
    # Strassen reduction: For 2x2*2x2, instead of 8 terms, use 7.
    # The 7 Strassen terms for matrices X=[[x11,x12],[x21,x22]], Y=[[y11,y12],[y21,y22]]:
    # M1 = (x11+x22)*(y11+y22)
    # M2 = (x21+x22)*y11
    # M3 = x11*(y22-y11)
    # M4 = x22*(y11-y22)
    # M5 = (x11+x12)*y22
    # M6 = (x21-x11)*(y11+y22)
    # M7 = (x12-x22)*(y22-y11)
    # Result:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    
    # For our problem, let's partition:
    # A (4x4): rows 0-1, 2-3; cols 0-1, 2-3
    # B (4x7): rows 0-1, 2-3; cols 0-3, 4-6 (split into 4 and 3)
    
    # C (4x7): rows 0-1, 2-3; cols 0-3, 4-6
    
    # C[0:2, 0:4] = A[0:2,0:2]*B[0:2,0:4] + A[0:2,2:4]*B[2:4,0:4]
    #   = A11*B11 + A12*B21  where B11 is 2x4, B21 is 2x4
    # C[0:2, 4:7] = A[0:2,0:2]*B[0:2,4:7] + A[0:2,2:4]*B[2:4,4:7]
    #   = A11*B12 + A12*B22  where B12 is 2x3, B22 is 2x3
    # C[2:4, 0:4] = A[2:4,0:2]*B[0:2,0:4] + A[2:4,2:4]*B[2:4,0:4]
    #   = A21*B11 + A22*B21
    # C[2:4, 4:7] = A[2:4,0:2]*B[0:2,4:7] + A[2:4,2:4]*B[2:4,4:7]
    #   = A21*B12 + A22*B22
    
    # For the 2x2 * 2x4 products, we can further split B into 2x2 + 2x2:
    # B[0:2,0:4] = [B[0:2,0:2], B[0:2,2:4]] = [B11a, B11b]
    # So A11*B11 = A11*B11a + A11*B11b (two 2x2*2x2 products)
    # Similarly for others.
    
    # Total 2x2*2x2 products needed:
    # C[0:2,0:2] = A11*B11a + A12*B21a  (2 products)
    # C[0:2,2:4] = A11*B11b + A12*B21b  (2 products)
    # C[0:2,4:6] = A11*B12a + A12*B22a  (2 products)
    # C[0:2,6:7] = A11*B12b + A12*B22b  (2 products, but these are 2x2*2x1)
    # C[2:4,0:2] = A21*B11a + A22*B21a  (2 products)
    # C[2:4,2:4] = A21*B11b + A22*B21b  (2 products)
    # C[2:4,4:6] = A21*B12a + A22*B22a  (2 products)
    # C[2:4,6:7] = A21*B12b + A22*B22b  (2 products, 2x2*2x1)
    
    # So we have 16 2x2*2x2 products and 4 2x2*2x1 products.
    # Using Strassen: 16*7 + 4*4 = 112 + 16 = 128? That's worse!
    # Wait, that can't be right. Let me recount.
    
    # Actually, the schoolbook for 2x2*2x2 is rank 8, and Strassen is rank 7.
    # 16 * 8 = 128 for schoolbook, 16 * 7 = 112 for Strassen.
    # Plus 4 * 4 = 16 for the 2x2*2x1 parts.
    # Total: 112 + 16 = 128 for Strassen+schoolbook, vs 128+16=144 for all schoolbook?
    # No wait, original schoolbook is 4*4*7 = 112.
    
    # I think I'm overcomplicating this. Let me just use a different strategy.
    
    # Strategy: Use the known result that rank(4x4*4x7) <= 64.
    # Reference: The rank of matrix multiplication tensor M_{n,m,p} satisfies
    # rank(M_{n,m,p}) <= n*m*p (schoolbook) and can be improved.
    
    # For now, let me construct a valid rank-112 decomposition (schoolbook)
    # and then try to optimize.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
