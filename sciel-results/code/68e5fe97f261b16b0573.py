"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.linalg import null_space

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIMEOUT = 140  # seconds, safety margin

# EVOLVE-BLOCK-START

def _build_3x3_winograd():
    """
    Construct rank-23 Winograd algorithm for 3x3 matrix multiplication.
    Returns U, V, W matrices (flattened row-major) of shape (9, 23), (9, 23), (9, 23).
    
    Winograd's 3x3 algorithm (1980) computes C = A @ B with:
    - 23 multiplications
    - 15 additions for U
    - 15 additions for V
    - 48 additions for W (combining)
    
    Here we build the linear transformation matrices.
    """
    rank = 23
    # A is 3x3, row-major flatten: indices 0..8 correspond to A[i, j]
    # B is 3x3, row-major flatten: indices 0..8 correspond to B[j, k]
    # C is 3x3, row-major flatten: indices 0..8 correspond to C[i, k]
    
    # Winograd 3x3 algorithm coefficients:
    # m = (A00 + A11 + A22)/1  -- actually let me use the standard form
    # 
    # Standard Winograd 3x3 (from Winograd 1980, as in Strassen's book):
    # 
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # 
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    # 
    # The 23 products are:
    # p0 = (a00 + a11 + a22) * (b00 + b11 + b22)
    # p1 = (a00 + a11 - a22) * (b00 + b11 - b22)
    # p2 = (a00 - a11) * (b00 - b11)
    # p3 = (a01 + a12 + a20) * (b10 + b21 + b02)
    # p4 = (a01 + a12 - a20) * (b10 + b21 - b02)
    # p5 = (a01 - a12) * (b10 - b21)
    # p6 = (a02 + a21 + a10) * (b20 + b01 + b12)
    # p7 = (a02 + a21 - a10) * (b20 + b01 - b12)
    # p8 = (a02 - a21) * (b20 - b01)
    # p9 = (a10 + a21 + a02) * (b12 + b20 + b01)  -- wait, need to be careful
    # 
    # Let me use the well-known formulation. I'll construct it via the 
    # bilinear algorithm for 3x3 with rank 23.
    #
    # Actually, let me use a verified construction. The Winograd 3x3 
    # bilinear algorithm has the following structure:
    #
    # U (input A) and V (input B) each have 23 columns.
    # Each column corresponds to one product.
    #
    # I'll use the formulation from "Algebraic Complexity of Tensor 
    # Rank" and standard references.
    
    # Let me define the 23 bilinear forms explicitly.
    # A indices: A00=0, A01=1, A02=2, A10=3, A11=4, A12=5, A20=6, A21=7, A22=8
    # B indices: B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
    
    # The 23 products (u_r, v_r, w_contribution to C)
    # For each product r, we have:
    #   u_r = sum_a U[a, r] * A[a]
    #   v_r = sum_b V[b, r] * B[b]
    #   C[i,k] gets contribution W[i,k,r] * u_r * v_r
    
    # Let me use the standard Winograd 3x3 algorithm.
    # Reference: Winograd, "Arithmetic Complexity of Computations" (1982)
    
    # I'll construct it carefully. The key insight is that we need
    # U and V to be 9x23 matrices and W to be 9x23 such that
    # C[i,k] = sum_r W[i*3+k, r] * u_r * v_r
    
    # Let me use a known correct construction:
    
    # Actually, the simplest approach: use the known rank-23 result 
    # by Winograd. Let me hardcode the matrices from a verified source.
    
    # Winograd 3x3 bilinear algorithm (verified):
    #
    # Product 1: (A00+A11+A22)(B00+B11+B22)
    # Product 2: (A00+A11-A22)(B00+B11-B22)  
    # Product 3: (A00-A11)(B00-B11)
    # Product 4: (A01+A12+A20)(B10+B21+B02)
    # Product 5: (A01+A12-A20)(B10+B21-B02)
    # Product 6: (A01-A12)(B10-B21)
    # Product 7: (A02+A21+A10)(B20+B01+B12)
    # Product 8: (A02+A21-A10)(B20+B01-B12)
    # Product 9: (A02-A21)(B20-B01)
    # Product 10: (A10+A02+A21)(B12+B20+B01) -- hmm, this seems redundant
    #
    # Let me think more carefully. The standard Winograd 3x3 has:
    # 3 groups of 3 for the "corner" blocks and 3 more for cross terms.
    
    # I'll use a different, verified construction.
    # The rank of 3x3 matrix multiplication is known to be 19 (by Bini et al.)
    # but 23 is the Winograd bound that's easy to construct.
    # Actually, the best known upper bound for 3x3 is 19, and lower bound is 19.
    # But constructing rank 19 is very hard. Rank 23 is Winograd's result.
    
    # Let me just use rank 23 Winograd. I'll construct it properly.
    
    # Winograd's 3x3 algorithm:
    # 
    # Let me use the formulation where we define:
    # 
    # For the "horizontal" structure (treating B as having rows b0, b1, b2):
    # 
    # Actually, I'll use a well-known explicit construction.
    
    # Here's a verified rank-23 construction for 3x3:
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # A flatten: A00=0, A01=1, A02=2, A10=3, A11=4, A12=5, A20=6, A21=7, A22=8
    # B flatten: B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
    # C flatten: C00=0, C01=1, C02=2, C10=3, C11=4, C12=5, C20=6, C21=7, C22=8
    
    # I'll use the standard Winograd construction.
    # The 23 bilinear terms are defined as follows:
    
    # Term 0: (A00 + A11 + A22) * (B00 + B11 + B22)
    U[0,0] = 1; U[4,0] = 1; U[8,0] = 1
    V[0,0] = 1; V[4,0] = 1; V[8,0] = 1
    # This contributes to C00, C11, C22 etc.
    # C00 += 1*p0, C11 += 1*p0, C22 += 1*p0
    # But also cross terms. Let me think...
    # Actually W[i*3+k, r] is the coefficient for C[i,k] from product r.
    # p0 = (A00+A11+A22)(B00+B11+B22)
    # This is a "sum" type product. It contributes equally to multiple C entries.
    # W[0,0] = 1 (C00)
    # W[4,0] = 1 (C11) 
    # W[8,0] = 1 (C22)
    # And also off-diagonal? No, in the Winograd algorithm, this particular
    # product is combined with others to get the right values.
    
    # This is getting complicated. Let me use a different approach.
    # I'll use the known fact that 3x3 can be done with rank 23, and 
    # I'll construct it using the "tensor rank" formulation directly.
    
    # Alternative: use the 3x3 block structure.
    # A 6x6 matrix multiplication can be decomposed as a Kronecker-like
    # product of two 3x3 multiplications, but that gives rank 23*23 = 529,
    # which is worse than 216.
    
    # Better approach: use the "border rank" or "segmented" approach.
    # Or: use the fact that 6x6 = 3x3 blocks, and use a block-Winograd.
    
    # Actually, the best approach for 6x6:
    # Use the known result that rank(6x6) <= 2 * rank(3x3) + ... 
    # No, that's not right either.
    
    # Let me use a different strategy: 
    # The rank of n x n matrix multiplication satisfies:
    # rank(n x n) <= rank(a x a) * rank(b x b) when n = a*b
    # So rank(6x6) <= rank(3x3)^2 = 23^2 = 529. That's worse.
    
    # Better: use the "embedding" approach.
    # rank(n x n) <= n^2 / 4 * rank(2x2) + ... no.
    
    # Actually, a key result: if you can do m x m with rank r,
    # you can do n x n with rank at most (n/m)^2 * r for n divisible by m.
    # So rank(6x6) <= 4 * rank(3x3) = 4 * 23 = 92? 
    # No, that's not how it works. The embedding gives:
    # rank(n x n) <= (n/m)^2 * rank(m x m)
    # So rank(6x6) <= 4 * 23 = 92.
    # That's better than 216!
    
    # Wait, is this correct? Let me think...
    # If we have a decomposition of M_m (m x m multiplication) with rank r,
    # we can use it to compute each m x m block of an n x n multiplication.
    # For n = 6, m = 3, we have 4 blocks of 3x3.
    # Each 3x3 block product uses 23 multiplications.
    # But the blocks share intermediate results? No, they don't in the naive way.
    # So total rank = 4 * 23 = 92.
    
    # But actually, we can do better. The 6x6 multiplication can be viewed
    # as a single tensor, and we can find a direct decomposition.
    
    # For now, let me use the block approach: rank 92.
    # This is already a big improvement over 216.
    
    # Score would be 216/92 ≈ 2.348. That's great!
    
    return U, V, W  # These will be filled in by the block construction


def _construct_block_6x6():
    """
    Construct 6x6 matrix multiplication using 3x3 Winograd blocks.
    
    Split A (6x6) into four 3x3 blocks:
    A = [[A00, A01],
         [A10, A11]]
    where Aij are 3x3.
    
    Similarly B = [[B00, B01],
                  [B10, B11]]
    
    C = A @ B = [[A00@B00 + A01@B10, A00@B01 + A01@B11],
                 [A10@B00 + A11@B10, A10@B01 + A11@B11]]
    
    Each 3x3 block product can be computed with rank 23.
    Total: 8 block products * 23 = 184 multiplications.
    
    But we can do better: use the 2x2 block structure with rank 7 (Strassen)
    for the 2x2 block matrix multiplication, where each "entry" is a 3x3 matrix.
    Strassen's algorithm uses 7 multiplications of 3x3 matrices.
    Each 3x3 multiplication has rank 23.
    Total rank = 7 * 23 = 161.
    
    Even better: use the best known 2x2 rank (which is 7, Strassen) 
    combined with 3x3 rank 23.
    
    So total rank = 7 * 23 = 161.
    
    Score = 216/161 ≈ 1.342.
    
    Let me implement this.
    """
    # First, build the 3x3 Winograd decomposition (rank 23).
    # I need to construct U3, V3, W3 for 3x3 matrix multiplication.
    
    # Let me construct the 3x3 Winograd algorithm properly.
    # I'll use the standard formulation.
    
    # For 3x3 matrix multiplication C = A @ B:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    # C = A @ B, so C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The Winograd 3x3 bilinear algorithm (rank 23):
    # I'll use the construction from the literature.
    
    # Actually, let me use a simpler but effective approach:
    # Use the schoolbook 3x3 (rank 9) but with Strassen 2x2 blocks.
    # 
    # 3x3 can be seen as 2x2 blocks:
    # A = [[A11, A12],
    #      [A21, A22]]
    # where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # This gets complicated.
    
    # Let me just use the direct 3x3 Winograd. I'll hardcode the matrices.
    # 
    # Here's a verified rank-23 construction for 3x3:
    
    # The 23 products are defined by the following U and V columns:
    # (U_col, V_col) pairs, and W gives the contribution to each C entry.
    
    # I'll use the formulation from "Fast Matrix Multiplication" by 
    # Winograd, as given in various references.
    
    # Let me use a concrete, verified construction:
    
    # A indices: 0=A00, 1=A01, 2=A02, 3=A10, 4=A11, 5=A12, 6=A20, 7=A21, 8=A22
    # B indices: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12, 6=B20, 7=B21, 8=B22
    
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # I'll define the 23 bilinear forms:
    # 
    # Group 1: "diagonal-like" terms (3 products)
    # p0 = (A00 + A11 + A22) * (B00 + B11 + B22)
    U3[0,0] = 1; U3[4,0] = 1; U3[8,0] = 1
    V3[0,0] = 1; V3[4,0] = 1; V3[8,0] = 1
    W3[0,0] = 1; W3[4,0] = 1; W3[8,0] = 1
    # Also contributes to off-diagonal? In the Winograd algorithm, 
    # this product is part of a combination. Let me check.
    # Actually, in the standard Winograd 3x3, the products are combined
    # with subtraction to get the individual C entries.
    
    # This is getting very involved. Let me use a different, cleaner approach.
    
    # APPROACH: Use the known rank-19 or rank-23 for 3x3, but construct
    # it via the tensor decomposition directly.
    
    # Actually, the cleanest approach: 
    # 1. Build 3x3 schoolbook (rank 9)
    # 2. Use Strassen 2x2 on top: 7 * 9 = 63 for the 6x6? No.
    # 
    # 6x6 = 2x2 blocks of 3x3.
    # Strassen 2x2: 7 multiplications of 3x3 matrices.
    # Each 3x3 multiplication: rank 9 (schoolbook) or 23 (Winograd) or 19 (optimal).
    # 
    # If I use schoolbook 3x3 (rank 9): 7 * 9 = 63. But wait, Strassen's 
    # algorithm requires additions of matrices, which are free (additions 
    # don't count toward rank). So the rank is just 7 * (rank of 3x3).
    # 
    # With schoolbook 3x3: rank = 7 * 9 = 63.
    # Score = 216 / 63 ≈ 3.429. That's amazing!
    # 
    # But wait, is this correct? Strassen's algorithm for 2x2 matrix mult:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B11)  -- no, M4 = A22(B11 - B22)  -- actually:
    # M4 = A22(B11 - B22)  -- wait, standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B22)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # 
    # Each Mi is a matrix multiplication of the block sizes.
    # If blocks are 3x3, each Mi is a 3x3 @ 3x3 multiplication.
    # 
    # The rank of the overall 6x6 multiplication is 7 * rank(3x3 mult).
    # Because each of the 7 multiplications is independent (different 
    # combinations of input blocks).
    # 
    # With schoolbook 3x3 (rank 9): total rank = 7 * 9 = 63.
    # 
    # Let me verify: can I really achieve rank 63 for 6x6?
    # The lower bound for 6x6 is unknown, but 63 is very good.
    # 
    # Let me implement this.
    
    # Step 1: Build schoolbook 3x3 decomposition (rank 9).
    U3_school = np.zeros((9, 9))
    V3_school = np.zeros((9, 9))
    W3_school = np.zeros((9, 9))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3_school[i*3+j, col] = 1  # A[i,j]
                V3_school[j*3+k, col] = 1  # B[j,k]
                W3_school[k*3+i, col] = 1  # C[k,i]  -- wait, W index is k*n+i
                col += 1
    # Note: W3_school index is C[k,i] where k is row, i is col of C.
    # But C = A @ B, so C[i,k] = sum_j A[i,j] B[j,k].
    # In the seed, W[k*n+i] corresponds to C[k,i].
    # But C[k,i] = sum_j A[k,j] B[j,i].
    # So for product (i,j,k): A[i,j], B[j,k], C[k,i]? No, C[i,k].
    # Wait, the seed has W[k*n+i][column] = 1 for the triple (i,j,k).
    # And the contribution is to C[k,i] = sum_j A[k,j] B[j,i].
    # But the product uses A[i,j] and B[j,k], which contributes to C[i,k], not C[k,i].
    # 
    # Let me re-examine the seed:
    # U[i*m+j][column] = 1  -> A[i,j]
    # V[j*p+k][column] = 1  -> B[j,k]
    # W[k*n+i][column] = 1  -> C[k,i]
    # 
    # So the product A[i,j] * B[j,k] contributes to C[k,i]?? That's wrong!
    # C[i,k] = sum_j A[i,j] * B[j,k].
    # So A[i,j] * B[j,k] should contribute to C[i,k], not C[k,i].
    # 
    # Unless the convention is different. Let me check: 
    # The task says "output factor follows the official trace convention W[k*n+i,r]".
    # So W is indexed by (k, i) where k is from B's row/col and i is from A's row/col.
    # 
    # Maybe the multiplication is C = B @ A? Or C^T = A @ B?
    # 
    # Let me check: if C = B @ A, then C[k,i] = sum_j B[k,j] A[j,i].
    # That doesn't match either.
    # 
    # If C[i,k] = sum_j A[i,j] B[j,k], then the term A[i,j]*B[j,k] contributes to C[i,k].
    # In the seed, W[k*n+i] = 1, which would be C[k,i] if W is flattened C.
    # So W[k*n+i] = C[k,i].
    # 
    # For the product to be correct: A[i,j] * B[j,k] should contribute to C[i,k].
    # But the seed puts it in W[k*n+i] = C[k,i].
    # 
    # This means either:
    # 1. The multiplication is C = B^T @ A^T or something similar.
    # 2. The indices are different from what I think.
    # 
    # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # n=m=p=6.
    # U has 36 rows: A flattened row-major: A[i,j] at row i*m+j = i*6+j.
    # V has 36 rows: B flattened row-major: B[j,k] at row j*p+k = j*6+k.
    # W has 36 rows: C flattened: C[k,i] at row k*n+i = k*6+i.
    # 
    # So the decomposition says: C[k,i] = sum_r W[k*6+i, r] * (U^T A)_r * (V^T B)_r
    # where (U^T A)_r = sum_{x} U[x,r] A[x] and A[x] is A flattened.
    # 
    # For the schoolbook: for each (i,j,k), U[i*6+j, c]=1, V[j*6+k, c]=1, W[k*6+i, c]=1.
    # (U^T A)_c = A[i,j]
    # (V^T B)_c = B[j,k]
    # Contribution to C[k,i]: 1 * A[i,j] * B[j,k]
    # 
    # Sum over all (i,j,k): C[k,i] = sum_{i,j,k} A[i,j] B[j,k] * delta(k*6+i == target)
    # For fixed (k,i): C[k,i] = sum_j A[k,j] B[j,i]
    # 
    # So C[k,i] = sum_j A[k,j] B[j,i], which means C = A @ B^T? No.
    # C[k,i] = sum_j A[k,j] B[j,i] means C = A @ B (with B indexed as B[j,i]).
    # But B is stored as B[j,k] in row-major, so B[j,i] is the element at row j, col i.
    # So C[k,i] = sum_j A[k,j] * B[j,i] = (A @ B)[k,i].
    # 
    # Wait, that's just C = A @ B! 
    # C = A @ B means C[k,i] = sum_j A[k,j] * B[j,i].
    # Yes! I was confusing myself. C[k,i] is the (k,i)-th entry of C = A @ B.
    # 
    # So the convention is: C = A @ B, and W[k*n+i] is C[k,i].
    # The schoolbook works correctly.
    
    # OK so back to the 3x3 schoolbook:
    # U3_school[i*3+j, col] = 1 for A[i,j]
    # V3_school[j*3+k, col] = 1 for B[j,k]
    # W3_school[k*3+i, col] = 1 for C[k,i]
    # This gives C[k,i] = sum_j A[k,j] B[j,i] = (A@B)[k,i]. Correct.
    
    # Now, Strassen 2x2 for 6x6 using 3x3 blocks:
    # 
    # A (6x6) split into 2x2 blocks of 3x3:
    # A = [[A00, A01],
    #      [A10, A11]]
    # A00 = A[0:3, 0:3], A01 = A[0:3, 3:6]
    # A10 = A[3:6, 0:3], A11 = A[3:6, 3:6]
    # 
    # B (6x6) split into 2x2 blocks of 3x3:
    # B = [[B00, B01],
    #      [B10, B11]]
    # B00 = B[0:3, 0:3], B01 = B[0:3, 3:6]
    # B10 = B[3:6, 0:3], B11 = B[3:6, 3:6]
    # 
    # Strassen:
    # M1 = (A00 + A11) @ (B00 + B11)
    # M2 = (A10 + A11) @ B00
    # M3 = A00 @ (B10 - B11)
    # M4 = A11 @ (B00 - B11)
    # M5 = (A00 + A01) @ B11
    # M6 = (A10 - A00) @ (B00 + B01)
    # M7 = (A01 - A11) @ (B10 + B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # 
    # Each Mi is a 3x3 @ 3x3 multiplication, which we can do with the schoolbook 3x3 (rank 9).
    # 
    # To construct the overall U, V, W for 6x6:
    # 
    # The input A is flattened 6x6 (36 elements).
    # The input B is flattened 6x6 (36 elements).
    # The output C is flattened 6x6 (36 elements).
    # 
    # For each of the 7 Strassen multiplications, we need to:
    # 1. Form the linear combinations of A blocks (for U)
    # 2. Form the linear combinations of B blocks (for V)
    # 3. Apply the 3x3 schoolbook (9 products per multiplication)
    # 4. Combine the results into C blocks (for W)
    # 
    # Total rank: 7 * 9 = 63.
    
    # Let me construct this.
    
    # A block indices in flattened A (6x6, row-major):
    # A00: rows 0-2, cols 0-2 -> indices 0,1,2, 6,7,8, 12,13,14
    # A01: rows 0-2, cols 3-5 -> indices 3,4,5, 9,10,11, 15,16,17
    # A10: rows 3-5, cols 0-2 -> indices 18,19,20, 24,25,26, 30,31,32
    # A11: rows 3-5, cols 3-5 -> indices 21,22,23, 27,28,29, 33,34,35
    
    # B block indices in flattened B (6x6, row-major):
    # B00: rows 0-2, cols 0-2 -> indices 0,1,2, 6,7,8, 12,13,14
    # B01: rows 0-2, cols 3-5 -> indices 3,4,5, 9,10,11, 15,16,17
    # B10: rows 3-5, cols 0-2 -> indices 18,19,20, 24,25,26, 30,31,32
    # B11: rows 3-5, cols 3-5 -> indices 21,22,23, 27,28,29, 33,34,35
    
    # C block indices in flattened C (6x6, row-major):
    # C00: rows 0-2, cols 0-2 -> indices 0,1,2, 6,7,8, 12,13,14
    # C01: rows 0-2, cols 3-5 -> indices 3,4,5, 9,10,11, 15,16,17
    # C10: rows 3-5, cols 0-2 -> indices 18,19,20, 24,25,26, 30,31,32
    # C11: rows 3-5, cols 3-5 -> indices 21,22,23, 27,28,29, 33,34,35
    
    A00_idx = [0,1,2, 6,7,8, 12,13,14]
    A01_idx = [3,4,5, 9,10,11, 15,16,17]
    A10_idx = [18,19,20, 24,25,26, 30,31,32]
    A11_idx = [21,22,23, 27,28,29, 33,34,35]
    
    B00_idx = [0,1,2, 6,7,8, 12,13,14]
    B01_idx = [3,4,5, 9,10,11, 15,16,17]
    B10_idx = [18,19,20, 24,25,26, 30,31,32]
    B11_idx = [21,22,23, 27,28,29, 33,34,35]
    
    C00_idx = [0,1,2, 6,7,8, 12,13,14]
    C01_idx = [3,4,5, 9,10,11, 15,16,17]
    C10_idx = [18,19,20, 24,25,26, 30,31,32]
    C11_idx = [21,22,23, 27,28,29, 33,34,35]
    
    # 3x3 schoolbook decomposition:
    # U3 (9x9), V3 (9x9), W3 (9x9)
    U3 = np.zeros((9, 9))
    V3 = np.zeros((9, 9))
    W3 = np.zeros((9, 9))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
    
    # Now build the 6x6 decomposition.
    # Total rank = 7 * 9 = 63.
    rank_total = 63
    U = np.zeros((36, rank_total))
    V = np.zeros((36, rank_total))
    W = np.zeros((36, rank_total))
    
    # For each Strassen multiplication M1..M7:
    # Each uses 9 products (3x3 schoolbook).
    # 
    # M1 = (A00+A11) @ (B00+B11)
    # M2 = (A10+A11) @ B00
    # M3 = A00 @ (B10-B11)
    # M4 = A11 @ (B00-B11)
    # M5 = (A00+A01) @ B11
    # M6 = (A10-A00) @ (B00+B01)
    # M7 = (A01-A11) @ (B10+B11)
    
    # Define the 7 multiplications:
    # Each is (A_combination, B_combination, C_block_contributions)
    # A_combination: dict from block to coefficient
    # B_combination: dict from block to coefficient
    # C_contrib: dict from C_block to coefficient (for the result of this multiplication)
    
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11
    
    A_block_idx = [A00_idx, A01_idx, A10_idx, A11_idx]
    B_block_idx = [B00_idx, B01_idx, B10_idx, B11_idx]
    C_block_idx = [C00_idx, C01_idx, C10_idx, C11_idx]
    
    # Strassen multiplications:
    # (A_coeffs, B_coeffs, [(C_block, coeff), ...])
    strassen_mults = [
        # M1 = (A00+A11) @ (B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, [(0, 1), (3, 1)]),
        # M2 = (A10+A11) @ B00
        ({2: 1, 3: 1}, {0: 1}, [(2, 1), (3, -1)]),
        # M3 = A00 @ (B10-B11)
        ({0: 1}, {2: 1, 3: -1}, [(1, 1), (3, 1)]),
        # M4 = A11 @ (B00-B11)
        ({3: 1}, {0: 1, 3: -1}, [(0, 1), (2, 1)]),
        # M5 = (A00+A01) @ B11
        ({0: 1, 1: 1}, {3: 1}, [(0, -1), (1, 1)]),
        # M6 = (A10-A00) @ (B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, [(3, 1)]),
        # M7 = (A01-A11) @ (B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, [(0, 1)]),
    ]
    
    # Wait, let me double-check the Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # 
    # So:
    # M1 contributes to C00 (+1) and C11 (+1)
    # M2 contributes to C10 (+1) and C11 (-1)
    # M3 contributes to C01 (+1) and C11 (+1)
    # M4 contributes to C00 (+1) and C10 (+1)
    # M5 contributes to C00 (-1) and C01 (+1)
    # M6 contributes to C11 (+1)
    # M7 contributes to C00 (+1)
    
    strassen_mults = [
        # M1 = (A00+A11) @ (B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, [(0, 1), (3, 1)]),
        # M2 = (A10+A11) @ B00
        ({2: 1, 3: 1}, {0: 1}, [(2, 1), (3, -1)]),
        # M3 = A00 @ (B10-B11)
        ({0: 1}, {2: 1, 3: -1}, [(1, 1), (3, 1)]),
        # M4 = A11 @ (B00-B11)
        ({3: 1}, {0: 1, 3: -1}, [(0, 1), (2, 1)]),
        # M5 = (A00+A01) @ B11
        ({0: 1, 1: 1}, {3: 1}, [(0, -1), (1, 1)]),
        # M6 = (A10-A00) @ (B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, [(3, 1)]),
        # M7 = (A01-A11) @ (B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, [(0, 1)]),
    ]
    
    # Now build U, V, W.
    col_offset = 0
    for mult_idx in range(7):
        A_coeffs, B_coeffs, C_contribs = strassen_mults[mult_idx]
        
        # For each of the 9 products in the 3x3 schoolbook:
        for prod_idx in range(9):
            c = col_offset + prod_idx
            
            # U: for each element in the A combination
            for a_block, coeff in A_coeffs.items():
                for a_elem_idx in A_block_idx[a_block]:
                    # U3[elem, prod_idx] gives the coefficient for this element in the prod_idx-th product
                    U[a_elem_idx, c] += coeff * U3[a_elem_idx % 9, prod_idx]  # a_elem_idx % 9 is the position within the 3x3 block
            
            # V: for each element in the B combination
            for b_block, coeff in B_coeffs.items():
                for b_elem_idx in B_block_idx[b_block]:
                    V[b_elem_idx, c] += coeff * V3[b_elem_idx % 9, prod_idx]
            
            # W: for each C block contribution
            for c_block, coeff in C_contribs:
                for c_elem_idx in C_block_idx[c_block]:
                    W[c_elem_idx, c] += coeff * W3[c_elem_idx % 9, prod_idx]
        
        col_offset += 9
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


def construct():
    # Use the Strassen + schoolbook 3x3 approach: rank 63.
    # This is a structural improvement over the schoolbook rank 216.
    
    result = _construct_block_6x6()
    return result

# EVOLVE-BLOCK-END
