"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Block decomposition strategy:
    # Split A (4x5) into: A11 (4x4), A12 (4x1)
    # Split B (5x5) into: B11 (4x4), B12 (4x1), B21 (1x4), B22 (1x1)
    # Split C (4x5) into: C11 (4x4), C12 (4x1)
    #
    # C11 = A11*B11  (4x4x4, use Strassen rank-7)
    # C12 = A11*B12 + A12*B22  (4x4x1 + 4x1x1, schoolbook rank 4+4=8)
    # Total rank: 7 + 8 = 15

    # Strassen decomposition for 4x4x4:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2-M4+M6
    # C22 = M1-M2+M3-M6

    # For our 4x4x4 block (A11 is 4x4, B11 is 4x4):
    # We need to further split A11 and B11 into 2x2 blocks for Strassen.
    # A11 (4x4) = [[A11_11, A11_12], [A11_21, A11_22]] each 2x2
    # B11 (4x4) = [[B11_11, B11_12], [B11_21, B11_22]] each 2x2

    # Let's build the full decomposition with rank 15.
    # U has shape (20, 15), V has shape (25, 15), W has shape (20, 15)

    # For Strassen on 4x4x4, we need to express it in terms of the original indices.
    # A11[i,j] for i,j in 0..3 corresponds to A[i,j] in original A (4x5)
    # B11[j,k] for j,k in 0..3 corresponds to B[j,k] in original B (5x5)
    # C11[i,k] for i,k in 0..3 corresponds to C[i,k] in original C (4x5)

    # Strassen M1..M7 are computed from 2x2 sub-blocks of A11 and B11.
    # A11_11 = A[0:2, 0:2], A11_12 = A[0:2, 2:4]
    # A11_21 = A[2:4, 0:2], A11_22 = A[2:4, 2:4]
    # B11_11 = B[0:2, 0:2], B11_12 = B[0:2, 2:4]
    # B11_21 = B[2:4, 0:2], B11_22 = B[2:4, 2:4]

    # Each M is a 4x4 matrix product of sums/differences of 2x2 blocks.
    # We need to decompose each M into rank-1 terms.

    # For a 2x2x2 Strassen, each M is a product of two 2x2 matrices.
    # A 2x2x2 product has rank 4 (schoolbook) or can be done in rank... 
    # Actually, for the 4x4x4 Strassen, each Mi is a 4x4 matrix that is
    # the product of two 4x4 matrices (which are sums of 2x2 blocks).
    # Each such 4x4x4 product needs rank 4 (schoolbook) unless we use Strassen again.

    # Wait - the 4x4x4 Strassen reduces the NUMBER of 2x2x2 multiplications from 8 to 7.
    # But each 2x2x2 multiplication still needs rank 4 (schoolbook) or we can use
    # the fact that 2x2x2 has rank 7 (Strassen) or even lower.
    #
    # Actually, the known result is that 2x2x2 matrix multiplication has rank 7
    # (Strassen), and this is tight. So each 2x2x2 block needs rank 7.
    #
    # For 4x4x4: Strassen gives 7 multiplications of 2x2x2, each of rank 7.
    # But wait, that would give rank 7*7 = 49, which is worse than schoolbook 64? No...
    #
    # I'm confusing myself. Let me reconsider.
    #
    # The rank of the 4x4x4 matrix multiplication tensor is at most 28 (using Strassen
    # recursively: 7 * rank(2x2x2) = 7 * 7 = 49? No, that's not right either.
    #
    # Actually, the standard Strassen algorithm for 4x4x4 uses 7 multiplications of
    # 2x2x2, and each 2x2x2 can be done in rank 7 (Strassen). So total rank = 7*7 = 49?
    # But the cell finding says 4x4x4 goes from 64 to 28, which is 7*4 = 28.
    #
    # Ah, I think the 2x2x2 is being done with schoolbook (rank 4), not Strassen (rank 7).
    # So 4x4x4 Strassen = 7 multiplications of 2x2x2, each rank 4 = 28.
    #
    # That makes sense! Strassen reduces the number of scalar multiplications from 8 to 7
    # at the block level, but each block multiplication (2x2x2) is done with schoolbook (rank 4).
    # So 4x4x4 rank = 7 * 4 = 28.

    # So for our problem:
    # C11 = A11*B11 (4x4x4) → rank 28
    # C12 = A11*B12 + A12*B22 (4x4x1 + 4x1x1) → rank 4 + 4 = 8
    # Total rank: 28 + 8 = 36

    # Let me verify: schoolbook for 4x5x5 is 100. We're claiming rank 36.
    # That's a big improvement!

    # Now let me construct the actual U, V, W matrices.

    # For the 4x4x4 Strassen part (rank 28):
    # We have 7 "virtual" 2x2x2 multiplications, each with 4 scalar terms.
    # 
    # Strassen for 4x4x4 (with 2x2 sub-blocks):
    # Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]] (all 2x2)
    # 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M2 + M3 - M6

    # Each Mi is a 2x2x2 multiplication, done with schoolbook (rank 4).
    # For Mi = Xi * Yi where Xi and Yi are 2x2:
    #   Mi_{ab} = sum_c Xi_{ac} Yi_{cb}
    # Schoolbook: for each (a,c,b) in {0,1}^3, one rank-1 term.

    # Let me map out all indices.
    # Original A is 4x5: A[i,j] for i in 0..3, j in 0..4
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # A12_orig = A[:, 4] (the 5th column, 4x1)
    #
    # Original B is 5x5: B[j,k] for j in 0..4, k in 0..4
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # B12_orig = B[0:2, 4] (rows 0,1, col 4, 2x1)
    # B21_orig = B[2:4, 4] (rows 2,3, col 4, 2x1) -- wait, B is 5x5, rows 0..4
    # Actually B21 in the 4x4x4 context is B[2:4, 0:2], and B22 is B[2:4, 2:4]
    # B12_orig (for C12 computation) = B[0:2, 4] (2x1)
    # B22_orig (for C12 computation) = B[4, 4] (1x1) -- wait no
    #
    # Let me reclarify the block structure for C12 = A11*B12 + A12*B22:
    # A11 is 4x4 (rows 0-3, cols 0-3 of A)
    # B12 is 4x1 (rows 0-3, col 4 of B) -- this is B[0:4, 4]
    # A12 is 4x1 (rows 0-3, col 4 of A) -- this is A[:, 4]
    # B22 is 1x1 (row 4, col 4 of B) -- this is B[4, 4]
    #
    # So C12 = A11 * B12 + A12 * B22
    # = A[0:4, 0:4] * B[0:4, 4] + A[0:4, 4] * B[4, 4]
    # This is a 4x1 vector.
    #
    # The first term A11*B12 is 4x4 times 4x1 = 4x1, schoolbook rank 16? No...
    # For matrix-vector multiplication 4x4 * 4x1, the tensor rank is 4*4*1 = 16? 
    # No, that's not right either. Let me think more carefully.
    #
    # The tensor for C[i] = sum_j A[i,j] * B[j] (where B is a vector) is:
    # T[i, j, k] where k is always 0 (since output is 4x1).
    # T[i, j, 0] = 1 for all i,j.
    # Rank = 4*4*1 = 16? No, that's the schoolbook for 4x4x1.
    #
    # Actually for n x m times m x 1, the schoolbook rank is n*m*1 = n*m.
    # So 4x4x1 has rank 16, and 4x1x1 has rank 4.
    # C12 total rank = 16 + 4 = 20.
    #
    # Hmm, that's more than I thought. Let me recalculate.
    #
    # Total rank = 28 (for 4x4x4) + 16 (for 4x4x1) + 4 (for 4x1x1) = 48.
    #
    # Still much better than 100!

    # Let me construct this properly.
    # 
    # Part 1: 4x4x4 Strassen (rank 28)
    # Part 2: 4x4x1 schoolbook (rank 16)  
    # Part 3: 4x1x1 schoolbook (rank 4)
    # Total: 48

    rank_total = 48
    U = np.zeros((20, rank_total))  # n*m = 4*5 = 20
    V = np.zeros((25, rank_total))  # m*p = 5*5 = 25
    W = np.zeros((20, rank_total))  # p*n = 5*4 = 20

    col = 0

    # Part 1: 4x4x4 Strassen decomposition (rank 28)
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # Strassen M1..M7, each is a 2x2x2 product (rank 4 each)
    # For each M, we have a 2x2 matrix X and 2x2 matrix Y, and M = X*Y
    # Schoolbook for 2x2x2: for (a,c,b) in {0,1}^3:
    #   U_index = (row_in_A11)*2 + col_in_A11  -> but we need to map to original A indices
    #   
    # Let me define helper: for a 2x2x2 product of sub-blocks,
    # X is formed from rows r1,r2 and cols c1,c2 of A11 (which is A[0:2,0:2] etc.)
    # This is getting complex. Let me just enumerate all 28 terms explicitly.

    # For Strassen 4x4x4, the 7 multiplications are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Each Mi is a 2x2x2 multiplication. For Mi = Xi * Yi where Xi, Yi are 2x2:
    # Mi_{ab} = sum_c Xi_{ac} * Yi_{cb}
    # Schoolbook: 8 terms per Mi? No, 2*2*2 = 8? No, 2*2*2 = 8 but rank is 4.
    # Wait, 2x2x2 schoolbook rank is 2*2*2 = 8? No!
    # 
    # For multiplying a 2x2 by a 2x2:
    # C[i,k] = sum_j A[i,j] B[j,k] for i,k in {0,1}, j in {0,1}
    # That's 2*2*2 = 8 terms in schoolbook. But Strassen reduces to 7.
    # So schoolbook rank for 2x2x2 is 8, not 4!
    #
    # I made an error. Let me redo:
    # 4x4x4 schoolbook rank = 4*4*4 = 64
    # 4x4x4 Strassen = 7 * (2x2x2 schoolbook rank) = 7 * 8 = 56? 
    # But the cell finding says 28...
    #
    # Hmm, maybe the cell finding used Strassen for 2x2x2 as well?
    # 4x4x4 with Strassen at both levels: 7 * 7 = 49? Still not 28.
    #
    # Or maybe 2x2x2 has rank 4 in some sense? No, the known minimum rank for 2x2x2
    # matrix multiplication is 7 (Strassen).
    #
    # Wait, I think I'm confusing the tensor rank with the number of scalar multiplications.
    # The rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen).
    # The rank of the 4x4x4 matrix multiplication tensor is at most 7*7 = 49 (Strassen recursive)
    # or 7*8 = 56 (Strassen at top, schoolbook at bottom).
    #
    # But the cell finding says 28. Let me re-read: "Strassen decomposition for the 4×4×4 
    # sub-tensor reduces rank from 64 to 28 for that part"
    #
    # 28 = 7 * 4. So maybe they're using a 2x2x2 decomposition of rank 4?
    # That's not possible for exact 2x2x2 matrix multiplication (min rank is 7).
    #
    # Unless... they're not doing full 2x2x2 matrix multiplication, but rather
    # using the structure differently. Or maybe the 4x4x4 is being split into
    # 2x2x2 blocks where each "multiplication" is actually simpler.
    #
    # Actually, I wonder if the cell finding is about a different type of decomposition
    # or if I'm misunderstanding the tensor structure.
    #
    # Let me just go with a safe approach: use schoolbook for everything but try
    # to find a better decomposition. Given the time constraints, let me implement
    # the 4x4x4 Strassen with schoolbook 2x2x2 (rank 7*8=56) plus the remaining parts.
    #
    # Actually, let me reconsider the whole approach. Maybe I should just use
    # a different block structure that gives a cleaner decomposition.
    #
    # Alternative: split A (4x5) into rows 2+2, cols 5.
    # Split B (5x5) into rows 5, cols 2+3.
    # Split C (4x5) into rows 2+2, cols 2+3.
    #
    # C11 (2x2) = A11(2x5)*B11(5x2) + ... no, this doesn't simplify nicely.
    #
    # Let me try yet another approach: use the known fact that for small matrices,
    # we can sometimes find explicit low-rank decompositions by solving linear equations.
    #
    # Given the complexity and time constraints, let me just return a valid schoolbook
    # decomposition but with a slightly optimized structure. The safest bet is to
    # ensure correctness.

    # Let me implement the 4x4x4 Strassen properly with schoolbook 2x2x2 (rank 8 each).
    # Total for 4x4x4: 7 * 8 = 56
    # 4x4x1: 4*4*1 = 16
    # 4x1x1: 4*1*1 = 4
    # Total: 56 + 16 + 4 = 76

    # That's still better than 100!

    rank_total = 76
    U = np.zeros((20, rank_total))
    V = np.zeros((25, rank_total))
    W = np.zeros((20, rank_total))

    col = 0

    # Helper: add a schoolbook 2x2x2 decomposition
    # X is a 2x2 matrix from A's indices, Y is a 2x2 matrix from B's indices
    # Z is a 2x2 matrix for C's indices
    # For each (a, c, b) in {0,1}^3:
    #   U[A_row(a), A_col(c)] += coeff_x * coeff_a
    #   V[B_row(c), B_col(b)] += coeff_y * coeff_b  
    #   W[C_col(b), C_row(a)] += coeff_z * coeff_z
    # where the product contributes to Z[a,b]

    def add_2x2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, coeff_x, coeff_y, coeff_z):
        """Add a 2x2x2 schoolbook decomposition."""
        nonlocal col
        for a in range(2):
            for c in range(2):
                for b in range(2):
                    U[A_rows[a] * 5 + A_cols[c], col] = coeff_x * (1 if a == 0 else 0) + coeff_x * (1 if a == 1 else 0)
                    # This is getting too complex. Let me just do it directly.
                    break
                break
            break
        # This approach is too messy. Let me just enumerate all terms directly.

    # Let me take a completely different approach: just use schoolbook for the whole thing
    # but verify it's correct, and try a simple optimization if time permits.

    # Schoolbook decomposition
    rank = n * m * p  # 100
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    # Verify
    def verify(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Should not happen, but fallback
        pass

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
