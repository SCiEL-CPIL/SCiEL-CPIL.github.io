"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import qr

_start_time = time.time()
_DEADLINE = _start_time + 145  # Safety margin before 150s kill


def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # Start with the known rank-53 Strassen+schoolbook decomposition
    # Then attempt to reduce rank by finding cross-block linear dependencies

    # Build the rank-53 decomposition
    rank = 53
    U = np.zeros((n * m, rank), dtype=float)  # 4 x 53
    V = np.zeros((m * p, rank), dtype=float)  # 30 x 53
    W = np.zeros((p * n, rank), dtype=float)  # 30 x 53

    # A indices in U: 0: A00, 1: A01, 2: A10, 3: A11
    # B indices in V: j*p + k
    # C indices in W: k*n + i

    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6

        # P1 = (A00+A11)(B0k0+B1k1)
        U[0, r1] = 1
        U[3, r1] = 1
        V[0 * p + k0, r1] = 1
        V[1 * p + k1, r1] = 1

        # P2 = (A10+A11) B0k0
        U[2, r2] = 1
        U[3, r2] = 1
        V[0 * p + k0, r2] = 1

        # P3 = A00 (B0k1 - B1k1)
        U[0, r3] = 1
        V[0 * p + k1, r3] = 1
        V[1 * p + k1, r3] = -1

        # P4 = A11 (B1k0 - B0k0)
        U[3, r4] = 1
        V[1 * p + k0, r4] = 1
        V[0 * p + k0, r4] = -1

        # P5 = (A00+A01) B1k1
        U[0, r5] = 1
        U[1, r5] = 1
        V[1 * p + k1, r5] = 1

        # P6 = (A10-A00) (B0k0+B0k1)
        U[2, r6] = 1
        U[0, r6] = -1
        V[0 * p + k0, r6] = 1
        V[0 * p + k1, r6] = 1

        # P7 = (A01-A11) (B1k0+B1k1)
        U[1, r7] = 1
        U[3, r7] = -1
        V[1 * p + k0, r7] = 1
        V[1 * p + k1, r7] = 1

        # W reconstruction
        # C0k0 = P1 + P4 - P5 + P7
        W[2 * k0, r1] = 1
        W[2 * k0, r4] = 1
        W[2 * k0, r5] = -1
        W[2 * k0, r7] = 1

        # C1k0 = P2 + P4
        W[2 * k0 + 1, r2] = 1
        W[2 * k0 + 1, r4] = 1

        # C0k1 = P3 + P5
        W[2 * k1, r3] = 1
        W[2 * k1, r5] = 1

        # C1k1 = P1 - P2 + P3 + P6
        W[2 * k1 + 1, r1] = 1
        W[2 * k1 + 1, r2] = -1
        W[2 * k1 + 1, r3] = 1
        W[2 * k1 + 1, r6] = 1

    # Add 7 Strassen blocks for pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)

    # Add trivial block for column 14
    base_r = 49
    U[0, base_r] = 1
    V[0 * p + 14, base_r] = 1
    W[2 * 14, base_r] = 1

    U[1, base_r + 1] = 1
    V[1 * p + 14, base_r + 1] = 1
    W[2 * 14, base_r + 1] = 1

    U[2, base_r + 2] = 1
    V[0 * p + 14, base_r + 2] = 1
    W[2 * 14 + 1, base_r + 2] = 1

    U[3, base_r + 3] = 1
    V[1 * p + 14, base_r + 3] = 1
    W[2 * 14 + 1, base_r + 3] = 1

    # Verify the decomposition is correct
    # The tensor identity: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
    # where T[i,j,k] = 1 if (i,j,k) corresponds to a valid product term

    # Now attempt to reduce rank by finding linear dependencies
    # Strategy: treat the 53 rank-1 terms as vectors in a high-dimensional space
    # and look for subsets that can be replaced by fewer terms

    # Build the constraint matrix for the tensor identity
    # Each rank-1 term r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We need sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T (the multiplication tensor)

    # The tensor T has entries T[i,j,k] = delta(i,j,k) for valid products
    # For 2x2x15: T[A_i, B_j, C_k] where A_i is row i of A, B_j is column j of B
    # Actually the tensor is: T[a,b,c] = 1 if c = a*b (matrix mult)

    # Let's think differently. The multiplication tensor for (n,m,p) has:
    # T[a,b,c] = 1 if a is row of A, b is col of B, c is entry of C = A*B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 (the tensor entry at position (i,j,k))

    # The rank-1 decomposition: sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]

    # Let's build the full constraint system
    # Number of tensor entries: n*m*p = 2*2*15 = 60
    # For rank R, we have 3*R*4 + 3*R*30 + 3*R*30 = 3R(4+30+30) = 186R variables
    # But the system is nonlinear (bilinear in U,V,W)

    # Alternative approach: fix two of the three matrices and solve for the third
    # Or: use the fact that if we have a valid decomposition of rank R,
    # we can try to find a decomposition of rank R-1 by checking if one term
    # is in the span of the others.

    # Let's try a more direct approach:
    # For each term r, check if it can be expressed as a linear combination
    # of the other terms. If so, we can eliminate it.

    # The condition for term r to be redundant:
    # U[:,r] ⊗ V[:,r] ⊗ W[:,r] = sum_{s≠r} α_s * U[:,s] ⊗ V[:,s] ⊗ W[:,s]

    # This is a system of 60 equations in 52 unknowns (α_s for s≠r)
    # But it's nonlinear in α_s because the tensor products don't factor nicely.

    # Better approach: use the block structure.
    # The 7 Strassen blocks operate on disjoint pairs of columns.
    # The only cross-block interactions are through the shared A matrix.
    # But since each block uses all 4 entries of A, there's no obvious
    # cross-block cancellation.

    # Let's try a randomized approach: perturb the decomposition slightly
    # and see if we can find a lower-rank equivalent.

    # Actually, let's try a different strategy: use the schoolbook method
    # for some blocks and Strassen for others, and see if mixing helps.

    # Or: try to use a rank-6 decomposition for 2x2x2 instead of rank-7.
    # But rank 6 is not known for 2x2x2 (Strassen's rank 7 is optimal).

    # Let's try: use Strassen for 6 pairs (rank 42) + a special 2x2x3 block (rank 10)
    # 6*7 + 10 = 52

    # For 2x2x3, the tensor has 12 entries. Can we decompose it in rank 10?
    # Schoolbook would be 4*3=12. Strassen on 2x2x2 gives 7, plus 4 for the
    # third column = 11. Can we do 10?

    # Let's try to construct a rank-10 decomposition for 2x2x3.
    # T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    # C01 = A00*B01 + A01*B11
    # C11 = A10*B01 + A11*B11
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12

    # Strassen on (k0,k1): 7 terms
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1+P4-P5+P7
    # C10 = P2+P4
    # C01 = P3+P5
    # C11 = P1-P2+P3+P6

    # For k2: schoolbook 4 terms
    # Q1 = A00*B02
    # Q2 = A01*B12
    # Q3 = A10*B02
    # Q4 = A11*B12
    # C02 = Q1+Q2
    # C12 = Q3+Q4

    # Total: 7+4 = 11. Can we do 10?

    # Try to merge: can any of the Q terms be expressed using P terms?
    # Q1 = A00*B02. The P terms involve B00, B01, B10, B11 but not B02, B12.
    # So no direct merge.

    # What if we use a different decomposition for 2x2x3?
    # Let's try rank 10 with a search.

    # Actually, let's just try the 52-rank construction and see if it's valid.
    # 6 Strassen blocks (columns 0-11) + 2x2x3 for columns 12,13,14

    # But 2x2x3 rank 10 is not proven. Let's use rank 11 for safety.
    # 6*7 + 11 = 53. Same as before.

    # Try: 5 Strassen blocks (35) + 2x2x5 block
    # 2x2x5 schoolbook = 20. Total = 55. Worse.

    # Try: 7 Strassen blocks (49) + 1 column (4) = 53. Current best.

    # Let's try to verify if rank 52 is possible with a different strategy.
    # Use Strassen for all 15 columns by padding with a zero column.
    # 15 = 7*2 + 1. Pad to 16 = 8*2. 8 Strassen blocks = 56. Worse.

    # Or: 15 = 3*5. Use 2x2x5 blocks. 2x2x5 rank?
    # Schoolbook: 20. Strassen on 2x2x2 (7) + 2x2x3 (11) = 18.
    # 3*18 = 54. Worse than 53.

    # Let's stick with 53 and return it.

    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
