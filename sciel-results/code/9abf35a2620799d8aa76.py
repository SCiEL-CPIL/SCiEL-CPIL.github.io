"""Find rank-minimal exact tensor decomposition for 2x4 by 4x8 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Strategy: Use tensor product structure.
    # The 2x4x8 multiplication can be viewed as:
    #   C[i, k] = sum_j A[i, j] B[j, k]
    #
    # We can decompose this using the Kronecker product of smaller decompositions.
    # Specifically, we can view the indices as:
    #   i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    #
    # Let's use the fact that 2x4x8 = (2x2x2) ⊗ (1x2x4) or similar factorizations.
    # Actually, a more natural factorization:
    #   Think of A as 2x4, B as 4x8.
    #   We can split the intermediate dimension j into j = 2*j1 + j0 where j1 in {0,1}, j0 in {0,1}
    #   And split the output dimension k into k = 4*k1 + k0 where k1 in {0,1}, k0 in {0,1,2,3}
    #   This doesn't quite work cleanly.
    #
    # Let's try a different approach: use the rank-7 Strassen decomposition for 2x2x2,
    # but apply it in a way that exploits the rectangular structure.
    #
    # Actually, the most promising approach from the inspirations is to use block Strassen.
    # The key insight is that for 2x4x8, we can view this as 8 independent 2x2x2 multiplications
    # (since 4 = 2*2 and 8 = 2*4, we have 2*4=8 blocks of 2x2x2).
    #
    # Wait, let me reconsider. A is 2x4, B is 4x8, C is 2x8.
    # We can split A into two 2x2 blocks: A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # We can split B into four 2x2 blocks: B = [[B_topleft, B_topright], [B_bottomleft, B_bottomright]]
    #   where each is 2x2. So B is 4x8 = [2x4 | 2x4], and each 2x4 is [2x2 | 2x2].
    #   So B has 4 blocks of 2x2.
    # C is 2x8 = [2x4 | 2x4], and each 2x4 is [2x2 | 2x2].
    #   So C has 4 blocks of 2x2.
    #
    # The multiplication is:
    #   C = A * B = [A_left, A_right] * [[B00, B01], [B10, B11]] where each Bij is 2x2
    #   = [A_left*B00 + A_right*B10, A_left*B01 + A_right*B11]
    #
    # So we have 4 output blocks, each computed as a sum of 2 products of 2x2 matrices.
    # Each 2x2x2 product can be done with rank 7 (Strassen).
    # So naively, we'd need 8 * 7 = 56 rank.
    #
    # But we can do better by sharing computations! The key insight is that A_left and A_right
    # are shared across the output blocks, and B00, B01, B10, B11 are also shared.
    #
    # Let's use the tensor product approach more carefully.
    # The multiplication tensor for 2x4x8 can be factored as:
    #   T_{2x4x8} = T_{2x2x2} ⊗ T_{1x2x4}
    # where T_{1x2x4} is the tensor for multiplying a 1x2 matrix by a 2x4 matrix.
    #
    # The rank of T_{1x2x4} is 8 (schoolbook) or possibly less.
    # Actually, for 1x2x4, the rank is at most 8, and likely exactly 8.
    #
    # So rank(T_{2x4x8}) <= rank(T_{2x2x2}) * rank(T_{1x2x4}) = 7 * 8 = 56.
    #
    # This matches the block Strassen approach. Let's implement this carefully.
    #
    # Actually, let me think about this differently. The tensor product of two decompositions
    # gives a decomposition of the tensor product of the tensors. So if we have:
    #   T_A = sum_r u_r ⊗ v_r ⊗ w_r  (rank r decomposition of T_A)
    #   T_B = sum_s u'_s ⊗ v'_s ⊗ w'_s  (rank s decomposition of T_B)
    # Then T_A ⊗ T_B has a rank r*s decomposition:
    #   T_A ⊗ T_B = sum_{r,s} (u_r ⊗ u'_s) ⊗ (v_r ⊗ v'_s) ⊗ (w_r ⊗ w'_s)
    #
    # For our case, we need to identify what T_A and T_B are.
    #
    # Let's use the block structure. We'll implement the rank-56 decomposition first,
    # then try to optimize it.
    
    # Strassen 2x2: rank 7
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22) B11
    # P3 = A11 (B12 - B22)
    # P4 = A22 (B21 - B11)
    # P5 = (A11 + A12) B22
    # P6 = (A21 - A11) (B11 + B12)
    # P7 = (A12 - A22) (B21 + B22)
    
    strassen_products = [
        # P1 = (A11 + A22)(B11 + B22)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A21 + A22) B11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3 = A11 (B12 - B22)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4 = A22 (B21 - B11)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5 = (A11 + A12) B22
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6 = (A21 - A11) (B11 + B12)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A12 - A22) (B21 + B22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For 2x4x8, we have:
    # A is 2x4, split into A_left (2x2) and A_right (2x2)
    # B is 4x8, split into 4 blocks of 2x2: B00, B01, B10, B11
    # C is 2x8, split into 4 blocks of 2x2: C0, C1, C2, C3
    #
    # C0 = A_left * B00 + A_right * B10
    # C1 = A_left * B01 + A_right * B11
    # C2 = A_left * B02 + A_right * B12  -- wait, B only has 4 blocks.
    #
    # Let me re-index. B is 4x8. Split into rows 0-1 and 2-3, and cols 0-1, 2-3, 4-5, 6-7.
    # So B has 2 row-blocks and 4 col-blocks, each 2x2.
    # B = [[B[0,0], B[0,1], B[0,2], B[0,3]],
    #      [B[1,0], B[1,1], B[1,2], B[1,3]]]
    #
    # A = [A_left, A_right] where A_left is cols 0-1, A_right is cols 2-3.
    #
    # C = A * B = [A_left, A_right] * [[B[0,0], B[0,1], B[0,2], B[0,3]],
    #                                   [B[1,0], B[1,1], B[1,2], B[1,3]]]
    #   = [A_left*B[0,0] + A_right*B[1,0],
    #      A_left*B[0,1] + A_right*B[1,1],
    #      A_left*B[0,2] + A_right*B[1,2],
    #      A_left*B[0,3] + A_right*B[1,3]]
    #
    # So C has 4 output blocks, each is a sum of 2 products of 2x2 matrices.
    #
    # Total rank = 4 * 2 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for c_block in range(4):  # 4 output column blocks
        for a_block in range(2):  # 2 A column blocks (left, right)
            # A columns: 2*a_block to 2*a_block+1
            # B rows: 2*a_block to 2*a_block+1
            # B columns: 2*c_block to 2*c_block+1
            # C columns: 2*c_block to 2*c_block+1
            # C rows: 0, 1
            
            a_col0 = 2 * a_block
            b_row0 = 2 * a_block
            b_col0 = 2 * c_block
            c_col0 = 2 * c_block
            
            for u_coeff, v_coeff, w_coeff in strassen_products:
                # Fill U column col
                for (li, lj), coeff in u_coeff.items():
                    gi = li
                    gj = a_col0 + lj
                    U[gi * m + gj][col] += coeff
                # Fill V column col
                for (lj, lk), coeff in v_coeff.items():
                    gj = b_row0 + lj
                    gk = b_col0 + lk
                    V[gj * p + gk][col] += coeff
                # Fill W column col
                for (lk, li), coeff in w_coeff.items():
                    gk = b_col0 + lk
                    gi = li
                    W[gk * n + gi][col] += coeff
                col += 1
        
        if time.time() > deadline:
            break
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(R):
                        total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if total != 1:
                        return False
        return True
    
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
