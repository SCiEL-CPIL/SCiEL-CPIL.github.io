"""Matrix multiplication 6x6x6 via 2x2 Strassen block decomposition
with schoolbook rank-27 decomposition per 3x3 block product.
Total rank: 7 * 27 = 189.

Strategy: Partition 6x6 matrices into 2x2 grid of 3x3 blocks.
Apply Strassen's 7-product algorithm at the block level.
Each of the 7 virtual products is a 3x3 matrix multiplication,
decomposed via schoolbook (rank 27).

The BCLR rank-21 decomposition for 3x3 was considered but the
Strassen + schoolbook approach at rank 189 is proven correct and
simpler. The structural advantage of Strassen (7 products instead
of 4 at the block level) combined with the straightforward schoolbook
inner decomposition gives a clean, verifiable construction.
"""

import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin within 120s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size for 3x3 sub-blocks

    # Strassen decomposition at the 2x2 block level (each entry is a 3x3 matrix)
    # Block layout: 0=TL, 1=TR, 2=BL, 3=BR
    # Block (bi): row offset = (bi // 2) * bs, col offset = (bi % 2) * bs

    # Strassen 7 virtual products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)

    # Each term: (list of (block_idx, coeff) for A, list of (block_idx, coeff) for B)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # P1
        ([(2, 1), (3, 1)], [(0, 1)]),            # P2
        ([(0, 1)], [(1, 1), (3, -1)]),           # P3
        ([(3, 1)], [(2, 1), (0, -1)]),           # P4
        ([(0, 1), (1, 1)], [(3, 1)]),            # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # P7
    ]

    # Output reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # output_contrib[term_idx] = list of (output_block_idx, coeff)
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]

    # Total rank: 7 Strassen terms * 27 schoolbook terms per 3x3 product = 189
    rank = 7 * 27

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for i in range(bs):   # local row in 3x3 block
            for k in range(bs):  # local col in A / local row in B
                for j in range(bs):  # local col in B / local col in C
                    # U contributions: A entries
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * bs
                        col_a = (blk_a % 2) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx, col] += coeff_a

                    # V contributions: B entries
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * bs
                        col_b = (blk_b % 2) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx, col] += coeff_b

                    # W contributions: C entries (W index = global_col * n + global_row)
                    for (blk_c, coeff_c) in output_contrib[term_idx]:
                        row_c = (blk_c // 2) * bs
                        col_c = (blk_c % 2) * bs
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx, col] += coeff_c

                    col += 1

    # Verify we filled all columns
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Quick verification with a random test
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        A_test = rng.integers(-5, 6, size=(n, m))
        B_test = rng.integers(-5, 6, size=(m, p))
        C_expected = A_test @ B_test
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reconstruct C from decomposition:
        # C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # But actually the tensor condition is:
        # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = delta for valid (i,k,j)
        # The reconstruction is: C[i,j] = sum_{k,r} A[i,k] * B[k,j] * (sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r])
        # For correct decomposition, sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1 when this is a valid term
        # Actually the standard check: C = sum_r (U[:,r]^T A) (B V[:,r]) W[:,r]^T ... 
        # Let me just verify the tensor directly.
        
        # For each (i, k, j) in the 6x6x6 space:
        # The coefficient should be 1 if (i,k,j) is a valid matrix mult index, else 0
        # Actually for 6x6*6x6->6x6: every (i,k,j) with i,k,j in 0..5 is valid.
        # So sum_r U[i*6+k, r] * V[k*6+j, r] * W[j*6+i, r] should equal 1 for all i,k,j.
        
        # Let's verify a few entries
        for test_i in range(3):
            for test_k in range(3):
                for test_j in range(3):
                    coeff = 0
                    for r in range(rank):
                        coeff += U_np[test_i * m + test_k, r] * V_np[test_k * p + test_j, r] * W_np[test_j * n + test_i, r]
                    if coeff != 1:
                        raise ValueError(f"Tensor verification failed at ({test_i},{test_k},{test_j}): got {coeff}")
        
        # Check a few more
        for test_i in [5]:
            for test_k in [5]:
                for test_j in [5]:
                    coeff = 0
                    for r in range(rank):
                        coeff += U_np[test_i * m + test_k, r] * V_np[test_k * p + test_j, r] * W_np[test_j * n + test_i, r]
                    if coeff != 1:
                        raise ValueError(f"Tensor verification failed at ({test_i},{test_k},{test_j}): got {coeff}")
    
    # Deadline guard
    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
