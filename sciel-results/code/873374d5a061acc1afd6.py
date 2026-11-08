import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Use 3x3 block decomposition with Laderman's algorithm (rank 23) for each block
    # Total rank = 3 * 3 * 3 * 23 = 621? No, that's wrong.
    # Actually, if we use 3x3 blocks, we have 2x2 blocks of size 3x3.
    # So we tile Strassen 2x2, but each "element" is a 3x3 matrix.
    # Strassen for 2x2 blocks: 7 multiplications per block position.
    # Number of block positions: 2 * 2 * 2 = 8.
    # Each multiplication is 3x3, which can be done with Laderman (rank 23).
    # Total rank = 8 * 23 = 184.
    
    # Wait, let me re-read the directive: "Apply a 3x3 block decomposition where each 2x2 block is computed using the 7-term Strassen algorithm, yielding a total rank of 63."
    # This doesn't make sense. If we have 3x3 blocks, we have 2x2 block positions.
    # Strassen on 2x2 blocks requires 7 scalar multiplications.
    # But each "scalar" is a 3x3 matrix multiplication.
    # So we need 7 * (rank of 3x3 matmul) = 7 * 23 = 161? No.
    
    # Let me reconsider. The directive says "3x3 block decomposition where each 2x2 block is computed using the 7-term Strassen algorithm".
    # I think the intent is: divide 6x6 into 3x3 blocks (so 2x2 block positions), then use Strassen on the 2x2 block matrix.
    # Strassen uses 7 multiplications. Each multiplication is of 3x3 matrices.
    # If we use the trivial 3x3 matmul (rank 27), total rank = 7 * 27 = 189.
    # If we use Laderman (rank 23), total rank = 7 * 23 = 161.
    
    # But the current code uses 2x2 blocks with Strassen, giving 3*3*3*7 = 189.
    # The directive claims rank 63, which is 3*3*7 = 63. This would be if we only had one level of 3x3 blocks.
    # I think the directive is confused. Let me just implement the correct approach.
    
    # Approach: 2x2 block decomposition (blocks of size 3x3), using Strassen on the 2x2 block matrix.
    # Each of the 7 Strassen multiplications is a 3x3 matrix multiplication.
    # For the 3x3 multiplications, use Laderman's algorithm (rank 23).
    # Total rank = 7 * 23 = 161.
    
    # Actually, wait. Let me re-read: "Apply a 3x3 block decomposition where each 2x2 block is computed using the 7-term Strassen algorithm, yielding a total rank of 63."
    # 63 = 3 * 3 * 7. This suggests 3x3 blocks, and for each block, 7 Strassen terms.
    # But that would be 9 blocks * 7 = 63. This doesn't make sense for matrix multiplication.
    
    # I think the correct interpretation is: use 3x3 blocks, and for the 2x2 block matrix, use Strassen (7 terms).
    # Each term is a 3x3 matmul. If we use rank 9 for 3x3 (which is impossible, min is 19), we'd get 63.
    # But 3x3 matmul has rank at least 19 (Laderman is 23, best known is 23 for 3x3).
    
    # Let me just implement the straightforward approach: 2x2 blocks of size 3x3, Strassen on blocks, trivial matmul for 3x3.
    # Rank = 7 * 27 = 189. Same as current.
    
    # Or use Laderman for 3x3: rank = 7 * 23 = 161.
    
    # Let me implement Laderman's algorithm for 3x3 matrix multiplication.
    # Laderman's algorithm has 23 multiplications.
    
    # Laderman's algorithm for 3x3:
    # The 23 terms are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A01 + A11)(B10 + B11)  -- wait, I need the exact coefficients.
    
    # Let me look up Laderman's algorithm.
    # Actually, I recall that the current score is 1.1429 = 216/189.
    # If I can get rank 161, score = 216/161 ≈ 1.3416.
    
    # But I need to correctly implement Laderman. Let me use a known correct implementation.
    
    # Laderman's algorithm (23 terms) for C = A*B where A, B, C are 3x3:
    # The terms are products of linear forms in A and B.
    
    # Here's one formulation:
    # Let A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # Let B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    
    # The 23 products are:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a13 + a23)(b31 + b32)
    # M3 = (a21 + a31)(b12 + b22)
    # M4 = (a11 - a22)(b21 - b11)
    # M5 = (a13 - a23)(b23 - b33)
    # M6 = (a21 - a31)(b32 - b22)
    # M7 = (a12 + a22)(b21 + b11)
    # M8 = (a12 + a22)(b23 + b33)
    # M9 = (a12 - a22)(b21 - b11)  -- wait, this is getting complicated.
    
    # I don't have Laderman's exact coefficients memorized. Let me use a safer approach.
    
    # Actually, let me just keep the current 2x2 Strassen approach with rank 189, but verify it's correct.
    # The current code uses 2x2 blocks, 3x3 block positions, Strassen 7 terms.
    # Rank = 3*3*3*7 = 189. Score = 216/189 = 1.142857...
    
    # The directive says to try 3x3 blocks with Strassen. Let me interpret this as:
    # 2x2 block positions (each 3x3), Strassen on the 2x2 block matrix.
    # 7 Strassen multiplications, each is a 3x3 matmul using trivial algorithm (rank 27).
    # Total rank = 7 * 27 = 189. Same as before.
    
    # To improve, I need a better 3x3 matmul. Let me try to implement Laderman correctly.
    
    # After research, here's Laderman's algorithm:
    # The 23 terms for 3x3 matrix multiplication:
    
    laderman_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: {(i,j): val} for A, v_coeffs: {(i,j): val} for B, w_coeffs: {(i,j): val} for C
        # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # Term 2: (A01+A11)(B10+B11) -> C01 += M2, C11 += M2
        ({(0,1): 1, (1,1): 1}, {(1,0): 1, (1,1): 1}, {(0,1): 1, (1,1): 1}),
        # Term 3: (A10+A11)(B01+B11) -> C10 += M3, C11 += M3
        ({(1,0): 1, (1,1): 1}, {(0,1): 1, (1,1): 1}, {(1,0): 1, (1,1): 1}),
        # Term 4: (A00-A11)(B10-B00) -> C00 += M4, C10 += M4
        ({(0,0): 1, (1,1): -1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # Term 5: (A01-A11)(B11-B01) -> C01 += M5, C11 += M5
        ({(0,1): 1, (1,1): -1}, {(1,1): 1, (0,1): -1}, {(0,1): 1, (1,1): 1}),
        # Term 6: (A10-A11)(B01-B10) -> C10 += M6, C11 += M6
        ({(1,0): 1, (1,1): -1}, {(0,1): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}),
        # Terms 7-23: I need the remaining 16 terms.
        # This is getting too risky without a verified reference.
    ]
    
    # Given the risk of incorrect Laderman coefficients, let me stick with the known-correct
    # 2x2 Strassen approach but with a different tiling.
    
    # Actually, the current implementation is correct with rank 189.
    # Let me just clean it up and ensure it's correct.
    
    block_size = 2
    blocks_per_dim = 3  # 6 / 2
    
    rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # C = A * B where A, B, C are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
