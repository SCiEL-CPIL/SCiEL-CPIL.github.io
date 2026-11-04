import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    # Patterns are (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(row, col): val} for A
    # V_coeffs: dict {(row, col): val} for B
    # W_coeffs: dict {(row, col): val} for C contribution
    laderman_23_patterns = [
        # 1. A00 * (B00 + B11)
        ({(0,0): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}), # This W is wrong, it's a scalar multiplication result added to specific C entries?
        # Wait, the pattern representation in the seed code was:
        # ({(r,c): val}, {(r,c): val}, {(r,c): val})
        # mu = sum A[r,c]*val
        # mv = sum B[r,c]*val
        # C[r,c] += val * mu * mv
        # This implies W_coeffs maps the RESULT of the scalar multiplication to specific entries of C.
        # For a $3 \times 3$ multiplication, each scalar multiplication contributes to multiple entries of C.
        # So W_coeffs should have multiple entries.
    ]
    
    # Correct Laderman Patterns for 3x3
    # M1 = A00 * (B00 + B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
    # C = AB.
    # C00 = A00B00 + A01B10 + A02B20
    # The Laderman algorithm computes linear combinations.
    # Let's use the verified patterns from a known source or derive them carefully.
    # Actually, I will use the patterns from the seed code but CORRECTED if I can, or just use a known good set.
    # Since I don't have a verified set in memory, I will try to construct the 161 rank solution using the assumption that Laderman is rank 23.
    # If I cannot verify Laderman, I will stick to 189.
    
    # Let's try to find a simpler improvement.
    # What if I use the "Kronecker Product" approach mentioned in the directive?
    # "Use the known rank-4 decomposition for 2x2 matrix multiplication... Kronecker product with standard rank-27 schoolbook 3x3 decomposition"
    # Rank 4 for 2x2? The minimum rank for 2x2 is 7. Rank 4 is impossible for exact multiplication.
    # Maybe it means rank 4 for something else? Or maybe it's a trick?
    # No, rank of 2x2 matrix multiplication is 7.
    # So that hint might be misleading or referring to a different problem (e.g. convolution).
    
    # Let's stick to the Strassen on 3x3 blocks with Laderman.
    # I will implement the verification. If it fails, I return 189.
    
    def build_strassen_6x6():
        """Build 6x6x6 using Strassen on 3x3 blocks. Rank = 7*23 = 161 if Laderman, else 7*27 = 189 (if schoolbook 3x3)."""
        # We will use Laderman for the 3x3 blocks.
        # If Laderman is not available/valid, we use schoolbook for 3x3 blocks?
        # Schoolbook 3x3 is rank 27.
        # Strassen on 3x3 blocks with schoolbook 3x3: Rank = 7 * 27 = 189.
        # This is the same as Strassen on 2x2 blocks with schoolbook 2x2?
        # Strassen on 2x2 blocks: 27 blocks * 7 = 189.
        # So 189 is the baseline.
        
        # To get 161, we need Laderman (23) for 3x3.
        
        # I will define the Laderman patterns here.
        # Reference: Laderman, L. (1976). "A scheme for multiplying 3x3 matrices using 23 multiplications."
        
        # The patterns are:
        # M1 = A11 * (B11 + B22)
        # M2 = (A11 + A22) * B22
        # M3 = A22 * (B11 + B12)
        # M4 = (A21 + A22) * (B11 + B22)
        # M5 = A11 * (B21 - B22)
        # M6 = (A11 + A12) * B21
        # M7 = A22 * (B12 - B11)
        # M8 = (A21 + A11) * B11
        # M9 = (A12 + A22) * (B12 + B22)
        # M10 = A12 * (B11 - B22)
        # M11 = (A11 + A12) * B11
        # M12 = A21 * (B22 - B11)
        # M13 = (A21 + A22) * B12
        # M14 = A11 * (B12 - B21)
        # M15 = (A11 + A21) * B12
        # M16 = A22 * (B21 - B12)
        # M17 = (A21 + A12) * B11
        # M18 = (A11 + A12 + A21 + A22) * (B11 + B22)
        # M19 = (A11 + A12 - A21 - A22) * (B11 - B22)
        # M20 = (A11 - A12 + A21 - A22) * (B11 + B22)
        # M21 = (A11 - A12 - A21 + A22) * (B11 - B22)
        # M22 = (A11 + A21) * (B12 + B22)
        # M23 = (A12 + A22) * (B11 + B21)
        
        # The reconstruction of C:
        # C11 = M1 - M5 + M12 - M16 + M19 - M20 + M21
        # C12 = M3 - M7 + M14 - M15 + M18 - M22
        # C21 = M4 - M8 + M13 - M14 + M20 - M21
        # C22 = M2 - M6 + M11 - M17 + M19 - M21
        # Wait, the reconstruction coefficients are complex.
        # I will not hardcode the reconstruction if I can't verify it.
        
        # Instead, I will use the Strassen 2x2 block approach which is known to work (Rank 189).
        # And I will try to add a "Laderman" attempt that is verified.
        
        # Since I cannot guarantee the Laderman patterns are correct without a long verification loop,
        # and the previous code's Laderman failed, I will stick to the safe 189 rank solution 
        # BUT I will try to optimize the 189 solution or find a simple rank reduction.
        
        # Actually, let's look at the score. 1.1429 is 189.
        # If I can get 184 (Laderman on 3x3 blocks directly, not Strassen), that's 1.17.
        # If I can get 161 (Strassen on 3x3 blocks with Laderman), that's 1.34.
        
        # I will implement the 184 solution (4 Laderman 3x3 multiplications) as a fallback if 161 is too risky.
        # But 161 is much better.
        
        # Let's try to implement the 161 solution with a robust verification step.
        
        pass

    def build_strassen_2x2_blocks():
        """Standard Strassen on 2x2 blocks. Rank 189."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val
                        col_idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # I will return the 189 solution as it is safe.
    # If I have time, I might try to implement the 161 solution, but it's risky.
    # Given the constraints, I'll stick to the known working 189.
    
    sol, rank = build_strassen_2x2_blocks()
    return sol
# EVOLVE-BLOCK-END
