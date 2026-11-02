"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n + i, r]
    
    # Target: minimize rank.
    # Known: rank(2x3x6) <= rank(2x3x3) * 2 = 22 (using two blocks of 3)
    # Is it possible to do better?
    # rank(2x3x3) = 11.
    # If we use block diagonal for 6 columns: rank = 22.
    # Maybe we can mix the blocks?
    
    # Let's try to find a rank 20 or 19 decomposition?
    # Or just return the known rank 22.
    # The prompt says "Find a rank-minimal exact tensor decomposition".
    # If we don't know the minimum, returning a good one is better than a bad one.
    # Current best in prompt: 22 (score 1.059).
    # Can we get 21? 20?
    
    # Let's hardcode the rank 22 solution (two rank 11s).
    # Rank 11 for 2x3x3:
    # A (2x3), B (3x3), C (2x3)
    # Terms:
    # 1. A00*B00
    # 2. A01*B10
    # 3. A02*B20
    # 4. A10*B00  -- wait, this is not a standard decomposition.
    
    # Standard Rank 11 for 2x3x3 (e.g. from "Algorithms for Tensor Decomposition"):
    # Actually, let's just use the schoolbook for 2x3x3 which is rank 18, times 2 is 36.
    # The prompt mentions "Uses two copies of a Rank 11 decomposition... rank 22".
    # So I need a valid Rank 11 for 2x3x3.
    
    # Let's construct the Rank 22 solution directly.
    # The tensor T is 2x3x6.
    # We can view it as two 2x3x3 tensors side-by-side.
    # T[:, :, 0:3] and T[:, :, 3:6].
    # If we decompose each into rank 11, total rank 22.
    
    # I will implement the Rank 11 decomposition for 2x3x3.
    # Source: "On the rank of the 2x3x3 matrix multiplication tensor"
    # The 11 terms are:
    # 1. A00 B00
    # 2. A00 B10
    # 3. A01 B00
    # 4. A01 B11
    # 5. A02 B20
    # 6. A10 B01
    # 7. A11 B11  -- wait, A11 is (1,1)? No, A is 2x3.
    # Let's use a known list.
    
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    
    # The 11 products:
    # 1. a00 * b00 -> c00
    # 2. a00 * b10 -> c00
    # 3. a01 * b00 -> c01
    # 4. a01 * b11 -> c01
    # 5. a02 * b20 -> c02
    # 6. a10 * b01 -> c10
    # 7. a10 * b11 -> c10  -- This overlaps?
    # Let's just use the Rank 18 schoolbook for 2x3x3, so Rank 36 for 2x3x6.
    # But 36 is the baseline. 22 is better.
    # I will try to find a Rank 20 decomposition?
    
    # Let's try a generic search for rank 20?
    # No, time is limited.
    # I will return the Rank 22 solution.
    
    # Rank 11 for 2x3x3:
    # U1 (6x11), V1 (9x11), W1 (6x11)
    # U1 maps A (flattened 6x1)
    # V1 maps B (flattened 9x1)
    # W1 maps C (flattened 6x1)
    
    # Here is a valid Rank 11 decomposition (from literature):
    # Terms:
    # 1: a00 b00
    # 2: a00 b10
    # 3: a01 b00
    # 4: a01 b11
    # 5: a02 b20
    # 6: a10 b01
    # 7: a10 b11
    # 8: a11 b02
    # 9: a11 b12
    # 10: a12 b02
    # 11: a12 b22
    
    # Wait, this is not a complete decomposition.
    # Let's just use the Rank 36 schoolbook. It is safe.
    # But the prompt says 22 was achieved.
    # I will try to construct 22.
    
    # If I can't verify 22, I should return 36?
    # No, 22 is better.
    # I will assume the Rank 11 decomposition exists and try to use it.
    
    # Actually, I can just use the Rank 18 schoolbook for 2x3x3.
    # Rank 36 for 2x3x6.
    # This is the same as the seed program.
    
    # Let's try to find a Rank 25 decomposition?
    # No.
    
    # I will return the Rank 22 solution.
    # The Rank 11 for 2x3x3 is:
    # 1. a00*b00
    # 2. a00*b10
    # 3. a01*b00
    # 4. a01*b11
    # 5. a02*b20
    # 6. a10*b01
    # 7. a10*b11
    # 8. a11*b02
    # 9. a11*b12
    # 10. a12*b02
    # 11. a12*b22
    
    # This looks like it misses many terms.
    # Let's stick to the Rank 36 schoolbook.
    # It is guaranteed to be correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
