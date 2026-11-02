"""Deterministic algebraic construction for 4x4 by 4x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Strategy: Use Kronecker product structure
    # Split 4x4 into 2x2 blocks of 2x2 submatrices
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bij is 2x3
    # C = [[C11, C12, C13], [C21, C22, C23]] where each Cij is 2x2
    
    # For the 2x2 by 2x3 multiplication, we can use a rank-7 algorithm
    # Actually, let's use Strassen for 2x2x2 part and schoolbook for the extra column
    
    # Better approach: Use the known fact that 4x4 by 4x4 can be done with rank 49
    # (Strassen's algorithm: 7 multiplications for 2x2, so 7^2 = 49 for 4x4)
    # Then extend to 4x4 by 4x7 by handling the extra 3 columns
    
    # Let's build the 4x4 by 4x4 Strassen decomposition first
    # Strassen's 7 products for 2x2 matrices:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B22)
    # P4 = A22(B12 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # For 4x4, we apply this at the block level where each "scalar" is a 2x2 matrix
    # This gives 7 block multiplications, each of which is 2x2 by 2x2
    # Each 2x2 by 2x2 multiplication can be done with 4 schoolbook products
    # Total: 7 * 4 = 28 products for 4x4 by 4x4
    
    # Wait, that's not right. Strassen replaces 8 scalar mults with 7.
    # For 4x4 blocks: each block is 2x2, so each "block multiplication" is 2x2 by 2x2 = 4 scalar mults
    # Strassen: 7 block-mults * 4 scalar-mults per block-mult = 28 scalar mults
    # Schoolbook: 16 scalar mults per block * 4 blocks = 64 scalar mults
    # So Strassen saves: 64 - 28 = 36 mults, rank goes from 64 to 28? No...
    
    # Actually, the rank of the tensor is the number of scalar multiplications
    # Schoolbook 4x4 by 4x4: 4*4*4 = 64
    # Strassen 4x4 by 4x4: 7 (block mults) * 4 (scalar mults per 2x2x2) = 28
    
    # Now for 4x4 by 4x7:
    # Split B into B_left (4x4) and B_right (4x3)
    # C = A * B_left + A * B_right
    # A * B_left: rank 28 (Strassen)
    # A * B_right: schoolbook 4*4*3 = 48
    # Total: 28 + 48 = 76
    
    # Can we do better? Maybe use Strassen-like for 4x4 by 4x3?
    # 4x4 by 4x3: schoolbook = 48
    # We can split this into 4x4 by 4x2 and 4x4 by 4x1
    # 4x4 by 4x1: each row of C is dot product of row of A with column of B, so 4*4 = 16
    # 4x4 by 4x2: can we use Strassen? 
    # Split 4x2 into two 4x1 matrices, use Strassen for each: 2 * 28 = 56? No, that's worse.
    
    # Let's stick with 76 for now and see if we can improve
    
    # Actually, let's reconsider. The known best for 4x4x4 is 49 (Laderman's algorithm)
    # But Laderman is complex. Let's use Strassen: rank 49 is achievable but let's verify.
    
    # For simplicity and correctness, let's use the Kronecker product approach:
    # 4x4 by 4x7 = (2x2 by 2x3) ⊗ (2x2 by 2x7/2)? No, that doesn't work cleanly.
    
    # Let's just build the schoolbook and then try to reduce using Strassen for the 4x4 part
    
    # Build Strassen for 4x4 by 4x4:
    # We need to express this in terms of the tensor indices
    # u = i*m + j = 4i + j, where i in {0,1,2,3}, j in {0,1,2,3}
    # v = j*p + k = 4j + k, where j in {0,1,2,3}, k in {0,1,2,3}
    # w = k*n + i = 4k + i, where k in {0,1,2,3}, i in {0,1,2,3}
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where A00 is rows 0-1, cols 0-1; A01 is rows 0-1, cols 2-3
    #    A10 is rows 2-3, cols 0-1; A11 is rows 2-3, cols 2-3
    # B = [[B00, B01], [B10, B11]] where B00 is rows 0-1, cols 0-1; B01 is rows 0-1, cols 2-3
    #    B10 is rows 2-3, cols 0-1; B11 is rows 2-3, cols 2-3
    # C = [[C00, C01], [C10, C11]] where C00 is rows 0-1, cols 0-1; C01 is rows 0-1, cols 2-3
    #    C10 is rows 2-3, cols 0-1; C11 is rows 2-3, cols 2-3
    
    # Strassen products (at block level, each is 2x2 by 2x2):
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B10 - B11)
    # P4 = A11(B01 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pk is a 2x2 by 2x2 multiplication, which has rank 4 (schoolbook)
    # So total rank = 7 * 4 = 28 for 4x4 by 4x4
    
    # Now extend to 4x4 by 4x7:
    # Split B into B4 (4x4) and B3 (4x3)
    # C = A*B4 + A*B3
    # A*B4: rank 28
    # A*B3: schoolbook, rank 4*4*3 = 48
    # Total: 76
    
    # Let's construct this
    
    rank = 28 + 48
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * rank for _ in range(p * n)]  # 28 x rank
    
    # Part 1: Strassen for 4x4 by 4x4 (first 4 columns of B)
    # Each of the 7 Strassen products is a 2x2 by 2x2 multiplication
    # For each such product, we add 4 terms (schoolbook for 2x2x2)
    
    # Define the 7 Strassen factors
    # For each factor, we need U and V coefficients for the 2x2x2 multiplication
    # Then sum the W contributions
    
    # Strassen factor 1: (A00 + A11)(B00 + B11)
    # A00: rows 0-1, cols 0-1 of A -> u indices: 0,1,4,5
    # A11: rows 2-3, cols 2-3 of A -> u indices: 10,11,14,15
    # B00: rows 0-1, cols 0-1 of B -> v indices: 0,1,4,5
    # B11: rows 2-3, cols 2-3 of B -> v indices: 10,11,14,15
    
    # For a 2x2x2 multiplication with matrices X (2x2), Y (2x2), Z (2x2):
    # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    
    # In tensor indices for 2x2x2:
    # u = i*2 + j, v = j*2 + k, w = k*2 + i
    # Terms:
    # (u=0, v=0, w=0): X[0,0]*Y[0,0] -> Z[0,0]
    # (u=1, v=2, w=0): X[0,1]*Y[1,0] -> Z[0,0]
    # (u=0, v=1, w=1): X[0,0]*Y[0,1] -> Z[0,1]
    # (u=1, v=3, w=1): X[0,1]*Y[1,1] -> Z[0,1]
    # (u=2, v=0, w=2): X[1,0]*Y[0,0] -> Z[1,0]
    # (u=3, v=2, w=2): X[1,1]*Y[1,0] -> Z[1,0]
    # (u=2, v=1, w=3): X[1,0]*Y[0,1] -> Z[1,1]
    # (u=3, v=3, w=3): X[1,1]*Y[1,1] -> Z[1,1]
    
    # Map to global indices:
    # For A00 + A11: u_global = u_block + offset_A
    # For B00 + B11: v_global = v_block + offset_B
    # For C00: w_global = w_block + offset_C
    
    # Let's define a helper to add a 2x2x2 multiplication
    def add_2x2x2(U, V, W, u_offset, v_offset, w_offset, col_start, a_coeff=1, b_coeff=1):
        """Add a 2x2x2 multiplication with given offsets and column start."""
        terms = [
            (0, 0, 0), (1, 2, 0),
            (0, 1, 1), (1, 3, 1),
            (2, 0, 2), (3, 2, 2),
            (2, 1, 3), (3, 3, 3)
        ]
        for t in range(4):
            u_b, v_b, w_b = terms[t]
            u_g = u_b + u_offset
            v_g = v_b + v_offset
            w_g = w_b + w_offset
            col = col_start + t
            U[u_g][col] += a_coeff
            V[v_g][col] += b_coeff
            W[w_g][col] += 1
        return col + 4
    
    # Strassen factors for 4x4 by 4x4:
    # Each factor is a sum/difference of 2x2x2 multiplications
    
    col = 0
    
    # P1 = (A00 + A11)(B00 + B11)
    # This is: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # 4 separate 2x2x2 mults
    col = add_2x2x2(U, V, W, 0, 0, 0, col)  # A00*B00 -> C00
    col = add_2x2x2(U, V, W, 0, 8, 8, col)  # A00*B11 -> C01
    col = add_2x2x2(U, V, W, 8, 0, 0, col)  # A11*B00 -> C10
    col = add_2x2x2(U, V, W, 8, 8, 8, col)  # A11*B11 -> C11
    
    # P2 = (A10 + A11)B00
    # A10*B00 + A11*B00
    col = add_2x2x2(U, V, W, 4, 0, 0, col)  # A10*B00 -> C10
    col = add_2x2x2(U, V, W, 8, 0, 0, col)  # A11*B00 -> C11
    
    # P3 = A00(B10 - B11)
    # A00*B10 - A00*B11
    col = add_2x2x2(U, V, W, 0, 4, 8, col)  # A00*B10 -> C01
    col = add_2x2x2(U, V, W, 0, 8, 8, col)  # A00*B11 -> C00 (subtract)
    
    # P4 = A11(B01 - B00)
    # A11*B01 - A11*B00
    col = add_2x2x2(U, V, W, 8, 4, 0, col)  # A11*B01 -> C10
    col = add_2x2x2(U, V, W, 8, 0, 0, col)  # A11*B00 -> C11 (subtract)
    
    # P5 = (A00 + A01)B11
    # A00*B11 + A01*B11
    col = add_2x2x2(U, V, W, 0, 8, 8, col)  # A00*B11 -> C00
    col = add_2x2x2(U, V, W, 2, 8, 8, col)  # A01*B11 -> C01
    
    # P6 = (A10 - A00)(B00 + B01)
    # A10*B00 + A10*B01 - A00*B00 - A00*B01
    col = add_2x2x2(U, V, W, 4, 0, 0, col)  # A10*B00 -> C10
    col = add_2x2x2(U, V, W, 4, 4, 0, col)  # A10*B01 -> C10
    col = add_2x2x2(U, V, W, 0, 0, 0, col)  # A00*B00 -> C00 (subtract)
    col = add_2x2x2(U, V, W, 0, 4, 0, col)  # A00*B01 -> C01 (subtract)
    
    # P7 = (A01 - A11)(B10 + B11)
    # A01*B10 + A01*B11 - A11*B10 - A11*B11
    col = add_2x2x2(U, V, W, 2, 4, 8, col)  # A01*B10 -> C01
    col = add_2x2x2(U, V, W, 2, 8, 8, col)  # A01*B11 -> C11
    col = add_2x2x2(U, V, W, 8, 4, 8, col)  # A11*B10 -> C10 (subtract)
    col = add_2x2x2(U, V, W, 8, 8, 8, col)  # A11*B11 -> C11 (subtract)
    
    # Now we have 28 columns used. But this is just the sum of all the 2x2x2 mults.
    # We need to combine them with the Strassen coefficients.
    # Actually, the above adds all the terms with coefficient +1.
    # But Strassen requires specific coefficients.
    
    # Let me reconsider. The Strassen algorithm says:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # where Pk are the 7 products.
    
    # Each Pk is itself a 2x2x2 multiplication. So we can't just sum them all.
    # We need to keep them separate and then combine.
    
    # This approach is getting complex. Let's use a different strategy.
    
    # Use the known Laderman algorithm for 4x4x4 with rank 49?
    # Or just use schoolbook for simplicity and correctness.
    
    # Given time constraints, let's return the schoolbook construction.
    # Rank = 112
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u][col] = 1
                V[v][col] = 1
                W[w][col] = 1
                col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify correctness if time permits
    if time.time() < deadline - 10:
        target = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target.add((i * m + j, j * p + k, k * n + i))
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for (u, v, w) in target:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                raise ValueError(f"Target entry ({u},{v},{w}) = {val}, expected 1")
        
        np.random.seed(42)
        for _ in range(5000):
            u = np.random.randint(0, n * m)
            v = np.random.randint(0, m * p)
            w = np.random.randint(0, p * n)
            if (u, v, w) in target:
                continue
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val) > 1e-9:
                raise ValueError(f"Non-target entry ({u},{v},{w}) = {val}, expected 0")
    
    return result
# EVOLVE-BLOCK-END
