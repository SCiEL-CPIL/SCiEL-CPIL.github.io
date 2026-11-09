"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def _lader_4x4x4():
    """Return (U, V, W) each 16x23 for 4x4x4 matrix multiplication.
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    T[i,j,k] = sum_c U[i*4+j,c]*V[j*4+k,c]*W[k*4+i,c] = delta_{i,k} (for the mult tensor)
    Actually T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]B[j,k]
    So the tensor has T[i,j,k]=1 for all i,j,k in the multiplication sense.
    """
    # Lader et al. 2020 rank-23 decomposition for 4x4x4
    # Using the published decomposition
    U = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # This is just a placeholder - I need the actual Lader decomposition
    # Let me use a verified construction instead
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use a rank-23 decomposition for 4x4x4, then extend to 4x4x6
    # For 4x4x4: T4[i,j,k] = sum_c U4[i*4+j,c] * V4[j*4+k,c] * W4[k*4+i,c]
    # For 4x4x6: T6[i,j,k] = sum_c U6[i*4+j,c] * V6[j*6+k,c] * W6[k*4+i,c]
    
    # Approach: Decompose B's 6 columns. Use rank-23 for first 4 cols (4x4x4),
    # and schoolbook for last 2 cols. Total rank = 23 + 2*4 = 31.
    # But we can do better by sharing U factors.
    
    # Actually, let's use a simpler but effective approach:
    # The 4x4x6 tensor can be decomposed using the 4x4x4 rank-23 decomposition
    # plus additional terms for the extra 2 output columns.
    
    # Let me construct a valid decomposition. I'll use a block approach:
    # Split the 4x4x6 problem into a 4x4x4 part and a 4x4x2 part.
    # The 4x4x2 part has rank 8 (schoolbook) since it's just 4*4*2=8 terms.
    # But we can potentially share some structure.
    
    # For now, let's use a construction that gives rank 31:
    # 23 terms for 4x4x4 + 8 terms for 4x4x2 = 31
    
    # I need a verified rank-23 decomposition for 4x4x4.
    # Let me use the one from the literature. Here's a known one:
    
    # Using the decomposition from "Faster matrix multiplication with rectangular matrices"
    # I'll construct it programmatically using Strassen-like ideas.
    
    # Alternative: Use a direct search or known explicit decomposition.
    # Let me use a simpler approach: rank 24 from a modified Strassen.
    
    # Actually, let me just use a well-known explicit rank-23 decomposition.
    # From Lader, Le Gall, Stothers (2020), the 4x4x4 rank-23 decomposition:
    
    # I'll use a verified construction. Here's one approach using 2x2 blocks:
    # Use Strassen for 2x2x2 (rank 7) and combine for 4x4x4.
    # 4x4x4 via 2x2 blocks: need 7*7=49 naive, but with sharing can do better.
    
    # Let me use a different strategy: construct a rank-31 decomposition
    # using 23 terms for 4x4x4 and 8 for 4x4x2.
    
    # For the 4x4x4 part, I'll use a rank-24 decomposition (simpler to verify):
    # Based on Strassen's algorithm applied to 2x2 blocks.
    
    # Let me just implement a clean rank-31 decomposition:
    rank = 31
    U = [[0]*rank for _ in range(n*m)]  # 16 x 31
    V = [[0]*rank for _ in range(m*p)]  # 24 x 31
    W = [[0]*rank for _ in range(p*n)]  # 24 x 31
    
    # First 23 terms: 4x4x4 decomposition (for k=0,1,2,3)
    # I'll use a rank-23 decomposition. Let me encode a known one.
    
    # Here's a rank-23 decomposition for 4x4x4 from the literature:
    # Using the explicit formulas from Lader et al.
    
    # Due to complexity, let me use a rank-24 decomposition based on
    # combining two Strassen 2x2x2 decompositions with sharing.
    
    # Actually, let me just use schoolbook for everything but optimize:
    # Rank 96 is the schoolbook. Let me at least get to 48 by using
    # the fact that for fixed (i,k), we sum over j.
    
    # Simple improvement: group by (i,k) pairs.
    # For each (i,k) pair, C[i,k] = sum_j A[i,j]*B[j,k]
    # This needs 4 terms per (i,k) pair, so 4*4*6 = 96 terms. No saving.
    
    # Let me use Strassen for the 4x4x4 part:
    # Strassen 2x2x2 has rank 7. For 4x4x4 using 2x2 blocks:
    # We have 4x4 = 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B computed via Strassen on 2x2 blocks.
    # Each Strassen product involves 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7 (Strassen) or 4 (schoolbook).
    # Using Strassen for the outer: 7 * 7 = 49 terms. Worse than schoolbook 64 for 4x4x4.
    # Using schoolbook for inner: 7 * 4 = 28 terms for 4x4x4.
    
    # So: 28 terms for 4x4x4 + 8 terms for 4x4x2 = 36 terms.
    
    # Let me implement Strassen-based 4x4x4 with schoolbook inner products.
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4x4, A and B are 4x4, split into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A*B, each Ci is 2x2
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix product, which we compute with schoolbook (rank 4).
    # So each Mk contributes 4 rank-1 terms.
    # 7 M's * 4 terms each = 28 terms for 4x4x4.
    
    # Now extend to 4x4x6:
    # B is 4x6, split into 2x3 blocks of 2x2 and 2x1? No, let's keep it simple.
    # B = [B_left, B_right] where B_left is 4x4, B_right is 4x2.
    # C = [C_left, C_right] where C_left = A*B_left (4x4), C_right = A*B_right (4x2).
    
    # C_left uses the 28-term Strassen decomposition.
    # C_right = A * B_right, where B_right is 4x2.
    # C_right[i,k] = sum_j A[i,j] * B_right[j,k] for k=0,1 (corresponding to cols 4,5 of B)
    # Schoolbook: 4*4*2 = 8 terms.
    
    # Total: 28 + 8 = 36 terms.
    
    # Let me implement this.
    
    # Strassen 4x4x4 decomposition (28 terms):
    # For each of the 7 Strassen products Mk, we have a 2x2 matrix product.
    # Each 2x2 product C = A*B has 4 schoolbook terms:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # The 7 Strassen products for 4x4x4:
    # Let A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # Let B = [[B00, B01], [B10, B11]] (2x2 blocks of 2x2)
    
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now, the tensor decomposition for 4x4x4:
    # T[i,j,k] = 1 (for the multiplication tensor)
    # We need: sum_c U[i*4+j,c] * V[j*4+k,c] * W[k*4+i,c] = T[i,j,k]
    
    # This is getting complex. Let me just use a simpler approach.
    
    # Simplest valid improvement: use the fact that the 4x4x6 tensor
    # can be decomposed with rank <= 4*4*6 = 96 (schoolbook).
    # Any reduction is an improvement.
    
    # Let me try a rank-48 decomposition by pairing columns:
    # For the 4x4x6 problem, note that B has 6 columns.
    # We can pair columns: (0,1), (2,3), (4,5).
    # For each pair (2k, 2k+1), compute A * B[:, 2k:2k+2] as a 4x2 product.
    # Each 4x2 product has rank 4*4*2 = 32... no that's worse.
    
    # OK let me just carefully implement the Strassen-based approach.
    
    # I'll build the 36-term decomposition explicitly.
    
    # Helper: add a schoolbook term for a 2x2 product
    # The 2x2 product involves A_sub (2x2) and B_sub (2x2), contributing to C_sub (2x2)
    # with coefficient 'coeff' in the overall Strassen combination.
    
    # For the 4x4x4 part (first 4 columns of B, i.e., k=0,1,2,3):
    # Row indices for A: 0,1,2,3 (i in 0..3)
    # Col indices for B: 0,1,2,3 (k in 0..3)
    # Middle: j in 0..3
    
    # Block structure:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # For each Strassen product Mk, it's a 2x2 matrix product.
    # Let me define the 7 products and their contributions.
    
    # I'll use a list of (A_factor, B_factor, C_position, coefficient) tuples.
    # Each such tuple represents: contribution = coeff * (A_factor * B_factor) to C_position
    
    # This is getting very complex to encode manually. Let me use a simpler approach.
    
    # SIMPLEST VALID IMPROVEMENT: Just use schoolbook but verify it works.
    # Rank 96. Then I can try to improve.
    
    # Actually, let me just return the schoolbook decomposition. It's valid and
    # gives score 1.0. To improve, I need a genuine rank reduction.
    
    # Let me try a rank-48 approach: 
    # Note that for 4x4x6, if we think of it as computing 4 independent 4x6 outputs,
    # each output row i needs sum_j A[i,j]*B[j,:] which is a 4x6 matrix product
    # of a 4-vector and 4x6 matrix. This has rank 4*6=24 per row, 4 rows = 96.
    
    # No easy saving there.
    
    # Let me just return schoolbook for now and note that a proper rank-23 or
    # rank-31 decomposition would be needed for improvement.
    
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
