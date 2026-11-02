import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    
    # Rank 28 for 4x4x4 (Strassen outer, Schoolbook inner)
    # Rank 48 for 4x4x3 (Schoolbook)
    # Total Rank = 76
    
    # We will build U, V, W lists
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx are global indices
    # coeffs are 1.0 (or -1.0)
    def add_term(u_idx, v_idx, w_idx, coeff):
        U.append((u_idx, coeff))
        V.append((v_idx, coeff))
        W.append((w_idx, coeff))
        
    # --- Part 1: 4x4x4 (Cols 0-3 of B, Cols 0-3 of C) ---
    # Strassen 2x2 formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Block indices: 00, 01, 10, 11
    # A_block[r][c], B_block[r][c], C_block[r][c]
    # r,c in {0,1}
    
    # For each Strassen term, we have X and Y (2x2 matrices of blocks)
    # We compute Z = X Y using schoolbook for 2x2 blocks
    # This generates 4 terms per block multiplication?
    # No, Z is a 2x2 matrix of blocks.
    # Each block Z[r][c] is a 2x2 matrix.
    # Z[r][c] = sum_k X[r][k] Y[k][c]
    # This is a sum of 2 block products.
    # Each block product is a 2x2x2 multiplication (Schoolbook, Rank 4).
    
    # So for each Strassen term, we have 4 blocks in Z.
    # Each block in Z is sum of 2 block products.
    # Each block product has rank 4.
    # So each Strassen term has rank 4 * 2 * 4 = 32?
    # Wait.
    # Z[r][c] = X[r][0]Y[0][c] + X[r][1]Y[1][c].
    # Term 1: X[r][0]Y[0][c]. Rank 4.
    # Term 2: X[r][1]Y[1][c]. Rank 4.
    # So Z[r][c] has rank 8.
    # Total Z has rank 32.
    # Total 7 terms -> 224.
    
    # This contradicts the "Rank 28" claim.
    # The Rank 28 claim assumes we can use Rank 7 for the 2x2x2 block multiplication.
    # If we use Rank 7 for the block multiplication, then:
    # Each block product X[r][k]Y[k][c] is not a single term.
    # The product of two 2x2 matrices has rank 7.
    # So X[r][0]Y[0][c] is not a single rank-1 term, it's a rank-7 tensor.
    # So we cannot just "add" them.
    
    # Correct interpretation:
    # The tensor for 4x4x4 is a sum of 7 tensors, each corresponding to a Strassen step.
    # Each Strassen step is a 2x2x2 multiplication of blocks.
    # The rank of 2x2x2 is 7.
    # So each Strassen step contributes 7 rank-1 terms to the block tensor.
    # Each block rank-1 term corresponds to a scalar multiplication of two 2x2 matrices.
    # So we need to expand these scalar multiplications.
    # A scalar multiplication of two 2x2 matrices is a 2x2x2 multiplication.
    # Its rank is 4 (Schoolbook).
    # So each block rank-1 term expands to 4 scalar rank-1 terms.
    # Total rank = 7 (Strassen steps) * 7 (Block rank) * 4 (Inner schoolbook) = 196.
    
    # This is worse than 112.
    
    # So, Rank 28 is NOT achievable with this simple nesting.
    # The Rank 28 result for 4x4x4 likely uses a specific non-nested algorithm.
    
    # Given the complexity, I will stick to the safe decomposition:
    # 4x4x4 Schoolbook (64) + 4x4x3 Schoolbook (48) = 112.
    # This is the current score.
    
    # Is there ANY improvement?
    # Maybe 4x4x4 can be done in 49?
    # If we use Strassen for the outer blocks (7 terms), and Strassen for the inner blocks (7 terms).
    # Total 49.
    # 49 + 48 = 97.
    
    # Let's implement Rank 49 for 4x4x4.
    # Outer Strassen: 7 terms.
    # Each term is a 2x2 block multiplication.
    # We use Strassen for the 2x2 block multiplication (Rank 7).
    # So each outer term yields 7 inner terms.
    # Each inner term is a scalar multiplication of two 2x2 matrices.
    # We use Schoolbook for the 2x2 matrix multiplication (Rank 4).
    # So each inner term yields 4 scalar terms.
    # Total 7 * 7 * 4 = 196.
    
    # Wait, if we use Strassen for the inner blocks, the "scalars" are 1x1.
    # So the inner terms are scalar multiplications.
    # So each inner term is 1 rank-1 term.
    # Total 7 * 7 = 49.
    
    # So, Rank 49 is achievable by using Strassen for BOTH outer and inner blocks.
    
    # Let's implement this.
    
    # Strassen 2x2 terms:
    # We need to map the 7 Strassen terms to the global indices.
    
    return {"U": [], "V": [], "W": []} # Placeholder

