"""Rewritten program: rank-49 Strassen decomposition for 4x4 matrix multiplication."""
import time
import numpy as np

# Deadline guard: kill at 150s, return best with safety margin
_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-49 Strassen decomposition for 4x4 matrix multiplication.
    
    Uses the standard 7-term Strassen recursion on 2x2 blocks, where each
    block product is itself computed via Strassen (7 terms), giving 7*7=49 terms.
    
    The 4x4 matrices A and B are partitioned into 2x2 blocks of 2x2 submatrices:
        A = [[A00, A01],
             [A10, A11]]
        B = [[B00, B10],  (note: B is partitioned with row blocks)
             [B01, B11]]
    
    Wait - let me be careful with the block partitioning convention.
    For C = A * B where A is n x m and B is m x p:
        C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
             [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Where A00 is top-left block (rows 0-1, cols 0-1), A01 is top-right (rows 0-1, cols 2-3),
    A10 is bottom-left (rows 2-3, cols 0-1), A11 is bottom-right (rows 2-3, cols 2-3).
    B00 is top-left (rows 0-1, cols 0-1), B01 is top-right (rows 0-1, cols 2-3),
    B10 is bottom-left (rows 2-3, cols 0-1), B11 is bottom-right (rows 2-3, cols 2-3).
    
    Strassen for the 2x2 block multiplication:
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B10 - B11)
        M4 = A11 * (B01 - B00)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B10)
        M7 = (A01 - A11) * (B01 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
    
    Each Mi is a product of two 2x2 matrices, computed via Strassen (7 terms each).
    Total: 7 * 7 = 49 rank-1 terms.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49) 
    # W has shape (p*n, rank) = (16, 49)
    # Convention: W[k*n + i, r] where i is row of C, k is col of C
    # C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] ... wait, let me re-examine.
    
    # Actually for matrix multiplication C = A*B:
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_{j} sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not quite right. The standard decomposition is:
    # C[i,k] = sum_{j} A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    # Hmm, that's not standard either. Let me think more carefully.
    
    # The matrix multiplication tensor T has entries T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # A rank-r decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where the "output" index is (i,k), so W is indexed by (i,k) with the trace convention W[k*n+i, r].
    
    # For Strassen, we work with block matrices. Let me construct the decomposition directly.
    
    # The 4x4 matrix A is flattened to index i*4+j for element A[i,j], so row i, col j -> index i*m+j = i*4+j.
    # B is flattened to j*4+k for element B[j,k], so row j, col k -> index j*p+k = j*4+k.
    # C output: W[k*n+i, r] means W index = k*4+i for C[i,k].
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Define the 2x2 block Strassen terms.
    # For a 2x2 matrix M = [[m00, m01], [m10, m11]], flattened as m00->0, m01->1, m10->2, m11->3.
    # Strassen 7 products for X*Y where X,Y are 2x2:
    # S1 = (X00 + X11) * (Y00 + Y11)
    # S2 = (X10 + X11) * Y00
    # S3 = X00 * (Y10 - Y11)
    # S4 = X11 * (Y01 - Y00)
    # S5 = (X00 + X01) * Y11
    # S6 = (X10 - X00) * (Y00 + Y10)
    # S7 = (X01 - X11) * (Y01 + Y11)
    # Result:
    # R00 = S1 + S4 - S5 + S7
    # R01 = S3 + S5
    # R10 = S2 + S4
    # R11 = S1 - S2 + S3 + S6
    
    # For each of the 7 big Strassen terms, we need to express X_k and Y_k as linear combos of 2x2 blocks,
    # then for each of the 7 small Strassen terms, express the factors as linear combos of individual 4x4 elements.
    
    # Let me define helper functions.
    # A 2x2 block of the 4x4 matrix is identified by (block_row, block_col) where block_row in {0,1}, block_col in {0,1}.
    # Block (br, bc) has elements at positions:
    #   (2*br + 0, 2*bc + 0) -> index (2*br)*4 + (2*bc+0)
    #   (2*br + 0, 2*bc + 1) -> index (2*br)*4 + (2*bc+1)
    #   (2*br + 1, 2*bc + 0) -> index (2*br+1)*4 + (2*bc+0)
    #   (2*br + 1, 2*bc + 1) -> index (2*br+1)*4 + (2*bc+1)
    
    # For U: A[i,j] is at index i*4+j.
    # For V: B[j,k] is at index j*4+k.
    
    # Define the 7 big Strassen X and Y factors as linear combos of 2x2 blocks.
    # Each block is a 2x2 matrix. We represent a linear combo of blocks as a dict mapping (block_row, block_col) to coefficient.
    
    # Big Strassen terms (X_k, Y_k) for 2x2 block multiplication:
    # M1: X = A00 + A11, Y = B00 + B11
    # M2: X = A10 + A11, Y = B00
    # M3: X = A00, Y = B10 - B11
    # M4: X = A11, Y = B01 - B00
    # M5: X = A00 + A01, Y = B11
    # M6: X = A10 - A00, Y = B00 + B10
    # M7: X = A01 - A11, Y = B01 + B11
    
    # For each big term k, X_k is a linear combo of 2x2 blocks of A, Y_k is a linear combo of 2x2 blocks of B.
    # Then for each small Strassen term s in the 2x2 product X_k * Y_k, we have factors that are linear combos
    # of the elements of X_k and Y_k, which are themselves linear combos of blocks, which are linear combos of 4x4 elements.
    
    # Let me build this step by step.
    
    # Small Strassen factors for 2x2 matrix X (elements x00, x01, x10, x11) and Y (elements y00, y01, y10, y11):
    # S1: X factor = x00 + x11, Y factor = y00 + y11
    # S2: X factor = x10 + x11, Y factor = y00
    # S3: X factor = x00, Y factor = y10 - y11
    # S4: X factor = x11, Y factor = y01 - y00
    # S5: X factor = x00 + x01, Y factor = y11
    # S6: X factor = x10 - x00, Y factor = y00 + y10
    # S7: X factor = x01 - x11, Y factor = y01 + y11
    
    # Output contributions:
    # S1 -> R00 (+1), R11 (+1)
    # S2 -> R10 (+1), R11 (-1)
    # S3 -> R01 (+1), R11 (+1)
    # S4 -> R00 (+1), R10 (+1)
    # S5 -> R00 (-1), R01 (+1)
    # S6 -> R11 (+1)
    # S7 -> R00 (+1)
    
    # For the 2x2 block output, R00 goes to C block (0,0), R01 to C block (0,1), R10 to C block (1,0), R11 to C block (1,1).
    
    # C block (br, bc) has elements at:
    #   (2*br+0, 2*bc+0), (2*br+0, 2*bc+1), (2*br+1, 2*bc+0), (2*br+1, 2*bc+1)
    # which map to C indices: (2*br+0)*4+(2*bc+0), etc.
    # And W index for C[i,k] is k*4+i.
    
    # Now let's construct each of the 49 terms.
    
    # For term number t (0 to 48), let big_term = t // 7, small_term = t % 7.
    
    # Define big X and Y factors.
    # Each is a dict: {(block_row, block_col): coeff}
    
    big_X = [
        {(0,0): 1, (1,1): 1},      # M1: A00 + A11
        {(1,0): 1, (1,1): 1},      # M2: A10 + A11
        {(0,0): 1},                # M3: A00
        {(1,1): 1},                # M4: A11
        {(0,0): 1, (0,1): 1},      # M5: A00 + A01
        {(1,0): 1, (0,0): -1},     # M6: A10 - A00
        {(0,1): 1, (1,1): -1},     # M7: A01 - A11
    ]
    
    big_Y = [
        {(0,0): 1, (1,1): 1},      # M1: B00 + B11
        {(0,0): 1},                # M2: B00
        {(1,0): 1, (1,1): -1},     # M3: B10 - B11
        {(0,1): 1, (0,0): -1},     # M4: B01 - B00
        {(1,1): 1},                # M5: B11
        {(0,0): 1, (1,0): 1},      # M6: B00 + B10
        {(0,1): 1, (1,1): 1},      # M7: B01 + B11
    ]
    
    # Small Strassen factors: for each small term s, we have (x_elements, y_elements) where each is a dict of element indices to coefficients.
    # Elements of a 2x2 matrix: 0=x00, 1=x01, 2=x10, 3=x11.
    small_X = [
        {0: 1, 3: 1},    # S1: x00 + x11
        {2: 1, 3: 1},    # S2: x10 + x11
        {0: 1},          # S3: x00
        {3: 1},          # S4: x11
        {0: 1, 1: 1},    # S5: x00 + x01
        {2: 1, 0: -1},   # S6: x10 - x00
        {1: 1, 3: -1},   # S7: x01 - x11
    ]
    
    small_Y = [
        {0: 1, 3: 1},    # S1: y00 + y11
        {0: 1},          # S2: y00
        {2: 1, 3: -1},   # S3: y10 - y11
        {1: 1, 0: -1},   # S4: y01 - y00
        {3: 1},          # S5: y11
        {0: 1, 2: 1},    # S6: y00 + y10
        {1: 1, 3: 1},    # S7: y01 + y11
    ]
    
    # Output contributions: small term s contributes to which R block positions with what coefficient.
    # R positions: 0=R00, 1=R01, 2=R10, 3=R11
    small_output = [
        {0: 1, 3: 1},    # S1: R00 +1, R11 +1
        {2: 1, 3: -1},   # S2: R10 +1, R11 -1
        {1: 1, 3: 1},    # S3: R01 +1, R11 +1
        {0: 1, 2: 1},    # S4: R00 +1, R10 +1
        {0: -1, 1: 1},   # S5: R00 -1, R01 +1
        {3: 1},          # S6: R11 +1
        {0: 1},          # S7: R00 +1
    ]
    
    # For a 2x2 block (br, bc) of the 4x4 matrix, element e (0=x00, 1=x01, 2=x10, 3=x11) maps to:
    # For A (input to U): A[2*br + e_row, 2*bc + e_col] where e_row = e // 2, e_col = e % 2.
    # Index in U: (2*br + e_row) * 4 + (2*bc + e_col)
    # For B (input to V): B[2*br + e_row, 2*bc + e_col], index in V: (2*br + e_row) * 4 + (2*bc + e_col)
    
    # For C output: C block (br, bc) element e maps to C[2*br + e_row, 2*bc + e_col].
    # W index: k * 4 + i where i = 2*br + e_row, k = 2*bc + e_col.
    
    for t in range(rank):
        big_t = t // 7
        small_t = t % 7
        
        # Build U vector: linear combo of A elements
        # U is formed by: for each block in big_X[big_t], for each element in small_X[small_t],
        # combine the contributions.
        
        # For each block (br, bc) with coefficient cX in big_X[big_t]:
        #   For each element e with coefficient cS in small_X[small_t]:
        #     The element A[2*br + e_row, 2*bc + e_col] gets coefficient cX * cS
        
        for (br, bc), cX in big_X[big_t].items():
            for e, cS in small_X[small_t].items():
                e_row = e // 2
                e_col = e % 2
                a_row = 2 * br + e_row
                a_col = 2 * bc + e_col
                u_idx = a_row * 4 + a_col
                U[u_idx, t] += cX * cS
        
        # Build V vector similarly
        for (br, bc), cY in big_Y[big_t].items():
            for e, cS in small_Y[small_t].items():
                e_row = e // 2
                e_col = e % 2
                b_row = 2 * br + e_row
                b_col = 2 * bc + e_col
                v_idx = b_row * 4 + b_col
                V[v_idx, t] += cY * cS
        
        # Build W vector: for each output block position (rpos) with coefficient cOut in small_output[small_t]:
        #   rpos 0 -> block (0,0), rpos 1 -> block (0,1), rpos 2 -> block (1,0), rpos 3 -> block (1,1)
        #   For each element e in the 2x2 output block, C[2*br + e_row, 2*bc + e_col] gets coefficient cOut
        #   But wait - the Strassen output R is a 2x2 matrix. The small term s contributes to specific positions
        #   of R. Each position of R corresponds to a 2x2 block of C. Within that block, all 4 elements get
        #   the same coefficient from the small term.
        
        for rpos, cOut in small_output[small_t].items():
            rbr = rpos // 2
            rbc = rpos % 2
            for e in range(4):
                e_row = e // 2
                e_col = e % 2
                c_row = 2 * rbr + e_row
                c_col = 2 * rbc + e_col
                w_idx = c_col * 4 + c_row  # W[k*n + i] = W[col * 4 + row]
                W[w_idx, t] += cOut
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
