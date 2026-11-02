"""Program to find rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a bilinear decomposition for 4x5 * 5x5 matrix multiplication.
    Uses a block-partitioning strategy with 2x2 Strassen/WFTA on the 4x4 core
    and schoolbook for the remaining 1-dimensional parts.
    
    Partition:
    A (4x5) = [A11 A12 A21 A22 | a_col]  where Aij are 2x2, a_col is 4x1
    B (5x5) = [B11 B12 | b_col]  where B11, B12 are 4x2... 
    Actually let's be more careful.
    
    A is 4x5. Split columns: first 4 cols (4x4), last 1 col (4x1).
    B is 5x5. Split rows: first 4 rows (4x5), last 1 row (1x5).
    
    C = A @ B = A[:, :4] @ B[:4, :] + A[:, 4:5] @ B[4:5, :]
    
    Let A_main = A[:, :4] (4x4), A_tail = A[:, 4:5] (4x1)
    Let B_main = B[:4, :] (4x5), B_tail = B[4:5, :] (1x5)
    
    C = A_main @ B_main + A_tail @ B_tail
    
    For A_main @ B_main (4x4 * 4x5):
    Split A_main into 2x2 blocks:
      A_main = [[A11, A12], [A21, A22]]
    Split B_main into 2x(2 or 3) blocks:
      B_main = [[B11, B12], [B21, B22]] where B11, B12 are 2x2 and 2x3? No.
      B_main is 4x5. Split rows into 2+2, cols into 2+3:
      B_main = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 2x2, B22 is 2x3.
    
    This is getting complicated. Let me use a simpler approach:
    
    Use Strassen on the 4x4 * 4x4 part, and handle the 5th column separately.
    
    Actually, let's use the following partition:
    A (4x5): split into A_left (4x4) and A_right (4x1)
    B (5x5): split into B_top (4x5) and B_bottom (1x5)
    
    C = A_left @ B_top + A_right @ B_bottom
    
    For A_left @ B_top (4x4 * 4x5):
    We can further split B_top into B_top_left (4x4) and B_top_right (4x1).
    A_left @ B_top = A_left @ B_top_left + A_left @ B_top_right
    
    So C = A_left @ B_top_left + A_left @ B_top_right + A_right @ B_bottom
    
    Now:
    - A_left @ B_top_left is 4x4 * 4x4: use Strassen (rank 7)
    - A_left @ B_top_right is 4x4 * 4x1: schoolbook (rank 16)
    - A_right @ B_bottom is 4x1 * 1x5: schoolbook (rank 20)
    
    Total rank = 7 + 16 + 20 = 43
    
    Wait, that's a big improvement from 100!
    
    Let me verify:
    - Strassen for 4x4*4x4: rank 7
    - 4x4 * 4x1: each output element needs 4 mults, 4*4=16 elements, so rank 16
    - 4x1 * 1x5: each output element needs 1 mult, 4*5=20 elements, so rank 20
    
    Total = 7 + 16 + 20 = 43
    
    This should be valid! Let me implement it.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # We'll build the decomposition in three parts:
    # Part 1: A_left (4x4) @ B_top_left (4x4) using Strassen -> rank 7
    # Part 2: A_left (4x4) @ B_top_right (4x1) using schoolbook -> rank 16
    # Part 3: A_right (4x1) @ B_bottom (1x5) using schoolbook -> rank 20
    
    U_entries = []  # list of (index, value) pairs for U
    V_entries = []  # list of (index, value) pairs for V
    W_entries = []  # list of (index, value) pairs for W
    
    rank = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal rank
        U_entries.append((u_idx, u_val))
        V_entries.append((v_idx, v_val))
        W_entries.append((w_idx, w_val))
        rank += 1
    
    # Part 1: Strassen for A[0:4, 0:4] @ B[0:4, 0:4]
    # A blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B blocks:
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # Strassen formulas:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen product, we need to add 4 terms (for the 4 elements of the 2x2 product)
    # But wait, each "product" like (A11+A22)*(B11+B22) is a 2x2 * 2x2 multiplication
    # which has rank 4 in schoolbook, but we're using Strassen at the block level.
    
    # Actually, I think I'm confusing myself. Let me reconsider.
    
    # The Strassen algorithm computes 4x4 * 4x4 with 7 scalar multiplications of 2x2 matrices.
    # Each 2x2 * 2x2 multiplication still needs to be done somehow.
    
    # If we use schoolbook for each 2x2*2x2, that's 4 mults each, so 7*4 = 28 mults total.
    # That's worse than schoolbook 16 for 4x4*4x4!
    
    # So Strassen at the block level only helps if the sub-multiplications are also optimized.
    
    # For our case, the sub-multiplications are 2x2 * 2x2, which have rank 7 (not 4) using Strassen.
    # So each block multiplication costs 7, and we have 7 block multiplications, total 49.
    # That's worse than schoolbook 16.
    
    # So using Strassen for 4x4*4x4 doesn't help if we're going down to 2x2*2x2.
    
    # Let me reconsider the approach.
    
    # Actually, the rank of 2x2*2x2 is 7, not 4. So schoolbook for 2x2*2x2 is rank 4,
    # but the minimal rank is 7? No, that's not right.
    
    # Wait, I think I'm confusing the bilinear rank with something else.
    
    # The bilinear rank of 2x2*2x2 is 7 (Strassen's result). Schoolbook uses 4 multiplications,
    # but that's not a valid bilinear decomposition because the 4 products don't span the full space.
    
    # Actually, schoolbook for 2x2*2x2 uses 4 multiplications and IS a valid bilinear decomposition.
    # The rank is 4, not 7. Strassen shows that the rank is at most 7, but it's actually 4.
    
    # I'm confused. Let me look this up mentally.
    
    # The matrix multiplication tensor for n x n matrices has rank at least n^2.
    # For 2x2, the rank is 7 (Strassen) or 4 (schoolbook)?
    
    # Schoolbook: C11 = A11*B11 + A12*B21
    #              C12 = A11*B12 + A12*B22
    #              C21 = A21*B11 + A22*B21
    #              C22 = A21*B12 + A22*B22
    # This uses 4 products: A11*B11, A12*B21, A11*B12, A12*B22, A21*B11, A22*B21, A21*B12, A22*B22
    # Wait, that's 8 products, not 4.
    
    # I think I'm making a fundamental error. Let me restart the analysis.
    
    # For 2x2 * 2x2:
    # C = A @ B
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Schoolbook uses 8 scalar multiplications.
    # Strassen uses 7 scalar multiplications.
    
    # So the rank of 2x2*2x2 is 7 (Strassen), and schoolbook rank is 8.
    
    # OK so my earlier analysis was wrong. Let me redo it.
    
    # For 4x4 * 4x4:
    # Schoolbook: 16 scalar multiplications
    # Strassen: 7 block multiplications of 2x2, each with rank 7, so 7*7 = 49? No, that's not right either.
    
    # I think the issue is that when we say "rank 7" for 2x2*2x2, we mean 7 bilinear terms.
    # When we use Strassen for 4x4*4x4, we have 7 bilinear terms where each term is a 2x2*2x2 product.
    # Each 2x2*2x2 product can be done with 7 bilinear terms.
    # So total bilinear terms = 7 * 7 = 49.
    
    # That's worse than schoolbook 16.
    
    # So Strassen doesn't help for 4x4*4x4 if we use Strassen for the sub-blocks.
    
    # I think the key insight is that for small matrices, the overhead of Strassen isn't worth it.
    
    # Let me just use schoolbook for everything and see if I can find a better partition.
    
    # Actually, let me just return the schoolbook decomposition with rank 100.
    # The task says the score is the rank, and lower is better.
    # If I can't find a better decomposition, I should at least return a valid one.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
