"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120  # safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try the simple Strassen-style recursive approach
    # Then if time permits, try to find a better decomposition
    
    # Let's implement a rank-reduction approach using the known
    # Strassen 2x2x2 decomposition and combine with schoolbook for remaining
    
    # The key insight: we can decompose the 4x4x6 multiplication
    # by splitting into smaller blocks and using Strassen where possible
    
    # Strategy: Split 4x4 into 2x2 blocks (2x2 blocks of size 2x2)
    # Split 4x6 into 2x3 blocks of size 2x2
    # This gives 2x2x3 blocks, each of which can use rank 11
    # (7 from Strassen 2x2x2 + 4 from schoolbook 2x2x1)
    # Total: 2*2*3 * 11 = 132... wait, that's worse.
    
    # Actually, let me think differently. The schoolbook rank is 4*4*6 = 96.
    # We need to find a decomposition with rank < 96.
    
    # Let's use a hybrid approach:
    # For the 4x4x6 case, we can potentially use known results.
    # The tensor rank for matrix multiplication [4,4,6] is unknown exactly,
    # but we know it's at most 96 (schoolbook) and at least some lower bound.
    
    # Let's try a structured approach using Kronecker products and Strassen
    
    # Actually, let me try a simpler improvement first:
    # Use the fact that we can apply Strassen to 2x2x2 subproblems
    
    # Split A (4x4) into four 2x2 blocks: A11, A12, A21, A22
    # Split B (4x6) into six 2x2 blocks: B11, B12, B13, B21, B22, B23
    # Product C = A*B has blocks C11, C12, C13, C21, C22, C23
    # Cij = sum over k of Akj * Bkj
    
    # For each output block Cij, we need to compute a 2x2x3 sum.
    # Each such sum can be done with rank 11 (7+4) as mentioned.
    # 6 output blocks * 11 = 66 rank terms.
    # Plus we need to handle the block structure overhead.
    
    # Wait, this doesn't account for how the factors combine.
    
    # Let me just implement a direct search for a better decomposition
    # using known algebraic identities.
    
    # For now, let's return the schoolbook decomposition as a baseline,
    # but with a deadline guard.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Check if we're still within deadline
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better decomposition using Strassen-style methods
    # Let's attempt a rank 84 decomposition using 2x2x3 Strassen terms
    
    # For 2x2x3 matrix multiplication, we can use:
    # M1 = (A11+A22)*(B11+B22+B33)  [not quite, need to be careful]
    
    # Actually, for multiplying a 2x2 by 2x3, the rank is 7 (Strassen-like)
    # Let's verify: 2*2*3 = 12 schoolbook, Strassen gives 7
    
    # The 2x2x2 Strassen decomposition has rank 7:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*(B12-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # For 2x2x3, we can split the 3 columns into 2+1 and apply
    # Strassen to the 2x2x2 part and schoolbook to the 2x2x1 part.
    # Rank = 7 + 4 = 11 per 2x2x3 block.
    
    # But we need to be careful about how these combine.
    
    # Let me implement the 2x2x3 rank 11 decomposition explicitly.
    
    def get_2x2x3_decomp():
        """Return U, V, W matrices for 2x2x3 matrix mult with rank 11."""
        # Split B (2x3) into B_left (2x2) and B_right (2x1)
        # C = A * B = A * B_left + A * B_right
        # A * B_left uses Strassen rank 7
        # A * B_right uses schoolbook rank 4
        # Total rank 11
        
        U = np.zeros((4, 11), dtype=object)  # 2x2 = 4 entries
        V = np.zeros((6, 11), dtype=object)  # 2x3 = 6 entries
        W = np.zeros((6, 11), dtype=object)  # 2x3 = 6 entries (output)
        
        # Strassen 2x2x2 part (columns 0-5 of B)
        # M1 = (A11+A22)*(B11+B22)
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        W[0, 0] += 1; W[3, 0] += 1  # Wait, output is 2x2, so W should be 4x11
        
        # Actually, let me redo this. The output C is 2x2 for the left part.
        # W should have shape (2*2, 11) = (4, 11) for the Strassen part.
        
        # Let me restructure: return separate decompositions
        return None
    
    # This is getting complex. Let me just return the schoolbook for now.
    # The deadline guard ensures we don't timeout.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
