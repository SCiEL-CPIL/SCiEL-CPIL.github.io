"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from fractions import Fraction

def construct():
    """
    Return the factor matrices U, V, W for a tensor decomposition of the
    (n, m, p) matrix multiplication tensor.
    
    We use a block decomposition strategy:
    - Split the middle dimension (m=4) into two blocks of size 2
    - For each block, compute the rank-2 subproblem
    - Combine the results
    
    For 3x2x5 (A is 3x2, B is 2x5, C is 3x5):
    This can be decomposed using known results or we can use schoolbook.
    
    Let's use a direct approach with known efficient decompositions:
    R(3,2,2) = 11 (Strassen-like)
    R(3,2,3) = 17 (can be built from R(3,2,2) + R(3,2,1))
    R(3,2,1) = 6 (schoolbook for matrix-vector)
    
    Actually, let's be more careful. For 3x4x5:
    We can split as:
    A (3x4) = [A1 (3x2) | A2 (3x2)]
    B (4x5) = [B1 (2x5); B2 (2x5)]
    C = A1*B1 + A2*B2
    
    Each term A_i * B_i is 3x2 times 2x5 = 3x5
    
    For 3x2x5, we can further decompose. Let's use:
    R(3,2,5) <= R(3,2,3) + R(3,2,2) = 17 + 11 = 28
    
    So total rank = 2 * 28 = 56
    
    But we need explicit decompositions. Let's build them carefully.
    
    First, R(3,2,2):
    A is 3x2, B is 2x2, C is 3x2
    We can write this as:
    C[0:2, :] = A[0:2, :] * B  (2x2x2, rank 7 via Strassen)
    C[2, :] = A[2, :] * B       (1x2x2, rank 4)
    Total: 11
    
    For R(3,2,3):
    A is 3x2, B is 2x3, C is 3x3
    Split B into B[:,0:2] and B[:,2:3]:
    C[:, 0:2] = A * B[:, 0:2]  (3x2x2, rank 11)
    C[:, 2:3] = A * B[:, 2:3]  (3x2x1, rank 6)
    Total: 17
    
    Let's implement these explicitly.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Build the full decomposition by combining sub-decompositions
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (20, rank) 
    # W: (p*n, rank) = (15, rank)
    
    # We'll build terms as lists of (u_indices, v_indices, w_indices, coefficient)
    # where u_indices, v_indices, w_indices are positions to set to coefficient
    
    terms = []  # Each term: (u_pos, v_pos, w_pos, coeff)
    
    def add_term(u_pos, v_pos, w_pos, coeff=1):
        terms.append((u_pos, v_pos, w_pos, coeff))
    
    # For the 2x2x2 Strassen decomposition:
    # A is 2x2 (indices 0-3: (0,0), (0,1), (1,0), (1,1))
    # B is 2x2 (indices 0-3: (0,0), (0,1), (1,0), (1,1))
    # C is 2x2 (indices 0-3: (0,0), (0,1), (1,0), (1,1))
    
    # M1 = (a00 + a11)(b00 + b11)
    # Contributes to: c00 (+1), c01 (+1), c11 (+1)
    add_term([0, 3], [0, 3], [0, 1, 3], 1)
    
    # M2 = (a10 + a11) * b00
    # Contributes to: c10 (+1), c11 (-1)
    add_term([2, 3], [0], [2, 3], 1)
    add_term([2, 3], [0], [3], -1)
    
    # M3 = a00 * (b10 - b11)
    # Contributes to: c01 (+1), c11 (+1)
    add_term([0], [2, 3], [1], 1)
    add_term([0], [2, 3], [3], 1)
    
    # M4 = a11 * (b01 - b00)
    # Contributes to: c00 (+1), c10 (+1)
    add_term([3], [1, 0], [0], 1)
    add_term([3], [1, 0], [2], 1)
    
    # M5 = (a00 + a01) * b11
    # Contributes to: c00 (-1), c11 (+1)
    add_term([0, 1], [3], [0], -1)
    add_term([0, 1], [3], [3], 1)
    
    # M6 = (a10 - a00) * (b00 + b10)
    # M7 = (a01 - a11) * (b01 + b11)
    
    # Actually, the standard Strassen formula is:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b10 - b11)
    # p4 = a11(b01 - b00)
    # p5 = (a00 + a01)b11
    # p6 = (a10 - a00)(b00 + b10)
    # p7 = (a01 - a11)(b01 + b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p1
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p5
    
    # Let me redo this properly with all 7 terms:
    terms = []
    
    # p1 = (a00 + a11)(b00 + b11)
    # u: a00, a11 -> indices 0, 3
    # v: b00, b11 -> indices 0, 3
    # w: c00 (+1), c01 (+1), c11 (+1) -> indices 0, 1, 3
    add_term([0, 3], [0, 3], [0, 1, 3], 1)
    
    # p2 = (a10 + a11)b00
    # u: a10, a11 -> indices 2, 3
    # v: b00 -> index 0
    # w: c10 (+1), c11 (-1) -> indices 2, 3
    add_term([2, 3], [0], [2], 1)
    add_term([2, 3], [0], [3], -1)
    
    # p3 = a00(b10 - b11)
    # u: a00 -> index 0
    # v: b10, b11 -> indices 2, 3
    # w: c01 (+1), c11 (+1) -> indices 1, 3
    add_term([0], [2, 3], [1], 1)
    add_term([0], [2, 3], [3], 1)
    
    # p4 = a11(b01 - b00)
    # u: a11 -> index 3
    # v: b01, b00 -> indices 1, 0
    # w: c00 (+1), c10 (+1) -> indices 0, 2
    add_term([3], [1, 0], [0], 1)
    add_term([3], [1, 0], [2], 1)
    
    # p5 = (a00 + a01)b11
    # u: a00, a01 -> indices 0, 1
    # v: b11 -> index 3
    # w: c00 (-1), c11 (+1) -> indices 0, 3
    add_term([0, 1], [3], [0], -1)
    add_term([0, 1], [3], [3], 1)
    
    # p6 = (a10 - a00)(b00 + b10)
    # u: a10, a00 -> indices 2, 0
    # v: b00, b10 -> indices 0, 2
    # w: This term doesn't appear in the standard formula I wrote...
    # Let me check the correct Strassen formula again.
    
    # Actually, I think I'm making an error. Let me use a verified 7-term decomposition.
    # The correct Strassen for 2x2x2 is:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b10)
    # M7 = (a01 - a11)(b01 + b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M1
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M5
    
    # Wait, M6 still doesn't appear. That's suspicious.
    
    # Let me just use the schoolbook for now to ensure correctness, then optimize.
    # Schoolbook for 3x4x5 has rank 60.
    
    # Actually, let me try a different approach: use known efficient algorithms.
    # For now, let's return a valid (but not optimal) decomposition.
    
    # Schoolbook: rank = n * m * p = 3 * 4 * 5 = 60
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
