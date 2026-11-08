"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Laderman's algorithm for 4x4x4 matrix multiplication (rank 20)
    # This is a well-known decomposition with integer entries.
    # The 20 terms are indexed 0..19.
    
    # For the 4x4x9 problem, we can use a block structure:
    # Split the 9 columns into two parts: 4 + 5
    # Use Laderman (rank 20) for the 4x4x4 part
    # Use schoolbook (rank 4*4*5 = 80) for the 4x4x5 part
    # Total rank: 20 + 80 = 100
    
    # Alternatively, we could try to find a better decomposition, but
    # for now let's construct a valid rank-100 decomposition.
    
    # Laderman's algorithm terms for 4x4x4:
    # Each term is a triple (A_part, B_part, C_part) where each part is a sparse matrix
    # We'll represent them as dictionaries mapping (row, col) -> value
    
    laderman_terms = [
        # Term 0: A[0,0]*B[0,0]
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # Term 1: A[0,1]*B[1,0]
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        # Term 2: A[1,0]*B[0,1]
        ({(1,0): 1}, {(0,1): 1}, {(0,1): 1}),
        # Term 3: A[1,1]*B[1,1]
        ({(1,1): 1}, {(1,1): 1}, {(1,1): 1}),
        # Term 4: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # Term 5: (A[0,1]+A[1,0])*(B[0,1]+B[1,0])
        ({(0,1): 1, (1,0): 1}, {(0,1): 1, (1,0): 1}, {(0,1): 1, (1,0): 1}),
        # Term 6: (A[0,0]-A[1,1])*(B[0,0]-B[1,1])
        ({(0,0): 1, (1,1): -1}, {(0,0): 1, (1,1): -1}, {(0,0): 1, (1,1): 1}),
        # Term 7: (A[0,1]-A[1,0])*(B[0,1]-B[1,0])
        ({(0,1): 1, (1,0): -1}, {(0,1): 1, (1,0): -1}, {(0,1): 1, (1,0): 1}),
        # Term 8: A[2,0]*B[0,2]
        ({(2,0): 1}, {(0,2): 1}, {(2,0): 1}),
        # Term 9: A[2,1]*B[1,2]
        ({(2,1): 1}, {(1,2): 1}, {(2,1): 1}),
        # Term 10: A[3,0]*B[0,3]
        ({(3,0): 1}, {(0,3): 1}, {(3,0): 1}),
        # Term 11: A[3,1]*B[1,3]
        ({(3,1): 1}, {(1,3): 1}, {(3,1): 1}),
        # Term 12: A[2,2]*B[2,0]
        ({(2,2): 1}, {(2,0): 1}, {(0,2): 1}),
        # Term 13: A[2,3]*B[3,0]
        ({(2,3): 1}, {(3,0): 1}, {(0,3): 1}),
        # Term 14: A[3,2]*B[2,1]
        ({(3,2): 1}, {(2,1): 1}, {(1,3): 1}),
        # Term 15: A[3,3]*B[3,1]
        ({(3,3): 1}, {(3,1): 1}, {(1,3): 1}),
        # Term 16: (A[2,0]+A[3,1])*(B[0,2]+B[1,3])
        ({(2,0): 1, (3,1): 1}, {(0,2): 1, (1,3): 1}, {(2,0): 1, (3,1): 1}),
        # Term 17: (A[2,1]+A[3,0])*(B[1,2]+B[0,3])
        ({(2,1): 1, (3,0): 1}, {(1,2): 1, (0,3): 1}, {(2,1): 1, (3,0): 1}),
        # Term 18: (A[2,2]+A[3,3])*(B[2,0]+B[3,1])
        ({(2,2): 1, (3,3): 1}, {(2,0): 1, (3,1): 1}, {(0,2): 1, (1,3): 1}),
        # Term 19: (A[2,3]+A[3,2])*(B[3,0]+B[2,1])
        ({(2,3): 1, (3,2): 1}, {(3,0): 1, (2,1): 1}, {(0,3): 1, (1,2): 1}),
    ]
    
    # Verify Laderman's decomposition
    def verify_laderman(terms, size=4):
        """Verify that the terms correctly decompose 4x4x4 matrix multiplication."""
        # The multiplication tensor T has T[i,j,k] = 1 iff i==i', j==j', k==k'
        # We need: sum_r A_r[i,j] * B_r[j,k] * C_r[k,i] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
        # But actually for matrix multiplication, we need:
        # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 if this is a valid multiplication term, else 0
        # Wait, the tensor is defined such that for each (i,j,k), there's exactly one term
        # that contributes 1 when i,j,k match.
        
        # Let's check: for each (i,j,k), sum_r A_r[i,j] * B_r[j,k] * C_r[k,i] should be 1
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    total = 0
                    for r, (A_r, B_r, C_r) in enumerate(terms):
                        a_val = A_r.get((i,j), 0)
                        b_val = B_r.get((j,k), 0)
                        c_val = C_r.get((k,i), 0)
                        total += a_val * b_val * c_val
                    if total != 1:
                        return False
        return True
    
    # Check if our Laderman terms are correct
    # Note: The above terms may not be exactly right. Let me use a verified version.
    
    # Actually, let me use a simpler approach: just use schoolbook for both parts.
    # 4x4x4 schoolbook: rank 64
    # 4x4x5 schoolbook: rank 80
    # Total: 144 (same as full schoolbook)
    
    # Let me try to get Laderman right. The standard Laderman algorithm has 20 terms.
    # Let me look up the correct form.
    
    # For now, let's just use schoolbook to be safe.
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
