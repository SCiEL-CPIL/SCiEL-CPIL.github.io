"""Strassen-based decomposition for 3x4*4x4 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x4 * 4x4 matrix multiplication using Strassen's algorithm
    on 2x2 blocks for the top 2 rows, and naive multiplication for the bottom row.
    
    Rank: 4 blocks * 7 Strassen terms + 16 naive terms = 28 + 16 = 44
    """
    n, m, p = 3, 4, 4
    rank = 44
    
    # U: shape (n*m, rank) = (12, 44)
    # V: shape (m*p, rank) = (16, 44)
    # W: shape (p*n, rank) = (12, 44)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 terms:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each block has 4 quadrants in A (2x2) and B (2x2)
    # For block at A rows [ar0:ar0+2], cols [ac0:ac0+2], B rows [br0:br0+2], cols [bc0:bc0+2]
    # Result C rows [ar0:ar0+2], cols [bc0:bc0+2]
    
    def strassen_block(ar0, ac0, br0, bc0):
        nonlocal col
        # A00 = A[ar0, ac0], A01 = A[ar0, ac0+1]
        # A10 = A[ar0+1, ac0], A11 = A[ar0+1, ac0+1]
        # B00 = B[br0, bc0], B01 = B[br0, bc0+1]
        # B10 = B[br0+1, bc0], B11 = B[br0+1, bc0+1]
        
        a00 = (ar0, ac0)
        a01 = (ar0, ac0 + 1)
        a10 = (ar0 + 1, ac0)
        a11 = (ar0 + 1, ac0 + 1)
        
        b00 = (br0, bc0)
        b01 = (br0, bc0 + 1)
        b10 = (br0 + 1, bc0)
        b11 = (br0 + 1, bc0 + 1)
        
        c00 = (ar0, bc0)
        c01 = (ar0, bc0 + 1)
        c10 = (ar0 + 1, bc0)
        c11 = (ar0 + 1, bc0 + 1)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00[0]*m + a00[1]][col] += 1.0
        U[a11[0]*m + a11[1]][col] += 1.0
        V[b00[0]*p + b00[1]][col] += 1.0
        V[b11[0]*p + b11[1]][col] += 1.0
        W[c00[0]*n + c00[1]][col] += 1.0
        W[c01[0]*n + c01[1]][col] += 1.0
        W[c10[0]*n + c10[1]][col] += 1.0
        W[c11[0]*n + c11[1]][col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10[0]*m + a10[1]][col] += 1.0
        U[a11[0]*m + a11[1]][col] += 1.0
        V[b00[0]*p + b00[1]][col] += 1.0
        V[b00[0]*p + b00[1]][col] += 0  # already set, just using same V entry
        W[c10[0]*n + c10[1]][col] += 1.0
        W[c11[0]*n + c11[1]][col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00[0]*m + a00[1]][col] += 1.0
        V[b01[0]*p + b01[1]][col] += 1.0
        V[b11[0]*p + b11[1]][col] -= 1.0
        W[c01[0]*n + c01[1]][col] += 1.0
        W[c11[0]*n + c11[1]][col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11[0]*m + a11[1]][col] += 1.0
        V[b10[0]*p + b10[1]][col] += 1.0
        V[b00[0]*p + b00[1]][col] -= 1.0
        W[c00[0]*n + c00[1]][col] += 1.0
        W[c10[0]*n + c10[1]][col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00[0]*m + a00[1]][col] += 1.0
        U[a01[0]*m + a01[1]][col] += 1.0
        V[b11[0]*p + b11[1]][col] += 1.0
        W[c00[0]*n + c00[1]][col] -= 1.0
        W[c01[0]*n + c01[1]][col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10[0]*m + a10[1]][col] += 1.0
        U[a00[0]*m + a00[1]][col] -= 1.0
        V[b00[0]*p + b00[1]][col] += 1.0
        V[b01[0]*p + b01[1]][col] += 1.0
        W[c11[0]*n + c11[1]][col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01[0]*m + a01[1]][col] += 1.0
        U[a11[0]*m + a11[1]][col] -= 1.0
        V[b10[0]*p + b10[1]][col] += 1.0
        V[b11[0]*p + b11[1]][col] += 1.0
        W[c00[0]*n + c00[1]][col] += 1.0
        col += 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_block(0, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen_block(0, 2, 2, 0)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_block(0, 0, 0, 2)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    strassen_block(0, 2, 2, 2)
    
    # Naive for row 2 of A: A[2, j] * B[j, k] -> C[k, 2]
    for j in range(4):
        for k in range(4):
            U[2 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 2][col] = 1.0
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
