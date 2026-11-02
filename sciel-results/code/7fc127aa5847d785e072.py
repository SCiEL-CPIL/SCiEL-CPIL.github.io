"""Rank-minimal tensor decomposition for 2x2 * 2x4 matrix multiplication."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns the rank-minimal tensor decomposition for 2x2 * 2x4 matrix multiplication.
    The minimal rank is 14.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    R = 14
    
    # Initialize factor matrices
    U = [[0.0]*R for _ in range(nm)]
    V = [[0.0]*R for _ in range(mp)]
    W = [[0.0]*R for _ in range(pn)]
    
    # Strassen's 7 products for 2x2 matrix multiplication.
    # Indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # We define the 7 Strassen products for a generic 2x2 * 2x2 multiplication.
    # Let B_left be columns 0,1 of the 2x4 matrix.
    # Let B_right be columns 2,3 of the 2x4 matrix.
    
    # Product definitions for one block:
    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    # M2 = (a10+a11)(b00)     -> C10 += M2, C11 -= M2
    # M3 = a00(b01-b11)       -> C01 += M3, C11 += M3
    # M4 = a11(b10-b00)       -> C00 += M4, C10 += M4
    # M5 = (a00+a01)(b11)     -> C00 -= M5, C01 += M5
    # M6 = (a10-a00)(b00+b01) -> C10 += M6
    # M7 = (a01-a11)(b10+b11) -> C00 += M7
    
    # We map these to the global indices.
    # U indices (A): 0:a00, 1:a01, 2:a10, 3:a11
    # V indices (B_left): 0:b00, 1:b01, 2:b10, 3:b11
    # V indices (B_right): 4:b02, 5:b03, 6:b12, 7:b13
    # W indices (C_left): 0:c00, 1:c01, 2:c10, 3:c11
    # W indices (C_right): 4:c02, 5:c03, 6:c12, 7:c13
    
    # Helper to add a Strassen block
    def add_strassen_block(offset_u, offset_v, offset_w):
        """
        offset_u: offset for U indices (always 0 for A)
        offset_v: offset for V indices (0 for left, 4 for right)
        offset_w: offset for W indices (0 for left, 4 for right)
        """
        # M1
        U[0][offset_u] += 1; U[3][offset_u] += 1
        V[0+offset_v][offset_u] += 1; V[2+offset_v][offset_u] += 1
        W[0+offset_w][offset_u] += 1; W[3+offset_w][offset_u] += 1
        
        # M2
        U[2][offset_u+1] += 1; U[3][offset_u+1] += 1
        V[0+offset_v][offset_u+1] += 1
        W[2+offset_w][offset_u+1] += 1; W[3+offset_w][offset_u+1] -= 1
        
        # M3
        U[0][offset_u+2] += 1
        V[1+offset_v][offset_u+2] += 1; V[3+offset_v][offset_u+2] -= 1
        W[1+offset_w][offset_u+2] += 1; W[3+offset_w][offset_u+2] += 1
        
        # M4
        U[3][offset_u+3] += 1
        V[2+offset_v][offset_u+3] += 1; V[0+offset_v][offset_u+3] -= 1
        W[0+offset_w][offset_u+3] += 1; W[2+offset_w][offset_u+3] += 1
        
        # M5
        U[0][offset_u+4] += 1; U[1][offset_u+4] += 1
        V[3+offset_v][offset_u+4] += 1
        W[0+offset_w][offset_u+4] -= 1; W[1+offset_w][offset_u+4] += 1
        
        # M6
        U[2][offset_u+5] += 1; U[0][offset_u+5] -= 1
        V[0+offset_v][offset_u+5] += 1; V[1+offset_v][offset_u+5] += 1
        W[2+offset_w][offset_u+5] += 1
        
        # M7
        U[1][offset_u+6] += 1; U[3][offset_u+6] -= 1
        V[2+offset_v][offset_u+6] += 1; V[3+offset_v][offset_u+6] += 1
        W[0+offset_w][offset_u+6] += 1

    # Add left block (products 0-6)
    add_strassen_block(0, 0, 0)
    
    # Add right block (products 7-13)
    add_strassen_block(7, 4, 4)
    
    # Convert to integers if they are close to integers
    for i in range(nm):
        for r in range(R):
            if abs(U[i][r] - round(U[i][r])) < 1e-9:
                U[i][r] = int(round(U[i][r]))
    for j in range(mp):
        for r in range(R):
            if abs(V[j][r] - round(V[j][r])) < 1e-9:
                V[j][r] = int(round(V[j][r]))
    for k in range(pn):
        for r in range(R):
            if abs(W[k][r] - round(W[k][r])) < 1e-9:
                W[k][r] = int(round(W[k][r]))
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
