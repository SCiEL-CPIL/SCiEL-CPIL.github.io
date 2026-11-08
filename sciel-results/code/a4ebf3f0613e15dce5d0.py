import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 50 construction
    # A is 2x4, B is 4x7
    # Split A into A1 (2x2), A2 (2x2)
    # Split B into B1 (2x7), B2 (2x7)
    # B1 split into B11(2x2), B12(2x2), B13(2x2), B14(2x1)
    # B2 split into B21(2x2), B22(2x2), B23(2x2), B24(2x1)
    
    # Strassen for 2x2x2 (Rank 7)
    # Direct for 2x2x1 (Rank 4)
    
    # Total: 6 * 7 + 2 * 4 = 42 + 8 = 50
    
    rank = 50
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    def add_strassen(A_start, B_start, C_start):
        # A_start: index in U for A1 or A2 (0 or 2)
        # B_start: index in V for B block (0, 2, 4 for cols)
        # C_start: index in W for C block (0, 2, 4 for cols)
        # A is 2x2: indices A_start, A_start+1, A_start+2, A_start+3 in U?
        # U index: i*m + j.
        # A1: i=0, j=0,1 -> U[0], U[1]
        #       i=1, j=0,1 -> U[2], U[3]
        # A2: i=0, j=2,3 -> U[4], U[5]
        #       i=1, j=2,3 -> U[6], U[7]
        
        # B is 2x2: indices B_start, B_start+1, B_start+2, B_start+3 in V?
        # V index: j*p + k.
        # B11: j=0, k=0,1 -> V[0], V[1]
        #       j=1, k=0,1 -> V[7], V[8]
        # B12: j=0, k=2,3 -> V[2], V[3]
        #       j=1, k=2,3 -> V[9], V[10]
        
        # C is 2x2: indices C_start, C_start+1, C_start+2, C_start+3 in W?
        # W index: k*n + i.
        # C11: k=0, i=0,1 -> W[0], W[1]
        #       k=1, i=0,1 -> W[2], W[3]
        
        # Strassen terms:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        nonlocal col
        
        # M1
        U[A_start, col] += 1
        U[A_start+2, col] += 1
        V[B_start, col] += 1
        V[B_start+2, col] += 1
        W[C_start, col] += 1
        W[C_start+1, col] += 1
        W[C_start+2, col] += 1
        W[C_start+3, col] -= 1
        col += 1
        
        # M2
        U[A_start+2, col] += 1
        U[A_start+3, col] += 1
        V[B_start, col] += 1
        W[C_start+2, col] += 1
        W[C_start+3, col] -= 1
        col += 1
        
        # M3
        U[A_start, col] += 1
        V[B_start+1, col] += 1
        V[B_start+3, col] -= 1
        W[C_start+1, col] += 1
        W[C_start+3, col] += 1
        col += 1
        
        # M4
        U[A_start+2, col] += 1
        V[B_start+2, col] += 1
        V[B_start, col] -= 1
        W[C_start, col] -= 1
        W[C_start+2, col] += 1
        col += 1
        
        # M5
        U[A_start, col] += 1
        U[A_start+1, col] += 1
        V[B_start+3, col] += 1
        W[C_start, col] -= 1
        W[C_start+1, col] += 1
        col += 1
        
        # M6
        U[A_start+2, col] += 1
        U[A_start, col] -= 1
        V[B_start, col] += 1
        V[B_start+1, col] += 1
        W[C_start+3, col] += 1
        col += 1
        
        # M7
        U[A_start+1, col] += 1
        U[A_start+3, col] -= 1
        V[B_start+2, col] += 1
        V[B_start+3, col] += 1
        W[C_start, col] += 1
        col += 1

    def add_direct(A_start, B_start, C_start):
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = a00 b0 + a01 b1
        # C1 = a10 b0 + a11 b1
        # 4 terms
        nonlocal col
        
        # T1: a00 b0 -> C0
        U[A_start, col] += 1
        V[B_start, col] += 1
        W[C_start, col] += 1
        col += 1
        
        # T2: a01 b1 -> C0
        U[A_start+1, col] += 1
        V[B_start+1, col] += 1
        W[C_start, col] += 1
        col += 1
        
        # T3: a10 b0 -> C1
        U[A_start+2, col] += 1
        V[B_start, col] += 1
        W[C_start+1, col] += 1
        col += 1
        
        # T4: a11 b1 -> C1
        U[A_start+3, col] += 1
        V[B_start+1, col] += 1
        W[C_start+1, col] += 1
        col += 1

    # A1 (rows 0,1 of A, cols 0,1) -> U indices 0,1,2,3
    # A2 (rows 0,1 of A, cols 2,3) -> U indices 4,5,6,7
    
    # B1 (rows 0,1 of B, cols 0,1) -> V indices 0,1,7,8
    # B1 (rows 0,1 of B, cols 2,3) -> V indices 2,3,9,10
    # B1 (rows 0,1 of B, cols 4,5) -> V indices 4,5,11,12
    # B1 (rows 0,1 of B, col 6)    -> V indices 6,13
    
    # B2 (rows 2,3 of B, cols 0,1) -> V indices 14,15,21,22
    # B2 (rows 2,3 of B, cols 2,3) -> V indices 16,17,23,24
    # B2 (rows 2,3 of B, cols 4,5) -> V indices 18,19,25,26
    # B2 (rows 2,3 of B, col 6)    -> V indices 20,27
    
    # C (cols 0,1) -> W indices 0,1,2,3
    # C (cols 2,3) -> W indices 4,5,6,7
    # C (cols 4,5) -> W indices 8,9,10,11
    # C (col 6)    -> W indices 12,13

    # A1 * B11
    add_strassen(0, 0, 0)
    # A1 * B12
    add_strassen(0, 2, 4)
    # A1 * B13
    add_strassen(0, 4, 8)
    # A1 * B14
    add_direct(0, 6, 12)
    
    # A2 * B21
    add_strassen(4, 14, 0)
    # A2 * B22
    add_strassen(4, 16, 4)
    # A2 * B23
    add_strassen(4, 18, 8)
    # A2 * B24
    add_direct(4, 20, 12)
    
    def to_list(mat):
        return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
