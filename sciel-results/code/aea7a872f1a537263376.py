import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    # Helper to add a product term
    def add_term(u_indices, v_indices, w_indices, col_idx):
        # u_indices, v_indices, w_indices are lists of (index, value)
        for idx, val in u_indices:
            U[idx][col_idx] += val
        for idx, val in v_indices:
            V[idx][col_idx] += val
        for idx, val in w_indices:
            W[idx][col_idx] += val

    # Block 1: A1 (2x2) * B1a (2x2) -> C (2x2)
    # A rows 0,1 cols 0,1. B rows 0,1 cols 0,1.
    # C rows 0,1 cols 0,1.
    # W indices: k*n + i
    # C[0,0] -> W[0]
    # C[1,0] -> W[1]
    # C[0,1] -> W[2]
    # C[1,1] -> W[3]
    
    # Strassen P1: (A11+A22)(B11+B22)
    # A11=U[0], A22=U[3]
    # B11=V[0], B22=V[2] (B row 1 col 1 is index 1*5+1=6? No, B is 3x5. Row 0 col 0 is 0. Row 1 col 1 is 1*5+1=6.)
    # Wait, V index is j*p + k.
    # B1a is B[0:2, 0:2].
    # B[0,0] -> V[0]
    # B[0,1] -> V[1]
    # B[1,0] -> V[5]
    # B[1,1] -> V[6]
    
    # Let's map local B indices to global V indices.
    # Local B[0,0] -> Global V[0]
    # Local B[0,1] -> Global V[1]
    # Local B[1,0] -> Global V[5]
    # Local B[1,1] -> Global V[6]
    
    # Local A indices to Global U indices.
    # A is 2x3.
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[3]
    # A[1,1] -> U[4]
    
    # Local C indices to Global W indices.
    # C[0,0] -> W[0]
    # C[1,0] -> W[1]
    # C[0,1] -> W[2]
    # C[1,1] -> W[3]

    # P1
    add_term([(0, 1), (3, 1)], [(0, 1), (6, 1)], [(0, 1), (3, 1)], col)
    col += 1
    
    # P2: (A21+A22)B11
    # A21=U[3], A22=U[4]
    # B11=V[0]
    # C21=W[1], C22=W[3]
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], col)
    col += 1
    
    # P3: A11(B21-B11)
    # A11=U[0]
    # B21=V[5], B11=V[0]
    # C12=W[2], C22=W[3]
    add_term([(0, 1)], [(5, 1), (0, -1)], [(2, 1), (3, 1)], col)
    col += 1
    
    # P4: A22(B12-B22)
    # A22=U[4]
    # B12=V[1], B22=V[6]
    # C11=W[0], C21=W[1]
    add_term([(4, 1)], [(1, 1), (6, -1)], [(0, 1), (1, 1)], col)
    col += 1
    
    # P5: (A11+A12)B22
    # A11=U[0], A12=U[1]
    # B22=V[6]
    # C11=W[0], C12=W[2]
    add_term([(0, 1), (1, 1)], [(6, 1)], [(0, -1), (2, 1)], col)
    col += 1
    
    # P6: (A21-A11)(B11+B12)
    # A21=U[3], A11=U[0]
    # B11=V[0], B12=V[1]
    # C22=W[3]
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], col)
    col += 1
    
    # P7: (A12-A22)(B21+B22)
    # A12=U[1], A22=U[4]
    # B21=V[5], B22=V[6]
    # C11=W[0]
    add_term([(1, 1), (4, -1)], [(5, 1), (6, 1)], [(0, 1)], col)
    col += 1

    # Block 2: A1 (2x2) * B1b (2x2) -> C (2x2)
    # B1b is B[0:2, 2:4].
    # B[0,2] -> V[2]
    # B[0,3] -> V[3]
    # B[1,2] -> V[7]
    # B[1,3] -> V[8]
    # C cols 2,3.
    # C[0,2] -> W[4]
    # C[1,2] -> W[5]
    # C[0,3] -> W[6]
    # C[1,3] -> W[7]
    
    # P1
    add_term([(0, 1), (3, 1)], [(2, 1), (8, 1)], [(4, 1), (7, 1)], col)
    col += 1
    
    # P2
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], col)
    col += 1
    
    # P3
    add_term([(0, 1)], [(7, 1), (2, -1)], [(6, 1), (7, 1)], col)
    col += 1
    
    # P4
    add_term([(4, 1)], [(3, 1), (8, -1)], [(4, 1), (5, 1)], col)
    col += 1
    
    # P5
    add_term([(0, 1), (1, 1)], [(8, 1)], [(4, -1), (6, 1)], col)
    col += 1
    
    # P6
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], col)
    col += 1
    
    # P7
    add_term([(1, 1), (4, -1)], [(7, 1), (8, 1)], [(4, 1)], col)
    col += 1

    # Block 3: A1 (2x2) * B1c (2x1) -> C (2x1)
    # B1c is B[0:2, 4].
    # B[0,4] -> V[4]
    # B[1,4] -> V[9]
    # C col 4.
    # C[0,4] -> W[8]
    # C[1,4] -> W[9]
    
    # Term 1: A[0,0] * B[0,4]
    add_term([(0, 1)], [(4, 1)], [(8, 1)], col)
    col += 1
    
    # Term 2: A[1,0] * B[0,4]
    add_term([(3, 1)], [(4, 1)], [(9, 1)], col)
    col += 1
    
    # Term 3: A[0,1] * B[1,4]
    add_term([(1, 1)], [(9, 1)], [(8, 1)], col)
    col += 1
    
    # Term 4: A[1,1] * B[1,4]
    add_term([(4, 1)], [(9, 1)], [(9, 1)], col)
    col += 1

    # Block 4: A2 (2x1) * B2 (1x5) -> C (2x5)
    # A2 is A[:, 2].
    # A[0,2] -> U[2]
    # A[1,2] -> U[5]
    # B2 is B[2, :].
    # B[2,0] -> V[10]
    # B[2,1] -> V[11]
    # B[2,2] -> V[12]
    # B[2,3] -> V[13]
    # B[2,4] -> V[14]
    
    # i=0
    for k in range(5):
        add_term([(2, 1)], [(10+k, 1)], [(k*2+0, 1)], col)
        col += 1
        
    # i=1
    for k in range(5):
        add_term([(5, 1)], [(10+k, 1)], [(k*2+1, 1)], col)
        col += 1

    return {"U": U, "V": V, "W": W}

