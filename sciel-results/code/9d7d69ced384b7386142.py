import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = 3, 4, 5
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U = np.zeros(n * m)
        V = np.zeros(m * p)
        W = np.zeros(p * n)
        U[u_idx] = u_val
        V[v_idx] = v_val
        W[w_idx] = w_val
        U_parts.append(U)
        V_parts.append(V)
        W_parts.append(W)

    # Block 1: A[0:2, :] * B[:, 0:2] -> C[0:2, 0:2]
    # A is 2x4, B is 4x2, C is 2x2.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # Split B into B1 (rows 0,1) and B2 (rows 2,3).
    # C = A1 B1 + A2 B2.
    
    # A1 B1 (2x2x2)
    # A indices: 0,1 (row 0), 4,5 (row 1)
    # B indices: 0,1 (row 0), 2,3 (row 1) -> Wait, B is 4x2.
    # B row 0: 0,1. B row 1: 2,3.
    # C indices: 0,3 (col 0), 1,4 (col 1)
    
    # Term 1: A[0,0] B[0,0] -> C[0,0]
    add_term(0, 1, 0, 1, 0, 1)
    # Term 2: A[0,0] B[0,1] -> C[0,1]
    add_term(0, 1, 1, 1, 3, 1)
    # Term 3: A[1,0] B[0,0] -> C[1,0]
    add_term(4, 1, 0, 1, 1, 1)
    # Term 4: A[1,0] B[0,1] -> C[1,1]
    add_term(4, 1, 1, 1, 4, 1)
    
    # Term 5: A[0,1] B[1,0] -> C[0,0]
    add_term(1, 1, 2, 1, 0, 1)
    # Term 6: A[0,1] B[1,1] -> C[0,1]
    add_term(1, 1, 3, 1, 3, 1)
    # Term 7: A[1,1] B[1,0] -> C[1,0]
    add_term(5, 1, 2, 1, 1, 1)
    # Term 8: A[1,1] B[1,1] -> C[1,1]
    add_term(5, 1, 3, 1, 4, 1)

    # A2 B2 (2x2x2)
    # A indices: 2,3 (row 0), 6,7 (row 1)
    # B indices: 4,5 (row 2), 6,7 (row 3)
    
    # Term 9: A[0,2] B[2,0] -> C[0,0]
    add_term(2, 1, 4, 1, 0, 1)
    # Term 10: A[0,2] B[2,1] -> C[0,1]
    add_term(2, 1, 5, 1, 3, 1)
    # Term 11: A[1,2] B[2,0] -> C[1,0]
    add_term(6, 1, 4, 1, 1, 1)
    # Term 12: A[1,2] B[2,1] -> C[1,1]
    add_term(6, 1, 5, 1, 4, 1)
    
    # Term 13: A[0,3] B[3,0] -> C[0,0]
    add_term(3, 1, 6, 1, 0, 1)
    # Term 14: A[0,3] B[3,1] -> C[0,1]
    add_term(3, 1, 7, 1, 3, 1)
    # Term 15: A[1,3] B[3,0] -> C[1,0]
    add_term(7, 1, 6, 1, 1, 1)
    # Term 16: A[1,3] B[3,1] -> C[1,1]
    add_term(7, 1, 7, 1, 4, 1)
    
    # Block 2: A[0:2, :] * B[:, 2:4] -> C[0:2, 2:4]
    # A is 2x4, B is 4x2, C is 2x2.
    # B indices:
    # Row 0: 4,5 (cols 2,3 of B? No. B is 4x5.
    # B col 0: 0,4,8,12.
    # B col 1: 1,5,9,13.
    # B col 2: 2,6,10,14.
    # B col 3: 3,7,11,15.
    # B col 4: 4,8,12,16.
    
    # Wait, my previous index mapping for B was:
    # B is 4x5.
    # Index in V: row * 5 + col.
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # Row 2: 10,11,12,13,14
    # Row 3: 15,16,17,18,19
    
    # So for B[:, 0:2] (cols 0,1):
    # Row 0: 0,1
    # Row 1: 5,6
    # Row 2: 10,11
    # Row 3: 15,16
    
    # My previous Block 1 used:
    # B1 (rows 0,1) cols 0,1:
    # Row 0: 0,1. Correct.
    # Row 1: 5,6. 
    # In Block 1 above, I used 2,3 for Row 1. That is WRONG.
    # Row 1 of B is indices 5,6,7,8,9.
    # Col 0 is 5, Col 1 is 6.
    
    # I need to redo the indices carefully.
    
    pass
