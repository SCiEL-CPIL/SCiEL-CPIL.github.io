import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on [2,2,4] blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We split A into A1 (cols 0,1) and A2 (cols 2,3)
    # We split B into Btop (rows 0,1) and Bbot (rows 2,3)
    # C = A1 * Btop + A2 * Bbot
    
    # Strassen for A1 * Btop (rank 14)
    # A1 is 2x2, Btop is 2x4 (split into Btop_left 2x2, Btop_right 2x2)
    # M1..M7 for left part, M8..M14 for right part
    
    # Let's define the 14 factors for A1 * Btop
    # A1 indices in U: 0,1 (row 0), 2,3 (row 1) -> actually U is n*m = 2*4 = 8 rows.
    # U[i*m+j]. 
    # A1 is at i=0,1 and j=0,1. So U rows 0,1,4,5?
    # i=0, j=0 -> 0*4+0=0
    # i=0, j=1 -> 0*4+1=1
    # i=1, j=0 -> 1*4+0=4
    # i=1, j=1 -> 1*4+1=5
    # A2 is at i=0,1 and j=2,3.
    # i=0, j=2 -> 0*4+2=2
    # i=0, j=3 -> 0*4+3=3
    # i=1, j=2 -> 1*4+2=6
    # i=1, j=3 -> 1*4+3=7
    
    # Btop is at j=0,1 (rows of B). V[j*p+k].
    # Btop_left is j=0,1 and k=0,1.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    # Btop_right is j=0,1 and k=2,3.
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 6
    # j=1, k=3 -> 7
    
    # Bbot is at j=2,3.
    # Bbot_left is j=2,3 and k=0,1.
    # j=2, k=0 -> 8
    # j=2, k=1 -> 9
    # j=3, k=0 -> 12
    # j=3, k=1 -> 13
    # Bbot_right is j=2,3 and k=2,3.
    # j=2, k=2 -> 10
    # j=2, k=3 -> 11
    # j=3, k=2 -> 14
    # j=3, k=3 -> 15

    # W indices: k*n+i. p=4, n=2.
    # C is 2x4.
    # C[k, i] ? No, C is 2x4.
    # W[k*n+i, r].
    # k is col of C (0..3), i is row of C (0..1).
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7

    # Strassen factors for A1 * Btop_left (cols 0,1 of C)
    # A1 = [[a00, a01], [a10, a11]]
    # Btop_left = [[b00, b01], [b10, b11]]
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We need to add these to U, V, W.
    # U indices for A1: 0,1,4,5
    # V indices for Btop_left: 0,1,4,5
    # W indices for C00..C11: 0,2,1,3 (k=0,1; i=0,1)
    # W[0] is C[0,0] (k=0, i=0)
    # W[1] is C[0,1] (k=0, i=1)
    # W[2] is C[1,0] (k=1, i=0)
    # W[3] is C[1,1] (k=1, i=1)
    
    # Let's implement a helper to add a Strassen block
    def add_strassen_block(u_rows, v_rows, w_rows, offset):
        # u_rows: [u00, u01, u10, u11]
        # v_rows: [v00, v01, v10, v11]
        # w_rows: [w00, w01, w10, w11]
        # offset: column index in U, V, W
        
        # M1
        U[u_rows[0]][offset] += 1
        U[u_rows[3]][offset] += 1
        V[v_rows[0]][offset] += 1
        V[v_rows[3]][offset] += 1
        W[w_rows[0]][offset] += 1
        W[w_rows[3]][offset] += 1
        
        # M2
        U[u_rows[2]][offset] += 1
        U[u_rows[3]][offset] += 1
        V[v_rows[0]][offset] += 1
        W[w_rows[2]][offset] += 1
        
        # M3
        U[u_rows[0]][offset] += 1
        V[v_rows[1]][offset] += 1
        V[v_rows[3]][offset] -= 1
        W[w_rows[1]][offset] += 1
        
        # M4
        U[u_rows[3]][offset] += 1
        V[v_rows[2]][offset] += 1
        V[v_rows[0]][offset] -= 1
        W[w_rows[0]][offset] += 1
        W[w_rows[2]][offset] += 1
        
        # M5
        U[u_rows[0]][offset] += 1
        U[u_rows[1]][offset] += 1
        V[v_rows[3]][offset] += 1
        W[w_rows[0]][offset] -= 1
        W[w_rows[1]][offset] += 1
        
        # M6
        U[u_rows[2]][offset] += 1
        U[u_rows[0]][offset] -= 1
        V[v_rows[0]][offset] += 1
        V[v_rows[1]][offset] += 1
        W[w_rows[3]][offset] += 1
        
        # M7
        U[u_rows[1]][offset] += 1
        U[u_rows[3]][offset] -= 1
        V[v_rows[2]][offset] += 1
        V[v_rows[3]][offset] += 1
        W[w_rows[0]][offset] += 1

    # Block 1: A1 * Btop_left
    u1 = [0, 1, 4, 5]
    v1 = [0, 1, 4, 5]
    w1 = [0, 1, 2, 3]
    add_strassen_block(u1, v1, w1, 0)
    
    # Block 2: A1 * Btop_right
    u2 = [0, 1, 4, 5]
    v2 = [2, 3, 6, 7]
    w2 = [4, 5, 6, 7]
    add_strassen_block(u2, v2, w2, 7)
    
    # Block 3: A2 * Bbot_left
    u3 = [2, 3, 6, 7]
    v3 = [8, 9, 12, 13]
    w3 = [0, 1, 2, 3]
    add_strassen_block(u3, v3, w3, 14)
    
    # Block 4: A2 * Bbot_right
    u4 = [2, 3, 6, 7]
    v4 = [10, 11, 14, 15]
    w4 = [4, 5, 6, 7]
    add_strassen_block(u4, v4, w4, 21)

    return {"U": U, "V": V, "W": W}
