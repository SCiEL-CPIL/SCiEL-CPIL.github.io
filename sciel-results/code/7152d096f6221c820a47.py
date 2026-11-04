import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 114 decomposition
    # 49 from Strassen for 4x4x4x4 (cols 0-3)
    # 49 from Strassen for 4x4x4x4 (cols 4-7)
    # 16 from Schoolbook for 4x4x4x1 (col 8)
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get 2x2 Strassen factors
    def get_strassen_2x2():
        # U, V, W are 4x7
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # P1
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        
        # P2
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[1][1] = 1; W2[3][1] = 1
        
        # P3
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[2][2] = 1; W2[3][2] = -1
        
        # P4
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[0][3] = 1; W2[1][3] = 1
        
        # P5
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[0][4] = -1; W2[2][4] = 1
        
        # P6
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[3][5] = 1
        
        # P7
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[0][6] = 1; W2[3][6] = 1
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_2x2()
    
    # Build 4x4 Strassen factors (49 terms)
    # U4, V4, W4 are 16x49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c4 = c1 * 7 + c2
            for i0 in range(2):
                for j0 in range(2):
                    for i1 in range(2):
                        for j1 in range(2):
                            u_idx4 = (i0 * 2 + i1) * 4 + (j0 * 2 + j1)
                            u_val = U2[i0 * 2 + j0][c1] * U2[i1 * 2 + j1][c2]
                            U4[u_idx4][c4] = u_val
                            
                            v_idx4 = (j0 * 2 + j1) * 4 + (i0 * 2 + i1) # Wait, V indices are j,k.
                            # For 4x4, V rows are j (0..3), k (0..3).
                            # j = j0*2 + j1? No, j is the column of A, row of B.
                            # In 4x4, j ranges 0..3.
                            # Let's stick to the Kronecker structure.
                            # V4[j, k, c] = V2[j0, k0, c1] * V2[j1, k1, c2]
                            # j = j0*2 + j1, k = k0*2 + k1
                            # We need to fill V4 for all j, k.
                            pass
                            
    # Simpler: just fill V4 and W4 similarly
    for c1 in range(7):
        for c2 in range(7):
            c4 = c1 * 7 + c2
            for j0 in range(2):
                for k0 in range(2):
                    for j1 in range(2):
                        for k1 in range(2):
                            v_idx4 = (j0 * 2 + j1) * 4 + (k0 * 2 + k1)
                            v_val = V2[j0 * 2 + k0][c1] * V2[j1 * 2 + k1][c2]
                            V4[v_idx4][c4] = v_val
                            
                            w_idx4 = (k0 * 2 + k1) * 4 + (i0 * 2 + i1) # Wait, W indices are k,i.
                            # We need i0, i1 here too.
                            # Let's restructure the loop to include i.
                            pass

    # Correct loop for Kronecker product
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c4 = c1 * 7 + c2
            for i0 in range(2):
                for j0 in range(2):
                    for k0 in range(2):
                        for i1 in range(2):
                            for j1 in range(2):
                                for k1 in range(2):
                                    # U
                                    u_idx4 = (i0 * 2 + i1) * 4 + (j0 * 2 + j1)
                                    u_val = U2[i0 * 2 + j0][c1] * U2[i1 * 2 + j1][c2]
                                    U4[u_idx4][c4] = u_val
                                    
                                    # V
                                    v_idx4 = (j0 * 2 + j1) * 4 + (k0 * 2 + k1)
                                    v_val = V2[j0 * 2 + k0][c1] * V2[j1 * 2 + k1][c2]
                                    V4[v_idx4][c4] = v_val
                                    
                                    # W
                                    w_idx4 = (k0 * 2 + k1) * 4 + (i0 * 2 + i1)
                                    w_val = W2[k0 * 2 + i0][c1] * W2[k1 * 2 + i1][c2]
                                    W4[w_idx4][c4] = w_val

    # Fill U, V, W for the two 4x4x4x4 blocks
    # Block 1: k in 0..3
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            for c in range(49):
                U[u_idx][c] = U4[u_idx][c]
                
    for j in range(4):
        for k in range(4):
            v_idx = j * 9 + k
            for c in range(49):
                V[v_idx][c] = V4[v_idx][c]
                
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            for c in range(49):
                W[w_idx][c] = W4[w_idx][c]
                
    # Block 2: k in 4..7
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            for c in range(49):
                U[u_idx][c + 49] = U4[u_idx][c]
                
    for j in range(4):
        for k in range(4):
            v_idx = j * 9 + (k + 4)
            for c in range(49):
                V[v_idx][c + 49] = V4[v_idx][c]
                
    for k in range(4):
        for i in range(4):
            w_idx = (k + 4) * 4 + i
            for c in range(49):
                W[w_idx][c + 49] = W4[w_idx][c]

    # Block 3: k = 8 (Schoolbook)
    col_offset = 98
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            c = col_offset + (i * 4 + j)
            U[u_idx][c] = 1
            V[v_idx][c] = 1
            W[w_idx][c] = 1

    return {"U": U, "V": V, "W": W}

def _schoolbook_solution(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                if total != 1:
                    return False
    return True

# EVOLVE-BLOCK-END
