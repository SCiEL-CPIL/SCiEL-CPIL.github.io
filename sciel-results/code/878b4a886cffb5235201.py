import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U: (n*m) x rank = 8 x rank
    # V: (m*p) x rank = 16 x rank
    # W: (p*n) x rank = 8 x rank
    
    # We use 4 Strassen blocks of rank 7
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions
    # X indices: 00, 01, 10, 11
    # Y indices: 00, 01, 10, 11
    # Products:
    # 1: (X00+X11) * (Y00+Y11)
    # 2: (X10+X11) * Y00
    # 3: X00 * (Y01-Y11)
    # 4: X11 * (Y10-Y00)
    # 5: (X00+X01) * Y11
    # 6: (X10-X00) * (Y00+Y01)
    # 7: (X01-X11) * (Y10+Y11)
    
    # W contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Instance 1: A1 (cols 0,1) * B11 (rows 0,1 cols 0,1)
    # A1 is X, B11 is Y
    # A indices in U: i*4 + j. A1: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B indices in V: j*4 + k. B11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # C1 is left part of C (cols 0,1). C is 2x4.
    # C indices in W: k*n + i.
    # C00 (i=0, k=0) -> W[0]
    # C01 (i=0, k=1) -> W[2]
    # C10 (i=1, k=0) -> W[1]
    # C11 (i=1, k=1) -> W[3]
    
    def add_strassen_block(u_map, v_map, w_map, base_col):
        # u_map: dict {(0,0): row, (0,1): row, (1,0): row, (1,1): row}
        # v_map: dict {(0,0): row, (0,1): row, (1,0): row, (1,1): row}
        # w_map: dict {(0,0): row, (0,1): row, (1,0): row, (1,1): row}
        
        # M1
        U[u_map[(0,0)]][base_col + 0] += 1
        U[u_map[(1,1)]][base_col + 0] += 1
        V[v_map[(0,0)]][base_col + 0] += 1
        V[v_map[(1,1)]][base_col + 0] += 1
        W[w_map[(0,0)]][base_col + 0] += 1
        W[w_map[(1,1)]][base_col + 0] += 1
        
        # M2
        U[u_map[(1,0)]][base_col + 1] += 1
        U[u_map[(1,1)]][base_col + 1] += 1
        V[v_map[(0,0)]][base_col + 1] += 1
        W[w_map[(1,0)]][base_col + 1] += 1
        W[w_map[(1,1)]][base_col + 1] -= 1
        
        # M3
        U[u_map[(0,0)]][base_col + 2] += 1
        V[v_map[(0,1)]][base_col + 2] += 1
        V[v_map[(1,1)]][base_col + 2] -= 1
        W[w_map[(0,1)]][base_col + 2] += 1
        W[w_map[(1,1)]][base_col + 2] += 1
        
        # M4
        U[u_map[(1,1)]][base_col + 3] += 1
        V[v_map[(1,0)]][base_col + 3] += 1
        V[v_map[(0,0)]][base_col + 3] -= 1
        W[w_map[(0,0)]][base_col + 3] += 1
        W[w_map[(1,0)]][base_col + 3] += 1
        
        # M5
        U[u_map[(0,0)]][base_col + 4] += 1
        U[u_map[(0,1)]][base_col + 4] += 1
        V[v_map[(1,1)]][base_col + 4] += 1
        W[w_map[(0,0)]][base_col + 4] -= 1
        W[w_map[(0,1)]][base_col + 4] += 1
        
        # M6
        U[u_map[(1,0)]][base_col + 5] += 1
        U[u_map[(0,0)]][base_col + 5] -= 1
        V[v_map[(0,0)]][base_col + 5] += 1
        V[v_map[(0,1)]][base_col + 5] += 1
        W[w_map[(1,1)]][base_col + 5] += 1
        
        # M7
        U[u_map[(0,1)]][base_col + 6] += 1
        U[u_map[(1,1)]][base_col + 6] -= 1
        V[v_map[(1,0)]][base_col + 6] += 1
        V[v_map[(1,1)]][base_col + 6] += 1
        W[w_map[(0,0)]][base_col + 6] += 1

    # Instance 1: A1 * B11 -> C1_left (cols 0,1 of C)
    # A1: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    u1 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    # B11: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    v1 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    # C1_left: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    w1 = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
    add_strassen_block(u1, v1, w1, 0)
    
    # Instance 2: A2 * B21 -> C1_right (cols 2,3 of C? No, C1 is cols 0,1)
    # Wait, C = A B.
    # A = [A1, A2]. B = [[B11, B12], [B21, B22]].
    # C1 (cols 0,1) = A1 B11 + A2 B21.
    # C2 (cols 2,3) = A1 B12 + A2 B22.
    
    # So Instance 2 is A2 * B21 contributing to C1 (cols 0,1).
    # A2: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    u2 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    # B21: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    v2 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
    # C1 (cols 0,1) is same as w1
    w2 = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
    add_strassen_block(u2, v2, w2, 7)
    
    # Instance 3: A1 * B12 -> C2 (cols 2,3)
    # A1 is same as u1
    u3 = u1
    # B12: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    v3 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    # C2 (cols 2,3): C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    w3 = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
    add_strassen_block(u3, v3, w3, 14)
    
    # Instance 4: A2 * B22 -> C2 (cols 2,3)
    # A2 is same as u2
    u4 = u2
    # B22: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    v4 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
    # C2 is same as w3
    w4 = w3
    add_strassen_block(u4, v4, w4, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
