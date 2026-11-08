"""
Rewritten program for matrix multiplication 2x4 by 4x5.
Uses a rank-36 decomposition based on 4 Strassen blocks (28 mults) 
and direct multiplication for the 5th column (8 mults).
"""

def construct():
    n, m, p = 2, 4, 5
    
    # Rank is 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    def add_term(u_row, v_row, w_row, val=1):
        nonlocal col
        U[u_row][col] += val
        V[v_row][col] += val
        W[w_row][col] += val
        col += 1

    def strassen_block(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        """
        Strassen's 7-multiplication algorithm for 2x2x2 block.
        Indices are pre-calculated row indices for U, V, W.
        """
        # M1 = (a00+a11)*(b00+b11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3 = a00*(b01-b11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = a11*(b10-b00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    # We decompose the multiplication into 4 Strassen blocks for the first 4 columns of B
    # and 8 direct multiplications for the 5th column.
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2] contribution
    strassen_block(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    )
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4] contribution
    strassen_block(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3),
        w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)
    )
    
    # Block 3: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2] contribution
    strassen_block(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    )
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4] contribution
    strassen_block(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3),
        w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)
    )
    
    # Block 5: Direct multiplication for the 5th column (index 4)
    # C[:, 4] = A[:, 0:2] * B[0:2, 4] + A[:, 2:4] * B[2:4, 4]
    
    # A[0,0] * B[0,4] -> C[0,4]
    U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    # A[0,1] * B[1,4] -> C[0,4]
    U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    # A[1,0] * B[0,4] -> C[1,4]
    U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    # A[1,1] * B[1,4] -> C[1,4]
    U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    
    # A[0,2] * B[2,4] -> C[0,4]
    U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    # A[0,3] * B[3,4] -> C[0,4]
    U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    # A[1,2] * B[2,4] -> C[1,4]
    U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    # A[1,3] * B[3,4] -> C[1,4]
    U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    
    assert col == rank, f"Expected {rank} but got {col}"
    return {"U": U, "V": V, "W": W}
