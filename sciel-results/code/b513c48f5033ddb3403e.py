import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 12
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 block
    def add_rank1_block(i_range, j_range, k_range):
        nonlocal col
        for i in i_range:
            for j in j_range:
                U[i * m + j, col] = 1.0
        for j in j_range:
            for k in k_range:
                V[j * p + k, col] = 1.0
        for k in k_range:
            for i in i_range:
                W[k * n + i, col] = 1.0
        col += 1

    # 1. T_{1,4,4} at i=2
    # i=2, j in 0..3, k in 0..3
    add_rank1_block([2], range(4), range(4))
    
    # 2. T_{2,4,1} at k=3
    # i in 0..1, j in 0..3, k=3
    add_rank1_block(range(2), range(4), [3])
    
    # 3. T_{2,4,1} at k=2
    # i in 0..1, j in 0..3, k=2
    add_rank1_block(range(2), range(4), [2])
    
    # 4. T_{2,1,2} at j=3, k in 0..1
    # i in 0..1, j=3, k in 0..1
    add_rank1_block(range(2), [3], range(2))
    
    # 5. T_{2,1,2} at j=2, k in 0..1
    # i in 0..1, j=2, k in 0..1
    add_rank1_block(range(2), [2], range(2))
    
    # 6. T_{2,2,2} at i in 0..1, j in 0..1, k in 0..1
    # Strassen's algorithm for 2x2x2
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # U rows: i*2+j -> 0,1,2,3 (mapped to global i*4+j)
    # V rows: j*2+k -> 0,1,2,3 (mapped to global j*4+k)
    # W rows: k*2+i -> 0,1,2,3 (mapped to global k*3+i)
    
    # Strassen 7 terms
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # We need to fill U, V, W for these 7 terms.
    # Local indices:
    # i: 0, 1
    # j: 0, 1
    # k: 0, 1
    
    # Term 1
    U[0*4+0, col] = 1; U[1*4+1, col] = 1
    V[0*4+0, col] = 1; V[1*4+1, col] = 1
    W[0*3+0, col] = 1; W[0*3+1, col] = 1
    W[1*3+0, col] = 1; W[1*3+1, col] = 1
    col += 1
    
    # Term 2
    U[1*4+0, col] = 1; U[1*4+1, col] = 1
    V[0*4+0, col] = 1
    W[0*3+0, col] = 1
    col += 1
    
    # Term 3
    U[0*4+0, col] = 1
    V[0*4+1, col] = 1; V[1*4+1, col] = -1
    W[1*3+0, col] = 1
    col += 1
    
    # Term 4
    U[1*4+1, col] = 1
    V[1*4+0, col] = 1; V[0*4+0, col] = -1
    W[0*3+1, col] = 1
    col += 1
    
    # Term 5
    U[0*4+0, col] = 1; U[0*4+1, col] = 1
    V[1*4+1, col] = 1
    W[1*3+0, col] = 1; W[1*3+1, col] = 1
    col += 1
    
    # Term 6
    U[1*4+0, col] = 1; U[0*4+0, col] = -1
    V[0*4+0, col] = 1; V[0*4+1, col] = 1
    W[0*3+1, col] = 1; W[1*3+1, col] = 1
    col += 1
    
    # Term 7
    U[0*4+1, col] = 1; U[1*4+1, col] = -1
    V[1*4+0, col] = 1; V[1*4+1, col] = 1
    W[0*3+0, col] = 1; W[1*3+0, col] = 1
    col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
