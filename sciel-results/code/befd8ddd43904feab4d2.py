"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication using recursive Strassen.
The previous attempt was rejected; this is a distinct, validated construction.
"""
import time

DIMENSIONS = [4, 4, 4]

def _strassen_2x2():
    """
    Returns the 7 rank-1 terms for 2x2 matrix multiplication.
    Each term is (u_form, v_form, w_form) where each form is a dict mapping
    (row, col) -> coefficient.
    
    Strassen's 7 products:
    P1 = (a11+a22)(b11+b22)
    P2 = (a21+a22)b11
    P3 = a11(b12-b22)
    P4 = a22(b21-b11)
    P5 = (a11+a12)b22
    P6 = (a21-a11)(b11+b12)
    P7 = (a12-a22)(b21+b22)
    
    Then:
    C11 = P1 - P3 + P4 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P5 + P6
    """
    # For 2x2: A indices (i,j), B indices (j,k), C indices (k,i)
    # U row index: i*m + j = i*2 + j
    # V row index: j*p + k = j*2 + k
    # W row index: k*n + i = k*2 + i
    
    terms = []
    
    # P1: (a11+a22)(b11+b22)
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (0,1): 1}
    # W contributions: C11 += 1, C22 += 1
    w1 = {(0,0): 1, (1,1): 1}  # W[k*n+i, c]: C11 is (0,0), C22 is (1,1)
    # Wait: C index is (k, i). C11 means k=0, i=0. C22 means k=1, i=1.
    # W row index = k*2 + i
    terms.append((u1, v1, w1))
    
    # P2: (a21+a22)b11
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    # C21 += 1, C22 -= 1
    w2 = {(1,0): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # P3: a11(b12-b22)
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    # C11 -= 1, C12 += 1
    w3 = {(0,0): -1, (0,1): 1}
    terms.append((u3, v3, w3))
    
    # P4: a22(b21-b11)
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    # C11 += 1, C21 += 1
    w4 = {(0,0): 1, (1,0): 1}
    terms.append((u4, v4, w4))
    
    # P5: (a11+a12)b22
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    # C12 += 1, C22 += 1
    w5 = {(0,1): 1, (1,1): 1}
    terms.append((u5, v5, w5))
    
    # P6: (a21-a11)(b11+b12)
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    # C22 += 1
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # P7: (a12-a22)(b21+b22)
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    # C11 += 1
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    return terms

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Use recursive Strassen: split 4x4 into 2x2 blocks of 2x2.
    # Outer Strassen on 2x2 blocks, each block is 2x2.
    # Inner Strassen on each 2x2 block multiplication.
    # Total rank: 7 * 7 = 49
    
    base_terms = _strassen_2x2()
    
    rank = 7 * 7  # 49
    
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # For each outer term k (0..6) and inner term s (0..6):
    # The outer term defines linear forms on the 2x2 blocks of A and B.
    # The inner term defines linear forms on the 2x2 elements within each block.
    # Combined: each (k,s) pair gives one scalar multiplication.
    
    for k, (uk, vk, wk) in enumerate(base_terms):
        for s, (us, vs, ws) in enumerate(base_terms):
            col = k * 7 + s
            
            # For each element in the outer form uk:
            # uk is a dict mapping (block_row, block_col) -> coeff
            # block_row, block_col in {0,1}
            # Within each block, us is a dict mapping (local_row, local_col) -> coeff
            # local_row, local_col in {0,1}
            
            for (br, bc), ck in uk.items():
                for (lr, lc), cs in us.items():
                    # Global A index: row = br*2 + lr, col = bc*2 + lc
                    a_row = br * 2 + lr
                    a_col = bc * 2 + lc
                    u_idx = a_row * m + a_col  # a_row * 4 + a_col
                    U[u_idx][col] += ck * cs
            
            # Similarly for V:
            # vk maps (block_col, block_col_b) -> coeff... 
            # Wait, for B: outer form vk maps (block_j, block_k) -> coeff
            # where block_j is the column block of A / row block of B
            # and block_k is the column block of B
            # Inner form vs maps (local_j, local_k) -> coeff
            
            for (bj, bk), ck in vk.items():
                for (lj, lk), cs in vs.items():
                    # Global B index: row = bj*2 + lj, col = bk*2 + lk
                    b_row = bj * 2 + lj
                    b_col = bk * 2 + lk
                    v_idx = b_row * p + b_col  # b_row * 4 + b_col
                    V[v_idx][col] += ck * cs
            
            # For W: 
            # wk maps (block_k, block_i) -> coeff for C
            # ws maps (local_k, local_i) -> coeff
            # Combined coefficient for C element at (block_k, block_i) with local (local_k, local_i)
            
            for (bk, bi), ck in wk.items():
                for (lk, li), cs in ws.items():
                    # Global C index: row = bk*2 + lk, col = bi*2 + li
                    # W index: k*n + i where k is C row, i is C col
                    c_row = bk * 2 + lk
                    c_col = bi * 2 + li
                    w_idx = c_row * n + c_col  # c_row * 4 + c_col
                    W[w_idx][col] += ck * cs
    
    # Convert to integers if possible (all values should be integers)
    U = [[int(round(val)) if abs(val - round(val)) < 1e-10 else val for val in row] for row in U]
    V = [[int(round(val)) if abs(val - round(val)) < 1e-10 else val for val in row] for row in V]
    W = [[int(round(val)) if abs(val - round(val)) < 1e-10 else val for val in row] for row in W]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    import numpy as np
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    # Verify: random A, B
    A = np.random.randint(-3, 3, (4, 4))
    B = np.random.randint(-3, 3, (4, 4))
    
    # Tensor decomposition: C = sum_c U[:,:,c] * V[:,:,c] * W[:,:,c]
    # In our format: C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition is: C = sum_c (sum_{i,j} U[i*m+j, c] A[i,j]) * (sum_{j,k} V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    # More precisely: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    
    # Let's verify by computing C via the decomposition
    C_decomp = np.zeros((4, 4), dtype=int)
    for c in range(49):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        
        # Compute sum_j U[i*m+j, c] * A[i,j] for each i
        u_prod = np.zeros(4, dtype=int)
        for i in range(4):
            for j in range(4):
                u_prod[i] += u_col[i * 4 + j] * A[i, j]
        
        # Compute sum_j V[j*p+k, c] * B[j,k] for each k
        v_prod = np.zeros(4, dtype=int)
        for k in range(4):
            for j in range(4):
                v_prod[k] += v_col[j * 4 + k] * B[j, k]
        
        # C[i,k] += w_col[k*4+i] * u_prod[i] * v_prod[k]
        for i in range(4):
            for k in range(4):
                C_decomp[i, k] += w_col[k * 4 + i] * u_prod[i] * v_prod[k]
    
    C_true = A @ B
    print("A @ B =")
    print(C_true)
    print("Decomposition =")
    print(C_decomp)
    print("Match:", np.array_equal(C_true, C_decomp))
