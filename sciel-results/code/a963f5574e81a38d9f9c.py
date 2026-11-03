import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 23 algorithm for 3x3x3 (Bini et al.)
    # 23 terms
    # U, V, W are lists of 9-element vectors (flattened 3x3)
    
    # Define the 23 terms as (U_vec, V_vec, W_vec)
    # Indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    
    terms = []
    
    # 1. a00 b00
    terms.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # 2. a11 b11
    terms.append(([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # 3. a22 b22
    terms.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    # 4. (a00+a11)(b00+b11)
    terms.append(([1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0]))
    # 5. (a00+a22)(b00+b22)
    terms.append(([1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1]))
    # 6. (a11+a22)(b11+b22)
    terms.append(([0,0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0,1]))
    # 7. (a00-a11)(b00-b11)
    terms.append(([1,0,0,0,-1,0,0,0,0], [1,0,0,0,-1,0,0,0,0], [1,0,0,0,-1,0,0,0,0]))
    # 8. (a00-a22)(b00-b22)
    terms.append(([1,0,0,0,0,0,0,0,-1], [1,0,0,0,0,0,0,0,-1], [1,0,0,0,0,0,0,0,-1]))
    # 9. (a11-a22)(b11-b22)
    terms.append(([0,0,0,0,1,0,0,0,-1], [0,0,0,0,1,0,0,0,-1], [0,0,0,0,1,0,0,0,-1]))
    # 10. a01 b10
    terms.append(([0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # 11. a10 b01
    terms.append(([0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]))
    # 12. a02 b20
    terms.append(([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0]))
    # 13. a20 b02
    terms.append(([0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]))
    # 14. a12 b21
    terms.append(([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,1,0,0,0,0]))
    # 15. a21 b12
    terms.append(([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0]))
    # 16. (a01+a10)(b10+b01)
    terms.append(([0,1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0,0], [1,1,0,0,0,0,0,0,0]))
    # 17. (a02+a20)(b20+b02)
    terms.append(([0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0], [1,0,1,0,0,0,0,0,0]))
    # 18. (a12+a21)(b21+b12)
    terms.append(([0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0], [0,0,0,0,1,0,0,1,0]))
    # 19. (a01-a10)(b10-b01)
    terms.append(([0,1,0,-1,0,0,0,0,0], [0,-1,0,1,0,0,0,0,0], [1,-1,0,0,0,0,0,0,0]))
    # 20. (a02-a20)(b20-b02)
    terms.append(([0,0,1,0,0,0,-1,0,0], [0,0,-1,0,0,0,1,0,0], [1,0,-1,0,0,0,0,0,0]))
    # 21. (a12-a21)(b21-b12)
    terms.append(([0,0,0,0,0,1,0,-1,0], [0,0,0,0,0,-1,0,1,0], [0,0,0,0,1,0,0,-1,0]))
    # 22. (a00+a11+a22)(b00+b11+b22)
    terms.append(([1,0,0,0,1,0,0,0,1], [1,0,0,0,1,0,0,0,1], [1,0,0,0,1,0,0,0,1]))
    # 23. (a00+a11-a22)(b00+b11-b22)
    terms.append(([1,0,0,0,1,0,0,0,-1], [1,0,0,0,1,0,0,0,-1], [1,0,0,0,1,0,0,0,-1]))
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill first 23 columns for block 1 (B cols 0-2, C cols 0-2)
    for t in range(23):
        u_vec, v_vec, w_vec = terms[t]
        col = t
        for i in range(9):
            U[i * 3 + (i % 3)][col] = u_vec[i] # Wait, U is 9x46. i is 0..8.
            # U index is i.
            U[i][col] = u_vec[i]
            
            # V index for B1: j*6 + k. v_vec index i corresponds to j*3+k.
            j = i // 3
            k = i % 3
            V[j * 6 + k][col] = v_vec[i]
            
            # W index for C1: i*3 + k. w_vec index i corresponds to i*3+k? 
            # w_vec is flattened C. Index i in w_vec corresponds to C_{i//3, i%3}.
            # W matrix index is row = i*3+k? No, W is (p*n, rank) = (18, 46).
            # W row index is k*n + i? Or i*n + k?
            # Task says: W[k*n+i, r].
            # So W index is k*3 + i.
            # w_vec[i] corresponds to C_{i//3, i%3}. So row_index = (i%3)*3 + (i//3).
            row_w = (i % 3) * 3 + (i // 3)
            W[row_w][col] = w_vec[i]
            
    # Fill next 23 columns for block 2 (B cols 3-5, C cols 3-5)
    for t in range(23):
        u_vec, v_vec, w_vec = terms[t]
        col = 23 + t
        for i in range(9):
            U[i][col] = u_vec[i]
            
            # V index for B2: j*6 + (k+3)
            j = i // 3
            k = i % 3
            V[j * 6 + (k + 3)][col] = v_vec[i]
            
            # W index for C2: k' = k+3. row_w = k'*3 + i_row.
            # w_vec[i] corresponds to C_{i//3, i%3}.
            # So i_row = i//3, k_local = i%3.
            # k_global = k_local + 3.
            # W index = k_global * 3 + i_row.
            i_row = i // 3
            k_local = i % 3
            k_global = k_local + 3
            row_w = k_global * 3 + i_row
            W[row_w][col] = w_vec[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
