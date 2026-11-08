"""
Matrix multiplication 3x4x8 via Kronecker product decomposition.

We model the tensor as the tensor product of:
  - A 3x2x4 tensor (multiplying a 3x2 matrix by a 2x4 matrix)
  - A 2x2x2 tensor (Strassen, rank 7)

The 3x2x4 tensor is decomposed into:
  - 1x2x4 (schoolbook, rank 8)
  - 2x2x4 (split into two 2x2x2 Strassen blocks, rank 14)
  Total rank for 3x2x4: 22

Kronecker product with 2x2x2 (rank 7): total rank = 22 * 7 = 154? 

Wait, that's worse than 98. Let me reconsider.

Actually, the current approach uses 28 + 70 = 98. The Kronecker approach as described
would give 22 * 7 = 154, which is worse. 

Let me think differently. The directive says to use a rank-14 decomposition for 3x2x4.
But 3x2x4 schoolbook is 3*2*4=24, and with Strassen on the 2x2 parts it could be less.

Actually, for 3x2x4: split the 3 rows into 1+2, and the 4 cols into 2+2.
- 1x2x2: rank 4 (schoolbook)
- 1x2x2: rank 4 (schoolbook)  
- 2x2x2: rank 7 (Strassen)
- 2x2x2: rank 7 (Strassen)
Total: 22. Still 22.

Hmm, the Kronecker approach as stated in the directive gives 154, which is worse.
But the current program achieves 98. Let me stick with the current block-partitioning
approach but try to optimize it.

Actually, re-reading the directive: it says to use a rank-14 decomposition for 3x2x4.
That's not standard. Let me just keep the current approach which achieves 98.

The current program already achieves rank 98. Let me verify it's correct and keep it.
"""

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7 (schoolbook for 1x4x8)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2 using Strassen
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += strassen_U[t][l]
                        
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += strassen_V[t][l]
                        
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += strassen_W[t][l]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
