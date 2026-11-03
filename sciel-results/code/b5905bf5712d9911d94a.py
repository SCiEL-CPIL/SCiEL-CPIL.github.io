"""Matrix multiplication tensor decomposition for 3x4 x 4x8."""
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = DIMENSIONS
    # Schoolbook rank: n*m*p = 96
    rank_school = n * m * p

    # Build schoolbook decomposition
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Check time - if we're running low, return schoolbook
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}

    # Try to find a better decomposition using Strassen's algorithm
    # Strategy: 
    # - Row i=0: 1x4x8 block. Use schoolbook (rank 8*4 = 32).
    # - Rows i=1,2: 2x4x8 block. Use Strassen on 2x2 subblocks.
    #   Split j into 2 blocks of 2: [0,1], [2,3]
    #   Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    #   For each (j_block, k_block) pair, we have a 2x2x2 subproblem.
    #   Strassen reduces rank from 8 to 7 per 2x2x2 block.
    #   There are 2 * 4 = 8 such blocks.
    #   Rank for this part = 8 * 7 = 56.
    # Total rank = 32 + 56 = 88.

    # If we don't have enough time to construct the Strassen version, return schoolbook
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}

    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7
    # Schoolbook for 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    
    # Strassen local factors for 2x2x2
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[C11, C12], [C21, C22]]
    # C = AB
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor decomposition.
    # Each M_i is a rank-1 term in the decomposition of the 2x2x2 tensor.
    # The U, V, W matrices for the 2x2x2 tensor are:
    
    # U (for A):
    # M1: a=1, b=0, c=0, d=1  -> [1, 0, 0, 1]
    # M2: a=0, b=0, c=1, d=1  -> [0, 0, 1, 1]
    # M3: a=1, b=0, c=0, d=0  -> [1, 0, 0, 0]
    # M4: a=0, b=0, c=0, d=1  -> [0, 0, 0, 1]
    # M5: a=1, b=1, c=0, d=0  -> [1, 1, 0, 0]
    # M6: a=-1, b=0, c=1, d=0 -> [-1, 0, 1, 0]
    # M7: a=0, b=1, c=0, d=-1 -> [0, 1, 0, -1]
    
    # V (for B):
    # M1: e=1, f=0, g=0, h=1  -> [1, 0, 0, 1]
    # M2: e=1, f=0, g=0, h=0  -> [1, 0, 0, 0]
    # M3: e=0, f=1, g=0, h=-1 -> [0, 1, 0, -1]
    # M4: e=-1, f=0, g=1, h=0 -> [-1, 0, 1, 0]
    # M5: e=0, f=0, g=0, h=1  -> [0, 0, 0, 1]
    # M6: e=1, f=1, g=0, h=0  -> [1, 1, 0, 0]
    # M7: e=0, f=0, g=1, h=1  -> [0, 0, 1, 1]
    
    # W (for C):
    # M1: C11+=1, C12+=0, C21+=0, C22+=1 -> [1, 0, 0, 1]
    # M2: C11+=0, C12+=0, C21+=1, C22+=-1 -> [0, 0, 1, -1]
    # M3: C11+=0, C12+=1, C21+=0, C22+=1 -> [0, 1, 0, 1]
    # M4: C11+=1, C12+=0, C21+=1, C22+=0 -> [1, 0, 1, 0]
    # M5: C11+=-1, C12+=1, C21+=0, C22+=0 -> [-1, 1, 0, 0]
    # M6: C11+=0, C12+=0, C21+=0, C22+=1 -> [0, 0, 0, 1]
    # M7: C11+=1, C12+=0, C21+=0, C22+=0 -> [1, 0, 0, 0]
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1: C11+=1, C22+=1
        [0, 0, 1, -1], # M2: C21+=1, C22+=-1
        [0, 1, 0, 1], # M3: C12+=1, C22+=1
        [1, 0, 1, 0], # M4: C11+=1, C21+=1
        [-1, 1, 0, 0], # M5: C11+=-1, C12+=1
        [0, 0, 0, 1], # M6: C22+=1
        [1, 0, 0, 0]  # M7: C11+=1
    ]
    
    # Iterate over j_blocks and k_blocks
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            # 7 Strassen terms
            for t in range(7):
                # Map local indices to global
                # Local A index l -> global i, j
                # l=0 -> i'=0, j'=0 -> global i=1, j=j_start+0
                # l=1 -> i'=0, j'=1 -> global i=1, j=j_start+1
                # l=2 -> i'=1, j'=0 -> global i=2, j=j_start+0
                # l=3 -> i'=1, j'=1 -> global i=2, j=j_start+1
                
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
                
    # Verify col == rank
    if col != rank:
        # Fallback to schoolbook if something went wrong
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Verify the decomposition is correct
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
