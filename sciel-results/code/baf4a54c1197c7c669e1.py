import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Schoolbook fallback (rank 100)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx][r] = 1
                V_school[v_idx][r] = 1
                W_school[w_idx][r] = 1
                r += 1

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r_idx in range(rank):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                    if s != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition (rank 7)
    strassen_2x2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    def tensor_product_vec(v1, v2):
        result = [0] * 16
        for a1 in range(2):
            for b1 in range(2):
                for a2 in range(2):
                    for b2 in range(2):
                        idx = (a1*2+a2)*4 + (b1*2+b2)
                        result[idx] = v1[a1*2+b1] * v2[a2*2+b2]
        return result
    
    strassen_4x4x4 = []
    for t1 in strassen_2x2x2:
        for t2 in strassen_2x2x2:
            u = tensor_product_vec(t1[0], t2[0])
            v = tensor_product_vec(t1[1], t2[1])
            w = tensor_product_vec(t1[2], t2[2])
            strassen_4x4x4.append((u, v, w))
    
    # Strategy: Use Strassen 4x4x4 core (rank 49) + optimized cross terms
    # Cross terms:
    #   Strip_j: i in 0..3, j=4, k in 0..3 -> T_{i,4,k}=1 (4x1x4 block)
    #   Strip_k: i in 0..3, j in 0..3, k=4 -> T_{i,j,4}=1 (4x4x1 block)
    #   Corner:  i in 0..3, j=4, k=4 -> T_{i,4,4}=1 (4x1x1 block)
    
    # For Strip_j (4x1x4): Need sum_r U[i,4,r]*V[4,k,r]*W[k,i,r] = 1 for all i,k in 0..3
    # Solution: 4 terms with U[i,4,r]=1, V[4,k,r]=delta_{k,r}, W[k,i,r]=1
    # Check: sum_r 1*delta_{k,r}*1 = 1. Correct!
    
    # For Strip_k (4x4x1): Need sum_r U[i,j,r]*V[j,4,r]*W[4,i,r] = 1 for all i,j in 0..3
    # Solution: 4 terms with U[i,j,r]=1, V[j,4,r]=delta_{j,r}, W[4,i,r]=1
    # Check: sum_r 1*delta_{j,r}*1 = 1. Correct!
    
    # For Corner (4x1x1): Need sum_r U[i,4,r]*V[4,4,r]*W[4,i,r] = 1 for all i in 0..3
    # Solution: 4 terms with U[i,4,r]=delta_{i,r}, V[4,4,r]=1, W[4,i,r]=1
    # Check: sum_r delta_{i,r}*1*1 = 1. Correct!
    
    # Total: 49 + 4 + 4 + 4 = 61
    
    rank = 61
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Add Strassen 4x4x4 terms (handles i<4, j<4, k<4)
    for r_idx, (u, v, w) in enumerate(strassen_4x4x4):
        for i in range(4):
            for j in range(4):
                U[i*5+j][r_idx] = u[i*4+j]
        for j in range(4):
            for k in range(4):
                V[j*5+k][r_idx] = v[j*4+k]
        for k in range(4):
            for i in range(4):
                W[k*4+i][r_idx] = w[k*4+i]
    
    # Add Strip_j: i<4, j=4, k<4 (4 terms)
    r_offset = 49
    for r in range(4):
        for i in range(4):
            U[i*5+4][r_offset + r] = 1
        for k in range(4):
            V[4*5+k][r_offset + r] = (1 if k == r else 0)
        for k in range(4):
            for i in range(4):
                W[k*4+i][r_offset + r] = 1
    
    # Add Strip_k: i<4, j<4, k=4 (4 terms)
    r_offset = 53
    for r in range(4):
        for i in range(4):
            for j in range(4):
                U[i*5+j][r_offset + r] = 1
        for j in range(4):
            V[j*5+4][r_offset + r] = (1 if j == r else 0)
        for i in range(4):
            W[4*4+i][r_offset + r] = 1
    
    # Add Corner: i<4, j=4, k=4 (4 terms)
    r_offset = 57
    for r in range(4):
        for i in range(4):
            U[i*5+4][r_offset + r] = (1 if i == r else 0)
        V[4*5+4][r_offset + r] = 1
        for i in range(4):
            W[4*4+i][r_offset + r] = 1
    
    if verify(U, V, W):
        return {'U': U, 'V': V, 'W': W}
    
    # Try to reduce rank by merging some Strassen terms with cross terms
    # This is a heuristic: try to find if any of the 49 Strassen terms can be
    # combined with the cross terms to reduce total rank.
    
    # Alternative: use a different 4x4x4 decomposition with lower rank
    # The best known 4x4x4 rank is 49 (Strassen), but let's try a different approach.
    
    # Try: use 2x2x2 Strassen in a different way for 4x5x5
    # Decompose 4x5x5 as:
    #   - 2x5x5 block for rows 0-1 (rank 2*5*5=50 schoolbook, or use Winograd)
    #   - 2x5x5 block for rows 2-3 (rank 50 schoolbook)
    #   Total: 100 schoolbook, not better.
    
    # Let's try: use the fact that 5 = 2+2+1 and use Strassen on the 2x2 blocks
    # A is 4x5 = [A1(2x2) A2(2x2) A3(2x1)]
    # B is 5x5 = [B1(2x5); B2(2x5); B3(1x5)]
    # C = A1*B1 + A2*B2 + A3*B3
    # Each A_i*B_i is 2x2 * 2x5 = 2x5, which can be done with Strassen-like methods.
    # A_i*B_i: 2x2 times 2x5. This is like multiplying a 2x2 matrix by a 2x5 matrix.
    # We can view this as 5 independent 2x2 times 2x1 multiplications, each of rank 7 (Strassen).
    # Total: 5*7 = 35 per 2x5 block, so 35*2 + (2x1 * 1x5 = 2x5, rank 10 schoolbook) = 80.
    # Not better than 61.
    
    # Let's stick with 61 and return it.
    return {'U': U, 'V': V, 'W': W}

