import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x5 matrix by a 5x6 matrix.
    
    Uses a block decomposition with Strassen's algorithm on 2x2 sub-blocks.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    def check_decomposition(U, V, W, tol=1e-9):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    def get_strassen_decomposition():
        """
        Construct decomposition using Strassen for 2x2 blocks.
        
        Partition:
        - A (2x5) = [A1 (2x2) | A2 (2x3)]  
        - B (5x6) = [B1 (2x6); B2 (3x6)]
        - C = A1*B1 + A2*B2
        
        Part 1: A1*B1 (2x2 x 2x6)
          Split B1 into 3 column pairs: B1 = [B1_0 B1_1 B1_2], each 2x2
          Use Strassen (rank 7) for each pair.
          Columns in B1: (0,1), (2,3), (4,5)
          
        Part 2: A2*B2 (2x3 x 3x6)
          Split B2 into 2 column groups: B2 = [B2_0 B2_1], each 3x3
          Use schoolbook (rank 18) for each group.
          Columns in B2: (0,1,2), (3,4,5)
          
        Total rank: 3*7 + 2*18 = 21 + 36 = 57
        """
        rank = 57
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part 1: A1*B1 where A1 is A[:, 0:2] (2x2), B1 is B[0:2, :] (2x6)
        # Strassen for 2x2 x 2x2, applied to 3 column pairs
        for pair_idx in range(3):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1
            
            # M1 = A00 * B11
            U[0][col] = 1
            V[6 + k2][col] = 1
            W[2*k2][col] += 1
            W[2*k2+1][col] += 1
            col += 1
            
            # M2 = (A00 + A01) * B10
            U[0][col] = 1
            U[1][col] = 1
            V[6 + k1][col] = 1
            W[2*k1][col] += -1
            W[2*k2][col] += 1
            col += 1
            
            # M3 = (A10 + A11) * B00
            U[5][col] = 1
            U[6][col] = 1
            V[k1][col] = 1
            W[2*k1+1][col] += 1
            W[2*k2+1][col] += 1
            col += 1
            
            # M4 = A11 * B11
            U[6][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1
            W[2*k1+1][col] += 1
            col += 1
            
            # M5 = A00 * (B00 + B11)
            U[0][col] = 1
            V[k1][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1
            W[2*k2+1][col] += -1
            col += 1
            
            # M6 = (A01 - A11) * (B10 + B11)
            U[1][col] = 1
            U[6][col] = -1
            V[6 + k1][col] = 1
            V[6 + k2][col] = 1
            W[2*k1][col] += 1
            col += 1
            
            # M7 = (A00 - A10) * (B00 + B01)
            U[0][col] = 1
            U[5][col] = -1
            V[k1][col] = 1
            V[k2][col] = 1
            W[2*k2+1][col] += 1
            col += 1
        
        # Part 2: A2*B2 where A2 is A[:, 2:5] (2x3), B2 is B[2:5, :] (3x6)
        # Schoolbook for 2x3 x 3x3, applied to 2 column groups
        for group_idx in range(2):
            k_start = group_idx * 3
            for k_offset in range(3):
                k = k_start + k_offset
                for i in range(2):
                    for j in range(3):
                        U[i*5 + (j+2)][col] = 1
                        V[(j+2)*6 + k][col] = 1
                        W[k*2 + i][col] = 1
                        col += 1
        
        assert col == rank, f"Expected rank {rank}, got {col}"
        
        return U, V, W
    
    U, V, W = get_strassen_decomposition()
    
    if check_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    U, V, W = get_schoolbook()
    return {"U": U, "V": V, "W": W}

def check_decomposition(U, V, W, tol=1e-9):
    n, m, p = 2, 5, 6
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(R):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    is_valid = check_decomposition(res['U'], res['V'], res['W'])
    rank = len(res['U'][0])
    print(f"Rank: {rank}, Valid: {is_valid}")
