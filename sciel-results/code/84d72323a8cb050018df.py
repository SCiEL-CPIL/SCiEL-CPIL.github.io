import numpy as np
import time
import random
from collections import defaultdict

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Helper to build the known rank 64 solution
    def build_rank_64():
        # Core: 4x4x4 using 4x (2x2x2 Strassen) -> Rank 28
        # Edge 1: 4x1 (A col 4) x 4x4 (B rows 0-3) -> Rank 16
        # Edge 2: 4x4 (A cols 0-3) x 1x4 (B row 4) -> Rank 16
        # Edge 3: 4x1 (A col 4) x 1x1 (B row 4, col 4) -> Rank 4
        # Total Rank: 28 + 16 + 16 + 4 = 64
        
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Helper to get 2x2x2 Strassen factors
        def get_strassen_2x2_factors():
            u = [[0]*7 for _ in range(4)]
            v = [[0]*7 for _ in range(4)]
            w = [[0]*7 for _ in range(4)]
            
            # Term 0: (a00+a11)(b00+b11)
            u[0][0]=1; u[3][0]=1
            v[0][0]=1; v[3][0]=1
            w[0][0]=1; w[3][0]=1
            
            # Term 1: (a10+a11)b00
            u[2][1]=1; u[3][1]=1
            v[0][1]=1
            w[2][1]=1; w[3][1]=-1
            
            # Term 2: a00(b01-b11)
            u[0][2]=1
            v[1][2]=1; v[3][2]=-1
            w[1][2]=1; w[3][2]=1
            
            # Term 3: a11(b10-b00)
            u[3][3]=1
            v[2][3]=1; v[0][3]=-1
            w[0][3]=1; w[2][3]=1
            
            # Term 4: (a00+a01)b11
            u[0][4]=1; u[1][4]=1
            v[3][4]=1
            w[0][4]=-1; w[1][4]=1
            
            # Term 5: (a10-a00)(b00+b01)
            u[0][5]=-1; u[2][5]=1
            v[0][5]=1; v[1][5]=1
            w[3][5]=1
            
            # Term 6: (a01-a11)(b10+b11)
            u[1][6]=1; u[3][6]=-1
            v[2][6]=1; v[3][6]=1
            w[0][6]=1
            
            return u, v, w

        s_u, s_v, s_w = get_strassen_2x2_factors()
        
        # 1. Core: 4x4x4 part
        # Split into 4 blocks of 2x2x2
        # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # Block (0,1): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        # Block (1,0): A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
        # Block (1,1): A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
        
        col = 0
        for br_a in range(2): # Row block of A
            for bc_a in range(2): # Col block of A
                for br_b in range(2): # Row block of B
                    for bc_b in range(2): # Col block of B
                        # A block is 2x2, B block is 2x2
                        # C block is 2x2 at (br_a*2, bc_b*2)
                        
                        for k in range(7):
                            # U factors
                            for lr in range(2):
                                for lc in range(2):
                                    local_idx = lr * 2 + lc
                                    val = s_u[local_idx][k]
                                    if val != 0:
                                        i = br_a * 2 + lr
                                        j = bc_a * 2 + lc
                                        U[i * 5 + j][col] += val
                            
                            # V factors
                            for lr in range(2):
                                for lc in range(2):
                                    local_idx = lr * 2 + lc
                                    val = s_v[local_idx][k]
                                    if val != 0:
                                        j = br_b * 2 + lr
                                        k_idx = bc_b * 2 + lc
                                        V[j * 5 + k_idx][col] += val
                            
                            # W factors
                            for lr in range(2):
                                for lc in range(2):
                                    local_idx = lr * 2 + lc
                                    val = s_w[local_idx][k]
                                    if val != 0:
                                        i = br_a * 2 + lr
                                        k_idx = bc_b * 2 + lc
                                        W[k_idx * 4 + i][col] += val
                            
                            col += 1
                            
        # col is now 28
        
        # 2. Edge 1: A[0:4, 4] * B[0:4, 0:4]
        # A is 4x1, B is 4x4. C is 4x4.
        # Rank 16.
        for i in range(4):
            for j in range(4):
                U[i * 5 + 4][col] = 1
                V[j * 5 + j][col] = 1 # B[j, j]
                W[j * 4 + i][col] = 1
                col += 1
                
        # col is now 44
        
        # 3. Edge 2: A[0:4, 0:4] * B[4, 0:4]
        # A is 4x4, B is 1x4. C is 4x4.
        # Rank 16.
        for i in range(4):
            for k in range(4):
                U[i * 5 + i][col] = 1 # A[i, i]
                V[4 * 5 + k][col] = 1 # B[4, k]
                W[k * 4 + i][col] = 1
                col += 1
                
        # col is now 60
        
        # 4. Edge 3: A[0:4, 4] * B[4, 4]
        # A is 4x1, B is 1x1. C is 4x1.
        # Rank 4.
        for i in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
        return {"U": U, "V": V, "W": W}

    baseline = build_rank_64()
    return baseline
