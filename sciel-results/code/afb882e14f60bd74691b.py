import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s wall-clock limit

    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72

    # Initialize with schoolbook decomposition as the fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank

    def check_decomposition_333(U, V, W, rank):
        """Check if the decomposition exactly represents 3x3x3 matrix multiplication."""
        # U: 9 x rank, V: 3 x rank, W: 3 x rank
        # T[(i,j),(j,k),(k,i)] = 1
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * 3 + j][r] * V[j][r] * W[k][r]
                    if s != 1:
                        return False
        return True

    def check_decomposition_332(U, V, W, rank):
        """Check if the decomposition exactly represents 3x3x2 matrix multiplication."""
        # U: 9 x rank, V: 6 x rank (2 cols), W: 6 x rank (2 cols)
        # Indices: V[j*2+k], W[k*3+i]
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * 3 + j][r] * V[j * 2 + k][r] * W[k * 3 + i][r]
                    if s != 1:
                        return False
        return True

    def search_333(target_rank, time_limit=10):
        """Search for a rank decomposition of 3x3x3."""
        start = time.time()
        random.seed(42)
        # Values to try
        vals = [-1, 0, 1]
        # More aggressive search with half-integers if needed
        vals2 = [-2, -1, -0.5, 0, 0.5, 1, 2]
        
        for attempt in range(10000):
            if time.time() > start + time_limit:
                return None
            
            U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(3)]
            W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(3)]
            
            # Occasionally use larger values
            if attempt % 10 == 0:
                for i in range(9):
                    for r in range(target_rank):
                        if random.random() < 0.1:
                            U[i][r] = random.choice(vals2)
                for i in range(3):
                    for r in range(target_rank):
                        if random.random() < 0.1:
                            V[i][r] = random.choice(vals2)
                            W[i][r] = random.choice(vals2)
                            
            if check_decomposition_333(U, V, W, target_rank):
                return U, V, W, target_rank
        return None

    def search_332(target_rank, time_limit=5):
        """Search for a rank decomposition of 3x3x2."""
        start = time.time()
        random.seed(43)
        vals = [-1, 0, 1]
        
        for attempt in range(10000):
            if time.time() > start + time_limit:
                return None
            
            U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(6)]
            W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(6)]
            
            if check_decomposition_332(U, V, W, target_rank):
                return U, V, W, target_rank
        return None

    # Strategy: Combine 3x3x3 blocks and 3x3x2 block
    # We need 2x 3x3x3 and 1x 3x3x2
    # Target ranks: 23 for 3x3x3, 18 for 3x3x2 (schoolbook)
    
    # Search for 3x3x3 rank 23
    res_333_1 = search_333(23, time_limit=20)
    res_333_2 = search_333(23, time_limit=20)
    res_332 = search_332(18, time_limit=10)
    
    # If we found them, construct the full decomposition
    if res_333_1 and res_333_2 and res_332:
        U1, V1, W1, r1 = res_333_1
        U2, V2, W2, r2 = res_333_2
        U3, V3, W3, r3 = res_332
        
        total_rank = r1 + r2 + r3
        
        U_full = [[0] * total_rank for _ in range(9)]
        V_full = [[0] * total_rank for _ in range(24)] # 3*8
        W_full = [[0] * total_rank for _ in range(24)] # 8*3
        
        # Block 1: k=0,1,2
        for i in range(9):
            for r in range(r1):
                U_full[i][r] = U1[i][r]
        for j in range(3):
            for k in range(3):
                for r in range(r1):
                    V_full[j * 8 + k][r] = V1[j][r]
        for k in range(3):
            for i in range(3):
                for r in range(r1):
                    W_full[k * 3 + i][r] = W1[k][r]
                    
        # Block 2: k=3,4,5
        offset1 = r1
        for i in range(9):
            for r in range(r2):
                U_full[i][offset1 + r] = U2[i][r]
        for j in range(3):
            for k in range(3):
                for r in range(r2):
                    V_full[j * 8 + (3 + k)][offset1 + r] = V2[j][r]
        for k in range(3):
            for i in range(3):
                for r in range(r2):
                    W_full[(3 + k) * 3 + i][offset1 + r] = W2[k][r]
                    
        # Block 3: k=6,7
        offset2 = r1 + r2
        for i in range(9):
            for r in range(r3):
                U_full[i][offset2 + r] = U3[i][r]
        for j in range(3):
            for k in range(2):
                for r in range(r3):
                    V_full[j * 8 + (6 + k)][offset2 + r] = V3[j * 2 + k][r]
        for k in range(2):
            for i in range(3):
                for r in range(r3):
                    W_full[(6 + k) * 3 + i][offset2 + r] = W3[k * 3 + i][r]
                    
        # Verify full decomposition
        def check_full(U, V, W, rank):
            for i in range(3):
                for j in range(3):
                    for k in range(8):
                        s = 0
                        for r in range(rank):
                            s = s + U[i * 3 + j][r] * V[j * 8 + k][r] * W[k * 3 + i][r]
                        if s != 1:
                            return False
            return True
            
        if check_full(U_full, V_full, W_full, total_rank):
            best_result = {"U": U_full, "V": V_full, "W": W_full}
            best_rank = total_rank
            print(f"Found decomposition with rank {total_rank}")
        else:
            print("Verification failed for combined blocks")
    else:
        print("Did not find optimal blocks, using schoolbook")
        
    return best_result
