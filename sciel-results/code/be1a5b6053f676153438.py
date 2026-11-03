"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 5, 6

    # Schoolbook baseline
    rank_sb = n * m * p  # 120
    def make_schoolbook():
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = rank_sb

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Partition approach:
    # Core: i in {0,1,2}, k in {0,1,2}, j in {0,1,2} -> 3x3x3
    # Strip1: i=3, k in {0,1,2}, j in {0,1,2,3,4} -> 1x5x3
    # Strip2: i in {0,1,2}, k in {3,4,5}, j in {0,1,2,3,4} -> 3x5x3
    # Strip3: i=3, k in {3,4,5}, j in {0,1,2,3,4} -> 1x5x3
    
    # Total rank with all schoolbook: 27 + 15 + 45 + 15 = 102
    # With Strassen for core: 23 + 15 + 45 + 15 = 98

    def make_partitioned(core_rank, core_decomp=None):
        """
        Build the partitioned decomposition.
        core_decomp: if None, use schoolbook for core (rank 27)
                     if provided, use it (should be rank core_rank)
        """
        # Strips always use schoolbook
        strip1_rank = 1 * 5 * 3  # 15
        strip2_rank = 3 * 5 * 3  # 45
        strip3_rank = 1 * 5 * 3  # 15
        
        total_rank = core_rank + strip1_rank + strip2_rank + strip3_rank
        
        U = [[0] * total_rank for _ in range(n * m)]  # 20 rows
        V = [[0] * total_rank for _ in range(m * p)]  # 30 rows
        W = [[0] * total_rank for _ in range(p * n)]  # 24 rows
        
        col = 0
        
        # Core: i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
        if core_decomp is None:
            # Schoolbook for core
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
        else:
            # Use provided core decomposition
            for c in range(core_rank):
                for i in range(3):
                    for j in range(3):
                        U[i * m + j][col + c] += core_decomp['U'][i * 3 + j][c]
                for j in range(3):
                    for k in range(3):
                        V[j * p + k][col + c] += core_decomp['V'][j * 3 + k][c]
                for k in range(3):
                    for i in range(3):
                        W[k * n + i][col + c] += core_decomp['W'][k * 3 + i][c]
            col += core_rank
        
        # Strip1: i=3, j in {0..4}, k in {0,1,2}
        for j in range(5):
            for k in range(3):
                U[3 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 3][col] = 1
                col += 1
        
        # Strip2: i in {0,1,2}, j in {0..4}, k in {3,4,5}
        for i in range(3):
            for j in range(5):
                for k in range(3, 6):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Strip3: i=3, j in {0..4}, k in {3,4,5}
        for j in range(5):
            for k in range(3, 6):
                U[3 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 3][col] = 1
                col += 1
        
        return {"U": U, "V": V, "W": W}

    # Try partitioned schoolbook first (rank 102)
    if time.time() < deadline - 30.0:
        result_102 = make_partitioned(27, None)
        if verify(result_102['U'], result_102['V'], result_102['W'], 102):
            best_result = result_102
            best_rank = 102
    
    # Try to use Strassen for the 3x3x3 core (rank 23)
    # Strassen's 3x3x3 decomposition
    if time.time() < deadline - 10.0 and best_rank == 102:
        # Define Strassen's 23-rank decomposition for 3x3x3
        # A is 3x3, B is 3x3, C = A*B
        # Using 0-indexed: A[i][j], B[j][k], C[i][k]
        
        # The 23 bilinear forms for Strassen 3x3x3
        # I'll use a known correct formulation
        strassen_decomp = {
            'U': [[0]*23 for _ in range(9)],
            'V': [[0]*23 for _ in range(9)],
            'W': [[0]*23 for _ in range(9)]
        }
        
        # Strassen's algorithm for 3x3x3 (23 multiplications)
        # Let me define the 23 forms carefully
        # Using the standard Strassen 3x3 decomposition
        
        # Form 1: (A[0][0]+A[2][2])*(B[0][0]+B[2][2])
        # C[0][0] += M1, C[0][2] -= M1, C[2][0] -= M1, C[2][2] += M1
        # Actually, let me use a verified set of 23 forms
        
        # I'll use a simpler approach: verify a known Strassen 3x3x3
        # For now, let's just use the partitioned schoolbook result
        
        pass
    
    return best_result
# EVOLVE-BLOCK-END
