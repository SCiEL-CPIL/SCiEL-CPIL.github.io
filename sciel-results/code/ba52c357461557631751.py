"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Splitting approach: decompose 4x4x5 into block sub-problems
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # This doesn't split cleanly, so use a different decomposition.
    
    # Actually, for block matrix multiplication:
    # If A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12, B13], [B21, B22, B23]] (2x3 blocks of 2x2, 2x2, 2x1)
    # Wait, B is 4x5, so blocks would be 2x2, 2x2, 2x1
    
    # C = A*B where C is 4x5
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    # C[0:2, 4:5] = A[0:2,0:2]*B[0:2,4:5] + A[0:2,2:4]*B[2:4,4:5]
    # C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    # C[2:4, 2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
    # C[2:4, 4:5] = A[2:4,0:2]*B[0:2,4:5] + A[2:4,2:4]*B[2:4,4:5]
    
    # This requires 12 sub-multiplications. Not ideal.
    
    # Better: use the fact that we can decompose the tensor directly.
    # For 4x4x5, let's use a rank-38 decomposition via splitting into:
    # - Two 2x2x2 Strassen (rank 7 each) = 14
    # - Two 2x2x3 schoolbook (rank 12 each) = 24
    # Total = 38
    
    # The decomposition:
    # Split A rows into {0,1} and {2,3}
    # Split B cols into {0,1} and {2,3,4}  
    # Split B rows into {0,1} and {2,3}
    
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # We can't separate these, so this splitting doesn't work directly.
    
    # Alternative: use a known rank decomposition.
    # Let's just use schoolbook rank 80 as baseline and try to improve.
    
    # Schoolbook decomposition: rank = n*m*p = 80
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    return best_result
# EVOLVE-BLOCK-END
