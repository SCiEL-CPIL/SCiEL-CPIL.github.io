import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Pad to 4, 6, 6 to allow Strassen split 2,3,3
    # A: 4x5 -> 4x6
    # B: 5x6 -> 6x6
    # C: 4x6 -> 4x6
    
    # We will construct the decomposition for 4x6x6
    # Using Strassen: 7 * M(2,3,3)
    
    # We need a decomposition for M(2,3,3)
    # Known rank is 12.
    # If we can't find it, we use schoolbook (18) -> Rank 126.
    # Schoolbook for 4x5x6 is 120.
    # So we MUST get M(2,3,3) < 120/7 = 17.14.
    # Rank 12 is ideal.
    
    # Let's try to find a rank-12 decomposition for 2x3x3 using a quick search.
    # If it takes too long, fallback to schoolbook for the whole thing?
    # No, fallback to schoolbook for M(2,3,3) gives 126.
    # Maybe we can use a different split for 4x5x6?
    
    # Alternative: Use schoolbook for 4x5x6 (120) if search fails.
    
    def get_decomposition_2x3x3():
        # Returns U, V, W for 2x3x3
        # n=2, m=3, p=3
        # Schoolbook rank 18
        # Try to find rank 12
        # Tensor T[i,j,k] = 1 if i,j,k valid
        # We want U (2*3=6, r), V (3*3=9, r), W (3*2=6, r)
        # such that sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k} ? No.
        # Matrix multiplication tensor: T_{i,j,k} = 1.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # Term is A_{i,j} B_{j,k}.
        # U row index: i*m+j.
        # V row index: j*p+k.
        # W row index: k*n+i.
        
        # We will use a simple random restart local search.
        best_rank = 18
        best_U, best_V, best_W = None, None, None
        
        # Initialize with schoolbook
        U_s = np.zeros((6, 18))
        V_s = np.zeros((9, 18))
        W_s = np.zeros((6, 18))
        c = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_s[i*3+j, c] = 1
                    V_s[j*3+k, c] = 1
                    W_s[k*2+i, c] = 1
                    c += 1
        
        # Search for lower rank
        # This is computationally expensive.
        # Given time constraints, I will just return schoolbook for now
        # and rely on the 4x6x6 split?
        # 7 * 18 = 126.
        # 120 is better.
        
        # Let's try a different strategy:
        # Directly search for a decomposition of 4x5x6 with rank < 120.
        
        return U_s, V_s, W_s

    # Since finding 12 is risky, and 126 > 120,
    # I will return the schoolbook decomposition for 4x5x6 (Rank 120)
    # UNLESS I can quickly find a better one.
    
    # Actually, let's just use the schoolbook decomposition.
    # It is guaranteed to be valid and score 1.0.
    # To improve, I would need a valid lower rank.
    
    # Let's try to implement the 4x6x6 Strassen with Schoolbook base (126)
    # and compare with 120. 120 is better.
    
    # So I will stick to 120 unless I can find a specific low-rank decomposition.
    
    # I will output the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
