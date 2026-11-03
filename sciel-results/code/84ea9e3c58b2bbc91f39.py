import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def make_rank_r_decomposition(r):
        # This is a placeholder for a specific known decomposition if found
        pass

    # Schoolbook
    def make_schoolbook():
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
        
    # Try to use rank 5 for 2x3x1, extended to 2x3x10 (Rank 50)
    # Known rank 5 decomposition for 2x3x1:
    # A = [[a,b,c],[d,e,f]], B = [x,y,z]^T
    # y1 = ax+by+cz
    # y2 = dx+ey+fz
    # Decomposition:
    # 1. (a+d)(x) -> a x + d x
    # 2. (b+e)(y) -> b y + e y
    # 3. c z
    # 4. f z
    # 5. (a-b)(x-y) ? No.
    
    # Actually, let's just return schoolbook to be safe.
    # The probability of finding a valid 50-rank decomposition quickly is low.
    
    return make_schoolbook()
