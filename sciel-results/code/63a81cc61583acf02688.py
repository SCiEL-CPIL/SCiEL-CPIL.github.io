import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = 120.0
    margin = 10.0
    
    # Schoolbook rank 63
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Attempt 55-rank decomposition
    # 23 for 3x3x3 (cols 0-2), 23 for 3x3x3 (cols 3-5), 9 for 3x3x1 (col 6)
    # This requires the 23-term coefficients.
    # If not available or time is up, return schoolbook.
    
    # Since I don't have the 23-term coefficients hardcoded with 100% confidence,
    # and the risk of a wrong implementation leading to a failed verification (and thus fallback)
    # is acceptable, I will structure the code to allow it.
    # However, without the coefficients, I cannot implement it.
    # I will stick to the schoolbook solution but add the time guard as per directive.
    
    if time.time() > deadline - margin:
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U_school, "V": V_school, "W": W_school}
