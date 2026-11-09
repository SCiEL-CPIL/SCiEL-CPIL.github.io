import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, num_samples=10):
        rng = random.Random(42)
        rank = len(U[0]) if U else 0
        for _ in range(num_samples):
            A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
            C_true = [[0] * p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += A[i][j] * B[j][k]
                    C_true[i][k] = s
            
            # Compute M_r
            M = [0] * rank
            for r in range(rank):
                sum_A = 0
                for i in range(n):
                    for j in range(m):
                        sum_A += U[i * m + j][r] * A[i][j]
                sum_B = 0
                for j in range(m):
                    for k in range(p):
                        sum_B += V[j * p + k][r] * B[j][k]
                M[r] = sum_A * sum_B
                
            C_fact = [[0] * p for _ in range(n)]
            for r in range(rank):
                for i in range(n):
                    for k in range(p):
                        C_fact[i][k] += W[k * n + i][r] * M[r]
                        
            for i in range(n):
                for k in range(p):
                    if C_true[i][k] != C_fact[i][k]:
                        return False
        return True

    # Hardcode a rank-23 3x3x3 decomposition
    # This is a known result.
    # If I can't recall the exact coefficients, I will use schoolbook for 3x3x3 (rank 27).
    # 27 + 36 = 63. No improvement.
    # I need a rank < 27 for 3x3x3 or a better split.
    
    # Let's try to find a rank-23 decomposition online in my training data?
    # I don't have it.
    
    # I will return schoolbook to be safe, as I cannot guarantee a correct rank < 63 decomposition.
    # However, the task asks to improve.
    # I will try a random search for rank 62 with the correct verify.
    
    best_result = schoolbook()
    
    # Try rank 62
    for target_rank in [62, 61]:
        if time.time() > deadline - 10: break
        for attempt in range(20):
            rng = random.Random(target_rank * 100 + attempt)
            U = [[rng.randint(-2, 2) if rng.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(9)]
            V = [[rng.randint(-2, 2) if rng.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(21)]
            W = [[rng.randint(-2, 2) if rng.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(21)]
            if verify(U, V, W, num_samples=5):
                best_result = {"U": U, "V": V, "W": W}
                break
        if best_result["U"] != schoolbook()["U"]: break
        
    return best_result
