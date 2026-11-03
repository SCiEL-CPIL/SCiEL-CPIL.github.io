"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
        """Fast check using matrix operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        
        for i in range(n):
            for k in range(p):
                expected = np.zeros((n*m, m*p))
                for j in range(m):
                    expected[i*m+j, j*p+k] = 1.0
                
                w_vec = W[k*n+i, :]
                M_actual = (U * w_vec[None, :]).T @ V
                
                if not np.allclose(M_actual, expected, atol=tol):
                    return False
        return True
    
    def schoolbook():
        n, m, p = DIMENSIONS
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
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try a structured approach: use block decomposition
    # Split the 9 columns into groups and use shared computations
    
    # Strategy: For C = A*B where A is 4x4, B is 4x9
    # We can write B = [B1 | B2 | B3] with B1,B2,B3 each 4x3
    # C = [A*B1 | A*B2 | A*B3]
    # 
    # For each A*Bx (4x3), we can use a rank-48 algorithm, but share A's linear combos.
    # Total rank would be at most 48*3 = 144 (no gain) or less if we share.
    #
    # Alternative: use the fact that 4x4x4 has known rank 49.
    # If we could use a 4x4x4 algorithm for 3 of the column blocks...
    
    # Let me try: use 3 separate 4x4x3 multiplications, each with rank 48, total 144.
    # That's the same as schoolbook.
    
    # Better idea: use a 4x4x4 algorithm (rank 49) for the first 4 columns,
    # and a 4x4x5 algorithm for the last 5 columns.
    # 4x4x5: schoolbook rank 80. Total: 49 + 80 = 129.
    # Or: 4x4x4 (rank 49) + 4x4x4 (rank 49) + 4x4x1 (rank 16) = 114.
    
    # Let's try: 4x4x4 (rank 49) + 4x4x4 (rank 49) + 4x4x1 (rank 16) = 114
    # But we need the 4x4x4 algorithm. Let me use a simple one.
    
    # For 4x4x4, use the "baby-step" Strassen: split into 2x2 blocks.
    # 2x2x2 has rank 7 (Strassen). 4x4x4 can be done with rank 7^2 = 49? No, that's not right.
    # Actually, 4x4x4 can be done with rank 49 using the 2x2 Strassen recursively, 
    # but the exact construction is complex.
    
    # For simplicity, let me just use schoolbook for 4x4x4 (rank 64) and see if I can 
    # combine with 4x4x1 (rank 16) to get 64+64+16 = 144, which is no better.
    
    # Let me try a different split: 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # If I use rank 49 for each 4x4x4, total = 49+49+16 = 114.
    # But I need a rank-49 4x4x4 algorithm.
    
    # For now, let me just return the schoolbook result if I can't find something better quickly.
    # The random search is unlikely to work, so let me just return schoolbook.
    
    # Actually, let me try one more thing: a simple random search with more tries
    # but only for rank 143 down to 130.
    
    random.seed(42)
    for R in range(best_rank - 1, 129, -1):
        if time.time() > deadline - 10:
            break
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            for r in range(R):
                for idx in random.sample(range(n * m), 3):
                    U[idx, r] = random.choice([-1, 1])
                for idx in random.sample(range(m * p), 3):
                    V[idx, r] = random.choice([-1, 1])
                for idx in random.sample(range(p * n), 3):
                    W[idx, r] = random.choice([-1, 1])
            if check_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = R
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
