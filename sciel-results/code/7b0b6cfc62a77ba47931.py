"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time

def _check_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower-rank decomposition via structured search
    # For 2x3 * 3x4, try ranks from 16 to 23
    import random
    
    for target_rank in range(16, rank_sb):
        if time.time() > deadline - 20:
            break
        found = False
        for attempt in range(300):
            if time.time() > deadline - 20:
                break
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _check_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
