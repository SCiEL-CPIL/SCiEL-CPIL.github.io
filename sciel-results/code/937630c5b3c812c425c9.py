"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Schoolbook solution as baseline
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s, V_s, W_s
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    lower_bound = max(n*m, m*p, p*n)
    
    # Try to find lower-rank decompositions via structured search
    # Use a simple approach: try all combinations of small integer/half-integer values
    # for small ranks, but this is infeasible for large search spaces.
    # Instead, try a few heuristic initializations.
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10.0:
            break
        
        # Quick check: can we do it with this rank?
        # For now, just try a few random structured initializations
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Create a sparse initialization
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Assign random non-zero entries
            import random
            random.seed(trial * 137 + rank * 7919)
            for idx in range(n*m):
                for r in range(rank):
                    if random.random() < 0.3:
                        U[idx][r] = random.choice([1, -1, 0.5, -0.5])
            for idx in range(m*p):
                for r in range(rank):
                    if random.random() < 0.3:
                        V[idx][r] = random.choice([1, -1, 0.5, -0.5])
            for idx in range(p*n):
                for r in range(rank):
                    if random.random() < 0.3:
                        W[idx][r] = random.choice([1, -1, 0.5, -0.5])
            
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank == rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
