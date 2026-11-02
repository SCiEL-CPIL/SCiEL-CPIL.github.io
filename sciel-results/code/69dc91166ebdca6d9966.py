"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
    def schoolbook():
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

    def verify_numpy(U, V, W):
        """Fast vectorized verification."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)

    def verify_exact(U, V, W):
        """Exact verification using fractions to avoid floating point issues."""
        from fractions import Fraction
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if total != 1:
                        return False
        return True

    best_result = schoolbook()
    best_rank = n * m * p

    # Strategy: greedy basis reduction on the schoolbook tensor
    # The schoolbook has 84 rank-1 components. We try to find linear dependencies
    # among the "rows" of the tensor to reduce rank.
    # 
    # Key insight: For fixed (i,k), the m terms (one per j) all contribute to C[i,k].
    # If we can find a way to express the contribution from multiple j's using fewer
    # rank-1 terms that also correctly handle other (i',k') pairs, we reduce rank.
    #
    # Alternative: Use known constructions. For 3x4x7, try rank from 28 upward
    # with structured random search using small integer/half-integer values.

    # Value pool for half-integer entries
    value_pool = [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5, 3, -3, 0.25, -0.25]

    # Try ranks from lower bound (28) up to current best (83)
    # Focus on promising ranges first
    rank_order = list(range(28, 50)) + list(range(50, 70)) + list(range(70, 84))
    
    for target_rank in rank_order:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue

        # Limit attempts per rank based on time budget
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # More attempts for lower ranks (harder but more valuable)
        if target_rank < 35:
            max_attempts = 200
        elif target_rank < 45:
            max_attempts = 100
        elif target_rank < 60:
            max_attempts = 50
        else:
            max_attempts = 20

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            random.seed(hash((target_rank, attempt)) % (2**32))
            
            # Different initialization strategies
            strategy = attempt % 6

            if strategy == 0:
                # Random from value pool
                U = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(p * n)]
            elif strategy == 1:
                # Sparse: mostly zeros with some ±1
                U = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(p * n)]
            elif strategy == 2:
                # Structured: partition columns into groups for each i
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                # Assign random columns to each (i,j) pair
                col_assign = {}
                for c in range(target_rank):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    if (i, j, k) not in col_assign:
                        col_assign[(i, j, k)] = []
                    col_assign[(i, j, k)].append(c)
                for (i, j, k), cols in col_assign.items():
                    for c in cols:
                        U[i * m + j][c] = 1
                        V[j * p + k][c] = 1
                        W[k * n + i][c] = 1
            elif strategy == 3:
                # Strassen-inspired: use block structure
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                # Random linear combinations
                for c in range(target_rank):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    U[i * m + j][c] = random.choice([1, -1])
                    V[j * p + k][c] = random.choice([1, -1])
                    W[k * n + i][c] = random.choice([1, -1])
            elif strategy == 4:
                # Half-integer heavy
                U = [[random.choice([0.5, -0.5, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice([0.5, -0.5, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice([0.5, -0.5, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            else:
                # Very sparse with larger values
                U = [[random.choice([0, 0, 0, 0, 2, -2]) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice([0, 0, 0, 0, 2, -2]) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice([0, 0, 0, 0, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p * n)]

            if verify_numpy(U, V, W):
                # Double-check with exact verification
                if verify_exact(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break

    return best_result
# EVOLVE-BLOCK-END
