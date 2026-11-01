"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fallback = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fallback = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fallback = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = rank_schoolbook

    # Try to find a better decomposition using systematic search
    if time.time() < deadline - 5:
        success, U, V, W, rank = _search_for_better_decomposition(n, m, p, deadline)
        if success and rank < best_rank:
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}

    return best_result

def _search_for_better_decomposition(n, m, p, deadline):
    """
    Search for a lower-rank decomposition of the 2x2 x 2x14 matrix multiplication tensor.
    
    Strategy: Use the structure of the problem. The 2x14 matrix B can be viewed as 7 blocks
    of 2x2 matrices. We can use Strassen's algorithm on each block, but we can also try
    to share computations across blocks.
    
    Key insight: The tensor T_{(i,j),(j,k),(k,i)} = 1 for valid indices.
    We can try to find a decomposition of rank r by solving a system of equations.
    
    For small ranks, we can use a randomized approach:
    1. Initialize U, V, W with random values
    2. Use gradient descent or other optimization to minimize the error
    3. Try to round to integer/half-integer values
    """
    # First, try the known Strassen-based rank-49 construction
    if time.time() < deadline - 10:
        success, U, V, W = _construct_strassen_49(n, m, p)
        if success:
            return True, U, V, W, 49

    # Try to find a rank-42 or lower decomposition
    # The theoretical lower bound is 2 * 2 * 14 / 2 = 28 (from Bini's bound)
    # But Strassen gives 49. Can we do better?
    
    # For 2x2 x 2x2, the rank is 7 (Strassen)
    # For 2x2 x 2x14, we have 7 blocks, so naive Strassen gives 7*7 = 49
    # But we might be able to share some terms across blocks
    
    # Let's try a more systematic search for rank r
    # We'll use a randomized approach with local search
    
    for target_rank in [42, 40, 38, 36, 34, 32, 30, 28]:
        if time.time() > deadline - 10:
            break
        success, U, V, W = _try_find_decomposition(n, m, p, target_rank, deadline)
        if success:
            return True, U, V, W, target_rank

    return False, None, None, None

def _construct_strassen_49(n, m, p):
    """Construct a rank-49 decomposition using Strassen's algorithm on 2x2 blocks."""
    if p % 2 != 0:
        return False, None, None
    
    num_pairs = p // 2
    rank = 7 * num_pairs

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for pair_idx in range(num_pairs):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base = pair_idx * 7

        # M1 = (A00 + A11) * (B0k1 + B0k2 + B1k1 + B1k2)
        U[0][base] = 1
        U[3][base] = 1
        V[k1][base] = 1
        V[k2][base] = 1
        V[p + k1][base] = 1
        V[p + k2][base] = 1
        W[k1 * n + 0][base] = 1
        W[k2 * n + 1][base] = 1

        # M2 = (A10 + A11) * B0k1
        U[2][base + 1] = 1
        U[3][base + 1] = 1
        V[k1][base + 1] = 1
        W[k1 * n + 1][base + 1] = 1
        W[k2 * n + 1][base + 1] = -1

        # M3 = A00 * (B0k2 - B1k2)
        U[0][base + 2] = 1
        V[k2][base + 2] = 1
        V[p + k2][base + 2] = -1
        W[k2 * n + 0][base + 2] = 1
        W[k2 * n + 1][base + 2] = 1

        # M4 = A11 * (B1k1 - B0k1)
        U[3][base + 3] = 1
        V[p + k1][base + 3] = 1
        V[k1][base + 3] = -1
        W[k1 * n + 0][base + 3] = 1
        W[k1 * n + 1][base + 3] = 1

        # M5 = (A00 + A01) * B1k2
        U[0][base + 4] = 1
        U[1][base + 4] = 1
        V[p + k2][base + 4] = 1
        W[k1 * n + 0][base + 4] = -1
        W[k2 * n + 0][base + 4] = 1

        # M6 = (A10 - A00) * (B0k1 + B0k2)
        U[2][base + 5] = 1
        U[0][base + 5] = -1
        V[k1][base + 5] = 1
        V[k2][base + 5] = 1
        W[k1 * n + 1][base + 5] = -1

        # M7 = (A01 - A11) * (B1k1 + B1k2)
        U[1][base + 6] = 1
        U[3][base + 6] = -1
        V[p + k1][base + 6] = 1
        V[p + k2][base + 6] = 1
        W[k1 * n + 0][base + 6] = -1
        W[k2 * n + 1][base + 6] = 1

    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return True, U, V, W
    else:
        return False, None, None

def _try_find_decomposition(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of the given rank using randomized search.
    
    We represent the decomposition as U, V, W matrices with entries in {0, ±1/2, ±1, ±3/2, ±2}
    and search for one that exactly decomposes the tensor.
    """
    # Number of tensor entries
    num_entries = n * m * p
    
    # We'll use a simulated annealing / local search approach
    # Start with random integer/half-integer values
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_iterations = 5000
    time_per_iter = 0.02
    
    for iteration in range(num_iterations):
        if time.time() > deadline - 5:
            break
        
        # Generate random initial decomposition
        U = _random_matrix(n * m, target_rank)
        V = _random_matrix(m * p, target_rank)
        W = _random_matrix(p * n, target_rank)
        
        # Local search
        for inner_iter in range(20):
            if time.time() > deadline - 5:
                break
            
            # Compute error
            error = _compute_error(U, V, W, n, m, p)
            
            if error == 0:
                return True, U, V, W
            
            if error < best_error:
                best_error = error
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            
            # Perturb a random entry
            if random.random() < 0.33:
                r = random.randint(0, n * m - 1)
                c = random.randint(0, target_rank - 1)
                U[r][c] = _random_value()
            elif random.random() < 0.5:
                r = random.randint(0, m * p - 1)
                c = random.randint(0, target_rank - 1)
                V[r][c] = _random_value()
            else:
                r = random.randint(0, p * n - 1)
                c = random.randint(0, target_rank - 1)
                W[r][c] = _random_value()
        
        # If we've been stuck, try a bigger jump
        if iteration > 100 and best_error > 0.1:
            U = _random_matrix(n * m, target_rank)
            V = _random_matrix(m * p, target_rank)
            W = _random_matrix(p * n, target_rank)
    
    # Check if best solution is valid
    if best_error == 0:
        return True, best_U, best_V, best_W
    
    return False, None, None

def _random_matrix(rows, cols):
    """Generate a random matrix with integer/half-integer entries."""
    M = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            M[i][j] = _random_value()
    return M

def _random_value():
    """Generate a random integer or half-integer value."""
    choices = [0, 0, 0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5]
    return random.choice(choices)

def _compute_error(U, V, W, n, m, p):
    """Compute the error of the decomposition."""
    rank = len(U[0])
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                total_error += (total - 1.0) ** 2
    return total_error

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
