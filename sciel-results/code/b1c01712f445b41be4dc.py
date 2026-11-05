"""Program for 3x3x4 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36

    # Default schoolbook decomposition (rank = 36)
    def schoolbook():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    # Try to find a lower-rank decomposition using randomized search
    # Strategy: Use a prime field and try to find rank r < 36 decomposition
    P = 2**61 - 1  # Mersenne prime

    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is valid using modular arithmetic."""
        U_mod = U.astype(np.int64) % P
        V_mod = V.astype(np.int64) % P
        W_mod = W.astype(np.int64) % P
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1 % P
                    computed = 0
                    for c in range(rank):
                        computed = (computed + U_mod[i * m + j, c] * V_mod[j * p + k, c] % P * W_mod[k * n + i, c]) % P
                    if computed != expected:
                        return False
        return True

    # Try to find lower rank decompositions using random search with verification
    # This is a heuristic approach; we try random integer/half-integer entries
    
    for target_rank in range(35, 27, -1):
        if time.time() > deadline - 10:
            break
        
        attempts = 0
        max_attempts = 50  # limited attempts per rank
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Generate random matrices with small integer entries
            # Use a mix of -2, -1, 0, 1, 2 for sparsity and simplicity
            U = np.random.randint(-2, 3, size=(n * m, target_rank))
            V = np.random.randint(-2, 3, size=(m * p, target_rank))
            W = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            if check_decomposition(U, V, W, target_rank):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # Convert to Python lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
