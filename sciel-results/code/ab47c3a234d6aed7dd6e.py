import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_max = n * m * p  # 72
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook fallback
    U_fb = [[0] * rank_max for _ in range(n * m)]
    V_fb = [[0] * rank_max for _ in range(m * p)]
    W_fb = [[0] * rank_max for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_max
    
    # Precompute tensor indices for fast checking
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Strategy: structured search for low-rank decompositions
    # Key insight: for matrix multiplication, the tensor has a specific structure.
    # We can try to find decompositions where the rank is less than n*m*p.
    
    # Try various ranks from low to high
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But actually for 3x4x6, the minimum rank might be higher.
    
    # Let's try a more systematic approach:
    # 1. Try random search with various rank values
    # 2. Use the structure of the problem to guide the search
    
    # Random search with half-integer values
    np.random.seed(42)
    
    # Try ranks from 24 upward
    for r in range(24, rank_max + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations for this rank
        max_attempts = max(1, int((deadline - time.time()) / 2))
        max_attempts = min(max_attempts, 500)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random initialization with small values
            # Use a mix of strategies:
            # 1. Small random values
            # 2. Sparse random values
            # 3. Values from a specific set
            
            strategy = attempt % 3
            
            if strategy == 0:
                # Small random values in [-1, 1]
                U = np.random.uniform(-1, 1, size=(n*m, r))
                V = np.random.uniform(-1, 1, size=(m*p, r))
                W = np.random.uniform(-1, 1, size=(p*n, r))
            elif strategy == 1:
                # Sparse: most zeros, some ±1, ±0.5
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for c in range(r):
                    # Pick a few random entries
                    for _ in range(np.random.randint(1, 4)):
                        U[np.random.randint(n*m), c] = np.random.choice([-1, -0.5, 0.5, 1])
                    for _ in range(np.random.randint(1, 4)):
                        V[np.random.randint(m*p), c] = np.random.choice([-1, -0.5, 0.5, 1])
                    for _ in range(np.random.randint(1, 4)):
                        W[np.random.randint(p*n), c] = np.random.choice([-1, -0.5, 0.5, 1])
            else:
                # Structured: try to exploit the matrix structure
                # Initialize based on schoolbook but with merged terms
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                
                # Try to assign each (i,j,k) to a column with possible sharing
                # This is heuristic
                for c in range(r):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    # Sometimes add extra terms
                    if np.random.random() < 0.3:
                        i2 = np.random.randint(n)
                        j2 = np.random.randint(m)
                        k2 = np.random.randint(p)
                        U[i2*m+j2, c] += 0.5
                        V[j2*p+k2, c] += 0.5
                        W[k2*n+i2, c] += 0.5
            
            # Check if this is an exact decomposition
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Vectorized check
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_rank < rank_max:
            break
    
    return best_result
# EVOLVE-BLOCK-END
