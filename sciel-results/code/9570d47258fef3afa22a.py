"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Use random search with local improvement
    np.random.seed(42)
    
    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T."""
        # T_reconstructed[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # We need this to equal T[i,j,k]
        # Efficient check: for each c, add U[:,c] * V[:,c] * W[:,c] to result
        result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.allclose(result, T, atol=1e-10)
    
    def try_rank(r, num_trials=50):
        """Try to find a rank-r decomposition."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, r))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, r))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, r))
            
            if check_decomposition(U, V, W):
                return (U, V, W)
        
        return None
    
    # Try ranks from lower bound up
    lower_bound = max(n*m, m*p, p*n)
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        result = try_rank(r, num_trials=20)
        if result is not None:
            best_rank = r
            best_result = result
            break  # Found the minimum, stop
    
    U, V, W = best_result
    
    # Convert to lists with integer/half-integer values
    def to_list(mat):
        return [[float(v) for v in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
