"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Trivial schoolbook decomposition (rank = n*m*p = 105)
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial), dtype=np.float64)
    V_trivial = np.zeros((mp, rank_trivial), dtype=np.float64)
    W_trivial = np.zeros((pn, rank_trivial), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1
                V_trivial[b, col] = 1
                W_trivial[c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = rank_trivial
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_computed, T_target, atol=1e-9)
    
    # Try to find lower-rank decompositions
    # Use structured random search with small integer/half-integer entries
    rng = np.random.default_rng(42)
    
    # Try ranks from best_rank-1 down to a reasonable lower bound
    # Lower bound: at least max(nm, mp, pn) = 35
    lower_bound = max(nm, mp, pn)
    
    for target_rank in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations for this rank
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Generate random matrices with small integer entries
            # Use entries from {-1, 0, 1} for simplicity
            U = rng.integers(-1, 2, size=(nm, target_rank)).astype(np.float64)
            V = rng.integers(-1, 2, size=(mp, target_rank)).astype(np.float64)
            W = rng.integers(-1, 2, size=(pn, target_rank)).astype(np.float64)
            
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                # Continue searching for even lower rank
                break
        
        # If we found a better rank, continue to next lower rank
        # If not, we could try more trials, but keep it bounded
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
