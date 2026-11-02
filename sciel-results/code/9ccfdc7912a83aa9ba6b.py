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
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_computed, T_target, atol=1e-9)
    
    # Try to find lower rank decompositions
    # Lower bound is max(nm, mp, pn) = 35
    lower_bound = max(nm, mp, pn)
    
    # Try ranks from lower_bound up to best_rank-1
    # For each rank, try random initializations
    rng = np.random.default_rng(42)
    
    # Try a few ranks, starting from a reasonable estimate
    # For 3x5 x 5x7, let's try ranks in a smart range
    # The minimum is at least max(15,35,21)=35, but likely higher
    
    # Try ranks from 50 to best_rank-1, with more trials for lower ranks
    for target_rank in range(max(lower_bound, 40), best_rank):
        if time.time() > deadline - 20:
            break
        
        # More trials for lower ranks
        max_trials = 50 if target_rank < 80 else 20
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integer entries
            # Use entries from {-1, 0, 1} with some sparsity
            U = rng.integers(-1, 2, size=(nm, target_rank)).astype(np.float64)
            V = rng.integers(-1, 2, size=(mp, target_rank)).astype(np.float64)
            W = rng.integers(-1, 2, size=(pn, target_rank)).astype(np.float64)
            
            # Add some sparsity
            for mat in [U, V, W]:
                mask = rng.random(mat.shape) < 0.5
                mat[~mask] = 0
            
            if verify(U, V, W):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                break
        
        # If we found a decomposition at this rank, we can stop
        # since we're going from low to high
        if best_rank == target_rank:
            break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
