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
    
    def try_reduce_rank(U, V, W, current_rank, rng, max_trials=100):
        """Try to reduce rank by finding linear dependencies."""
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        for _ in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            # Try to find a linear combination of two columns that can be eliminated
            # or try random perturbations
            
            # Strategy 1: Try to merge two terms by finding if their sum/difference
            # can be expressed with fewer terms
            if best_rank <= 2:
                break
            
            # Pick two random columns to try to merge
            c1, c2 = rng.choice(best_rank, size=2, replace=False)
            
            # Try to see if we can replace these two columns with one
            # by finding alpha, beta such that the combination works
            # This is complex, so try a simpler approach:
            
            # Try random sign flips and see if we can find a dependency
            # Actually, let's try a different approach: use structured patterns
            
            # For now, just try random small modifications
            new_U = best_U.copy()
            new_V = best_V.copy()
            new_W = best_W.copy()
            
            # Try to zero out some entries and see if it still works
            # This is too naive. Let's try a better approach.
            
            # Approach: Try to find if there's a linear relation among the terms
            # by checking if we can remove one term and adjust others
            
            # For efficiency, let's try specific structured reductions
            # based on the matrix multiplication structure
            
            # Skip this trial for now, use a different strategy
            break
        
        return best_U, best_V, best_W, best_rank
    
    # Better strategy: Use known algebraic structures
    # For 3x5 x 5x7, try block decomposition
    
    # Strategy: Try to use a decomposition that exploits the structure
    # of matrix multiplication more effectively
    
    # Let's try a systematic search with better initialization
    rng = np.random.default_rng(42)
    
    # Try different ranks with better structured initialization
    lower_bound = max(nm, mp, pn)
    
    # First, try to find if rank 104 is possible by small modifications
    for target_rank in range(best_rank - 1, max(lower_bound, 80) - 1, -1):
        if time.time() > deadline - 20:
            break
        
        max_trials = 30
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Start from trivial decomposition and try to reduce
            # by finding linear dependencies
            
            # Create a modified version with one term removed
            # and try to adjust the others
            
            # For now, use random search but with better entry range
            # Use entries from {-2, -1, 0, 1, 2}
            U = rng.integers(-2, 3, size=(nm, target_rank)).astype(np.float64)
            V = rng.integers(-2, 3, size=(mp, target_rank)).astype(np.float64)
            W = rng.integers(-2, 3, size=(pn, target_rank)).astype(np.float64)
            
            # Sparsify: keep only a fraction of non-zero entries
            for mat in [U, V, W]:
                mask = rng.random(mat.shape) < 0.3
                mat[~mask] = 0
            
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
