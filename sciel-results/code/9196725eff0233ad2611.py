"""
Matrix multiplication 3x5 x 5x7: rank-minimal tensor decomposition.

The schoolbook rank is 105. We attempt to find a lower-rank exact decomposition.
Strategy:
1. Start with the trivial rank-105 decomposition.
2. Use structured search: try to find linear dependencies among the 105 rank-1 terms.
3. Use a smarter approach: exploit the structure of matrix multiplication.

Key insight: The matrix multiplication tensor T has entries T[i,j,k] = 1 if and only if
the indices are consistent with matrix multiplication (i.e., T[a,b,c]=1 where
a = i*m+j, b = j*p+k, c = k*n+i).

For a decomposition of rank r, we need matrices U (nm x r), V (mp x r), W (pn x r)
such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c].

We use an iterative approach:
1. Start from the trivial decomposition.
2. Try to find linear combinations that allow removing terms.
3. Use random restarts with structured patterns.
"""
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
    
    def verify(U, V, W, atol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_computed, T_target, atol=atol)
    
    def verify_fast(U, V, W):
        """Faster verification using sparse structure."""
        # Only check non-zero entries of T_target
        indices = np.argwhere(T_target != 0)
        for a, b, c in indices:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-8:
                return False
        # Also check some zero entries to ensure no false positives
        # Sample a few zero entries
        zero_indices = np.argwhere(T_target == 0)
        if len(zero_indices) > 0:
            sample = zero_indices[np.random.choice(len(zero_indices), size=min(50, len(zero_indices)), replace=False)]
            for a, b, c in sample:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-8:
                    return False
        return True
    
    def try_reduce_by_dependency(U, V, W, current_rank, rng, max_trials=200):
        """Try to reduce rank by finding linear dependencies among terms."""
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            if best_rank <= 2:
                break
            
            # Try to find two terms that can be merged
            # Pick two random columns
            c1, c2 = rng.choice(best_rank, size=2, replace=False)
            
            # Try to see if we can eliminate one of them by adjusting others
            # This is a heuristic approach
            
            # Strategy: try to find alpha such that removing column c2 and 
            # adding alpha*column c1 to other columns still works
            # This is complex, so use a simpler approach:
            
            # Try random perturbations of entries
            new_U = best_U.copy()
            new_V = best_V.copy()
            new_W = best_W.copy()
            
            # Try to zero out some entries
            for _ in range(3):
                mat_choice = rng.integers(0, 3)
                if mat_choice == 0:
                    row, col = rng.integers(0, new_U.shape[0]), rng.integers(0, new_U.shape[1])
                    new_U[row, col] = 0
                elif mat_choice == 1:
                    row, col = rng.integers(0, new_V.shape[0]), rng.integers(0, new_V.shape[1])
                    new_V[row, col] = 0
                else:
                    row, col = rng.integers(0, new_W.shape[0]), rng.integers(0, new_W.shape[1])
                    new_W[row, col] = 0
            
            if verify_fast(new_U, new_V, new_W):
                # Check if we can remove a column
                for col_to_remove in range(new_U.shape[1]):
                    if np.all(new_U[:, col_to_remove] == 0) or \
                       np.all(new_V[:, col_to_remove] == 0) or \
                       np.all(new_W[:, col_to_remove] == 0):
                        reduced_U = np.delete(new_U, col_to_remove, axis=1)
                        reduced_V = np.delete(new_V, col_to_remove, axis=1)
                        reduced_W = np.delete(new_W, col_to_remove, axis=1)
                        if verify_fast(reduced_U, reduced_V, reduced_W):
                            best_U = reduced_U
                            best_V = reduced_V
                            best_W = reduced_W
                            best_rank -= 1
                            break
        
        return best_U, best_V, best_W, best_rank
    
    def try_structured_reduction(U, V, W, current_rank, rng):
        """Try structured reductions based on matrix multiplication structure."""
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        # For matrix multiplication, there are known algebraic identities
        # that can reduce the rank. We try to exploit them.
        
        # One approach: group terms by (i,j) pairs and try to find
        # linear combinations over k
        
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            
            if best_rank <= 2:
                break
            
            # Try to find a linear dependency among terms with same (i,j)
            # but different k values
            
            # Pick a random (i,j) pair
            i, j = rng.integers(0, n), rng.integers(0, m)
            
            # Find all columns corresponding to this (i,j)
            cols = []
            for col in range(best_rank):
                a = i * m + j
                if abs(best_U[a, col]) > 1e-8:
                    cols.append(col)
            
            if len(cols) > 1:
                # Try to find a linear combination
                # This is a simplified approach
                pass
        
        return best_U, best_V, best_W, best_rank
    
    # Main search loop
    rng = np.random.default_rng(42)
    
    # First, try to reduce from the trivial decomposition
    best_U, best_V, best_W, best_rank = try_reduce_by_dependency(
        best_U, best_V, best_W, best_rank, rng, max_trials=100
    )
    
    # Try structured reductions
    best_U, best_V, best_W, best_rank = try_structured_reduction(
        best_U, best_V, best_W, best_rank, rng
    )
    
    # Try random restarts with lower target ranks
    lower_bound = max(nm, mp, pn)
    
    for target_rank in range(best_rank - 1, max(lower_bound, 90) - 1, -1):
        if time.time() > deadline - 20:
            break
        
        max_trials = 20
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Create a sparse random initialization
            U = np.zeros((nm, target_rank), dtype=np.float64)
            V = np.zeros((mp, target_rank), dtype=np.float64)
            W = np.zeros((pn, target_rank), dtype=np.float64)
            
            # Initialize with structured patterns
            for r in range(target_rank):
                # Each rank-1 term should correspond to some (i,j,k) pattern
                i, j, k = rng.integers(0, n), rng.integers(0, m), rng.integers(0, p)
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                # Add some randomization
                U[a, r] = rng.choice([-1, 1])
                V[b, r] = rng.choice([-1, 1])
                W[c, r] = rng.choice([-1, 1])
                
                # Add some additional non-zero entries with small probability
                if rng.random() < 0.2:
                    extra_a = rng.integers(0, nm)
                    U[extra_a, r] = rng.choice([-1, 1]) * 0.5
                if rng.random() < 0.2:
                    extra_b = rng.integers(0, mp)
                    V[extra_b, r] = rng.choice([-1, 1]) * 0.5
                if rng.random() < 0.2:
                    extra_c = rng.integers(0, pn)
                    W[extra_c, r] = rng.choice([-1, 1]) * 0.5
            
            if verify_fast(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to trivial decomposition
        best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
        best_rank = rank_trivial
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
