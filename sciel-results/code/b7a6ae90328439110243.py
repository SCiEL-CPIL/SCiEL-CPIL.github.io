"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has indices (i, j, k) with T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Naive rank-60 decomposition
    rank_naive = n * m * p  # 60
    U_naive = np.zeros((n*m, rank_naive))
    V_naive = np.zeros((m*p, rank_naive))
    W_naive = np.zeros((p*n, rank_naive))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m + j, r] = 1
                V_naive[j*p + k, r] = 1
                W_naive[k*n + i, r] = 1
                r += 1
    
    best_rank = rank_naive
    best_decomp = (U_naive.copy(), V_naive.copy(), W_naive.copy())
    
    def verify_fast(U, V, W, tol=1e-9):
        """Fast vectorized verification."""
        n, m, p = DIMENSIONS
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_all(U, V, W, tol=1e-9):
        """Verify all (i,j,k) triples."""
        n, m, p = DIMENSIONS
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Use tensor train / HOOI-inspired rank reduction
    # Start from naive decomposition and iteratively try to reduce rank
    
    # First, try the Strassen-based block approach for structured reduction
    # Split j into pairs and use 7-product Strassen-like schemes
    
    def try_strassen_blocks():
        """Try Strassen-based block decomposition."""
        n, m, p = DIMENSIONS
        
        # Split j = {0,1,2,3,4} into pairs: (0,1), (2,3), and singleton {4}
        # Split k = {0,1,2,3,4,5} into pairs: (0,1), (2,3), (4,5)
        
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        # For each (j_pair, k_pair), use 7 Strassen products for the 2x2 block
        # Plus handle the singleton j=4 with naive 2*6=12 terms
        
        # Total: 2 * 3 * 7 + 12 = 42 + 12 = 54
        target_rank = 2 * len(j_pairs) * len(k_pairs) * 7 + n * p  # 42 + 12 = 54
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen products for 2x2 * 2x2 block
                # We're computing C[i, k0..k1] for i in {0,1} using A[:, j0..j1] * B[j0..j1, :]
                
                # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
                if col < target_rank:
                    U[0*m+j0, col] += 1
                    U[1*m+j1, col] += 1
                    V[j0*p+k0, col] += 1
                    V[j1*p+k1, col] += 1
                    W[k0*n+0, col] += 1
                    W[k1*n+1, col] += 1
                    col += 1
                
                # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
                if col < target_rank:
                    U[1*m+j0, col] += 1
                    U[1*m+j1, col] += 1
                    V[j0*p+k0, col] += 1
                    W[k0*n+1, col] += 1
                    W[k1*n+1, col] -= 1
                    col += 1
                
                # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
                if col < target_rank:
                    U[0*m+j0, col] += 1
                    V[j0*p+k1, col] += 1
                    V[j1*p+k1, col] -= 1
                    W[k1*n+0, col] += 1
                    W[k1*n+1, col] += 1
                    col += 1
                
                # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
                if col < target_rank:
                    U[1*m+j1, col] += 1
                    V[j1*p+k0, col] += 1
                    V[j0*p+k0, col] -= 1
                    W[k0*n+0, col] += 1
                    W[k0*n+1, col] += 1
                    col += 1
                
                # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
                if col < target_rank:
                    U[0*m+j0, col] += 1
                    U[0*m+j1, col] += 1
                    V[j1*p+k1, col] += 1
                    W[k0*n+0, col] -= 1
                    W[k1*n+0, col] += 1
                    col += 1
                
                # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
                if col < target_rank:
                    U[1*m+j0, col] += 1
                    U[0*m+j0, col] -= 1
                    V[j0*p+k0, col] += 1
                    V[j0*p+k1, col] += 1
                    W[k1*n+1, col] += 1
                    col += 1
                
                # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
                if col < target_rank:
                    U[0*m+j1, col] += 1
                    U[1*m+j1, col] -= 1
                    V[j1*p+k0, col] += 1
                    V[j1*p+k1, col] += 1
                    W[k0*n+0, col] += 1
                    col += 1
        
        # Handle j=4 (singleton) with naive terms
        for k in range(p):
            if col < target_rank:
                U[0*m+4, col] += 1
                V[4*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
            if col < target_rank:
                U[1*m+4, col] += 1
                V[4*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        if col == target_rank and verify_fast(U, V, W):
            return target_rank, U.copy(), V.copy(), W.copy()
        return None
    
    # Try the Strassen block approach
    result = try_strassen_blocks()
    if result is not None:
        best_rank, U_best, V_best, W_best = result
        best_decomp = (U_best, V_best, W_best)
    
    # Now try to reduce further using tensor train / iterative methods
    # Strategy: Try to find linear dependencies among the rank-1 components
    
    # For each candidate rank r from best_rank-1 down to 30:
    #   Try to find U, V, W of rank r that satisfy the tensor equation
    
    # Use a structured approach: 
    # 1. Start from the current best decomposition
    # 2. Try to merge pairs of terms by finding linear combinations
    
    # Alternative: Use the fact that we can absorb terms
    # If we have terms r1 and r2, we can try to replace them with a single term
    # if the products align appropriately
    
    # Let's try a more systematic approach using random initialization + verification
    
    rng = np.random.default_rng(123)
    
    # Try random rank-r decompositions
    # Initialize with random values, then check if they work
    # This is unlikely to work directly, but we can use it as a starting point
    
    # Better: Use the block structure more cleverly
    # Try different groupings of j and k indices
    
    # Try grouping j into (0,1,2), (3,4) and k into (0,1,2,3), (4,5)
    # Or other variations
    
    # For now, let's try to reduce the Strassen result further
    # by looking for linear dependencies
    
    # If we have a valid decomposition of rank R, we can try to find
    # a lower-rank one by checking if any rank-1 component is redundant
    
    # A rank-1 component r is redundant if removing it and adjusting
    # the remaining components can still satisfy the tensor equation
    
    # This is complex, so let's try a simpler approach:
    # Randomly perturb the decomposition and see if we can reduce rank
    
    # Actually, let's try a different structured approach:
    # Use 3x3 Strassen-like schemes where applicable
    
    # For 2x5 * 5x6, we can view it as:
    # - 2x2 * 2x2 blocks (handled above)
    # - Remaining terms
    
    # Let's try to use a 3-term scheme for some blocks
    
    # Try: Can we do better than 54?
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But for matrix multiplication, the rank is at least n*m*p / max(n,m,p)^2 or similar
    
    # Let's try to find rank 50, 48, 45, etc.
    
    # Strategy: Use random linear combinations of the existing terms
    # to try to absorb some terms into others
    
    if best_rank <= 54 and time.time() < DEADLINE - 30:
        # Try to reduce from 54 to lower ranks
        for target_rank in range(best_rank - 1, 29, -1):
            if time.time() > DEADLINE - 25:
                break
            
            # Try to find a rank-target_rank decomposition
            # by randomly merging terms from the current best
            
            U_curr, V_curr, W_curr = best_decomp
            
            # Try random mergers: pick two terms and try to combine them
            # This is heuristic and may not work, but worth trying
            
            # Alternative: Use the fact that we can have non-integer entries
            # Try to find U, V, W with half-integer entries
            
            # For now, just try random rank-target_rank with structured init
            attempts = 0
            max_attempts = 20
            
            while attempts < max_attempts and time.time() < DEADLINE - 20:
                attempts += 1
                
                # Start from current best, randomly zero out some columns
                # and try to adjust the remaining ones
                
                if target_rank >= 40:
                    # Try to absorb (best_rank - target_rank) terms into existing ones
                    U_try = U_curr[:, :target_rank].copy()
                    V_try = V_curr[:, :target_rank].copy()
                    W_try = W_curr[:, :target_rank].copy()
                    
                    # Add contributions from removed terms
                    for r_rem in range(target_rank, best_rank):
                        # Try to absorb into a random existing term
                        r_target = rng.integers(0, target_rank)
                        U_try[:, r_target] += U_curr[:, r_rem]
                        V_try[:, r_target] += V_curr[:, r_rem]
                        W_try[:, r_target] += W_curr[:, r_rem]
                    
                    if verify_fast(U_try, V_try, W_try):
                        best_rank = target_rank
                        best_decomp = (U_try.copy(), V_try.copy(), W_try.copy())
                        break
    
    # Use the best decomposition found
    U_best, V_best, W_best = best_decomp
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U_best]
    V_list = [[float(x) for x in row] for row in V_best]
    W_list = [[float(x) for x in row] for row in W_best]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
