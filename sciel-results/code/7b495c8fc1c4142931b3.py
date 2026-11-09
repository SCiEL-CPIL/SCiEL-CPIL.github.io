"""Program attempting to find a lower-rank exact tensor decomposition for 3x3*3x5 matrix multiplication.
The rank of the tensor is the minimum number of rank-1 terms in an exact decomposition.
Schoolbook gives rank n*m*p = 3*3*5 = 45. Known bounds: rank(3x3x3)=23 (Landsberg).
For 3x3x5, no better than 45 is known in this context, but we attempt a search."""

import numpy as np
import random
import math
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a dictionary with keys 'U', 'V', 'W' containing factor matrices
    of shapes (n*m, rank), (m*p, rank), (p*n, rank) such that the tensor
    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    holds for all i in [0,n), j in [0,m), k in [0,p).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Start with schoolbook rank 45
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m + j*p + k  # linear index for (i,j,k)
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    # Verify schoolbook is correct
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(actual - expected) > tol:
                        return False
        return True

    # Attempt to find a lower rank using random initialization and gradient-free local search
    # This is a heuristic; may not find anything better than schoolbook.
    # We'll try ranks from 44 down to 30, with limited attempts.
    
    def random_decomposition(rank, seed=None):
        """Create a random initial decomposition with the given rank."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.default_rng()
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
        # Scale to reasonable magnitude
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        return U, V, W

    def compute_residuals(U, V, W):
        """Compute the residuals for all (i,j,k) entries."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residuals[i,j,k] = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
        return residuals

    def residual_norm(residuals):
        """Compute the max absolute residual."""
        return np.max(np.abs(residuals - 1.0))

    def local_search(U, V, W, max_iter=500, tol=1e-12):
        """Simple coordinate descent / local search to reduce residuals."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        best_resnorm = residual_norm(compute_residuals(U, V, W))
        
        for it in range(max_iter):
            improved = False
            # Try small perturbations in random directions
            for _ in range(10):
                dU = np.random.standard_normal((n*m, rank)) * 0.01
                dV = np.random.standard_normal((m*p, rank)) * 0.01
                dW = np.random.standard_normal((p*n, rank)) * 0.01
                new_U = U + dU
                new_V = V + dV
                new_W = W + dW
                new_resnorm = residual_norm(compute_residuals(new_U, new_V, new_W))
                if new_resnorm < best_resnorm - 1e-15:
                    U, V, W = new_U, new_V, new_W
                    best_resnorm = new_resnorm
                    improved = True
                    break
            if not improved:
                # Try coordinate-wise adjustments
                for idx in range(min(50, U.size)):
                    axis, pos = np.unravel_index(idx, U.shape)
                    orig = U[axis, pos]
                    for delta in [0.01, -0.01, 0.005, -0.005]:
                        U[axis, pos] = orig + delta
                        resnorm = residual_norm(compute_residuals(U, V, W))
                        if resnorm < best_resnorm - 1e-15:
                            best_resnorm = resnorm
                            improved = True
                            break
                        U[axis, pos] = orig
                if not improved:
                    break
            if best_resnorm < tol:
                break
        return U, V, W, best_resnorm

    # Try to find a rank-44 or lower decomposition
    # This is very unlikely to succeed with random search, but we try.
    # We'll spend limited time on each rank attempt.
    time_per_rank = 10.0  # seconds per rank attempt
    
    for target_rank in range(rank_schoolbook - 1, 29, -1):
        if time.time() - start_time > deadline - time_per_rank:
            break
        
        # Try a few random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline:
                break
            seed = attempt * 1000 + target_rank
            U, V, W = random_decomposition(target_rank, seed=seed)
            
            # Run local search with limited iterations
            U, V, W, resnorm = local_search(U, V, W, max_iter=200, tol=1e-12)
            
            if resnorm < 1e-10:
                # Found a valid decomposition with lower rank!
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                print(f"Found rank-{target_rank} decomposition!")
                break
        else:
            # Continue to next rank
            continue
        break  # Exit the rank loop if we found a better one

    # If we still have the schoolbook, return it
    # Convert to lists as required
    def to_list(np_array):
        return np_array.tolist()

    if best_rank == rank_schoolbook:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    else:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
