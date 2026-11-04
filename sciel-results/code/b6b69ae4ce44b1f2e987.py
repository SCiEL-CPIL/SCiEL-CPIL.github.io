"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is exact within tolerance.
    The matrix multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i,j,k (since it represents the identity in the bilinear map).
    """
    rank = U.shape[1]
    # Build the tensor T of shape (n, m, p)
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank(n, m, p, target_rank, max_iterations=5000, deadline=None):
    """
    Attempt to find a decomposition of exactly target_rank using iterative refinement.
    Uses a randomized initialization and gradient-free local search.
    """
    if deadline is not None:
        if time.time() > deadline:
            return None
    
    # Strategy: start with schoolbook and try to zero out columns via row operations
    # This is a known technique: if we can find linear dependencies among the rank-1 terms,
    # we can reduce the rank.
    
    # Alternative: use the fact that for small matrices, we can set up a system of equations.
    # The tensor T is of size n*m*p = 63.
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 63 equations in r*(n*m + m*p + p*n) unknowns.
    # For r < n*m*p, this is underdetermined, so solutions exist if the rank of the
    # associated tensor allows it.
    
    # We'll use a simple approach: try to find a solution by fixing some variables and solving.
    # Given the small size, we can use a randomized local search.
    
    # Initialize with random small values
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Scale to get closer to the right magnitude
    # The product U*V*W should be ~1, so each factor should be ~1
    U = U / (np.linalg.norm(U) / np.sqrt(n * m))
    V = V / (np.linalg.norm(V) / np.sqrt(m * p))
    W = W / (np.linalg.norm(W) / np.sqrt(p * n))
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-9:
            break
        
        # Perturb one factor randomly
        factor = random.choice(['U', 'V', 'W'])
        if factor == 'U':
            row = random.randint(0, n * m - 1)
            col = random.randint(0, target_rank - 1)
            U[row, col] += random.gauss(0, 0.1)
        elif factor == 'V':
            row = random.randint(0, m * p - 1)
            col = random.randint(0, target_rank - 1)
            V[row, col] += random.gauss(0, 0.1)
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, target_rank - 1)
            W[row, col] += random.gauss(0, 0.1)
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    # The theoretical minimum rank for 3x3 * 3x7 is at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But actually, the lower bound is more subtle. Let's try from a reasonable lower bound.
    
    # Known: rank(3x3 * 3x3) = 7 (Strassen)
    # For 3x3 * 3x7, the rank is at least the rank of 3x3 * 3x3 which is 7, but likely higher.
    # Let's try ranks from 21 down to see if we can find a solution.
    
    # Actually, let's think about this differently. The matrix multiplication tensor
    # for (n,m,p) has a rank that is at least the max of the ranks of the constituent
    # sub-tensors. For 3x3 * 3x7, we can view it as 7 copies of 3x3 * 3x1, or 3 copies
    # of 3x3 * 3x3 plus some extra.
    
    # Let's just try to find any valid decomposition with rank less than n*m*p.
    # We'll try a few different strategies.
    
    # Strategy 1: Try to use Strassen-like ideas for the 3x3 block
    # This is complex, so let's stick with the search approach.
    
    # Try ranks from a lower bound up to the schoolbook rank
    # Lower bound: the rank must be at least the maximum of the "fiber" dimensions
    # which is max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, target_rank, max_iterations=2000, deadline=deadline)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break  # Found a smaller rank, stop
    
    return best_result

# EVOLVE-BLOCK-END
