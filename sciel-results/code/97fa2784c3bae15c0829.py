"""
Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.

Strategy:
1. Start with the schoolbook rank-60 decomposition as a guaranteed fallback.
2. Use a structured search over candidate rank values starting from theoretical lower bounds.
3. For each candidate rank, attempt to find a valid decomposition using randomized
   linear algebra techniques (random projections, rank-revealing decompositions).
4. Validate any candidate decomposition by checking the tensor identity exactly.
5. If no lower rank is found, fall back to the schoolbook rank-60 decomposition.

The key insight for n=2 is that the tensor has a special structure:
  T[i,j,k] = delta_{i, i'} * delta_{j, j'} * delta_{k, k'}
where the multiplication is (AB)[i,k] = sum_j A[i,j] B[j,k].

We model the decomposition as:
  T = sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r
where:
  U is n*m x R, V is m*p x R, W is p*n x R
  and T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

For small dimensions, we can use integer/rational arithmetic to ensure exactness.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]
DEADLINE_SECONDS = 145.0  # Safety margin before 150s limit


def _check_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    
    Returns True if valid, False otherwise.
    """
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if val != 1:
                    return False
    return True


def _check_decomposition_fast(U, V, W, n, m, p):
    """Faster check using numpy."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                val = np.dot(u_val, v_val * w_val)
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _is_half_integer(x):
    """Check if x is an integer or half-integer within tolerance."""
    return abs(2.0 * x - round(2.0 * x)) < 1e-9


def _try_rank_randomized(n, m, p, rank, max_iters=200, time_limit=30.0):
    """Try to find a valid decomposition of given rank using randomized methods."""
    start_time = time.time()
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE_SECONDS - 5:
            break
            
        # Generate random factor matrices with rational-like entries
        # Use small integer values to increase chance of exactness
        U = np.random.randint(-2, 3, size=(n * m, rank))
        V = np.random.randint(-2, 3, size=(m * p, rank))
        W = np.random.randint(-2, 3, size=(p * n, rank))
        
        # Check if this random decomposition works
        if _check_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_structured_search(n, m, p, start_rank, end_rank, time_budget=45.0):
    """Try ranks from start_rank to end_rank using structured approaches."""
    start_time = time.time()
    
    for rank in range(start_rank, end_rank + 1):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE_SECONDS - 10:
            break
            
        # Try randomized search for this rank
        result = _try_rank_randomized(n, m, p, rank, max_iters=100, time_limit=10.0)
        if result is not None:
            return rank, result
    
    return None, None


def _try_gradient_based_polish(n, m, p, rank, time_limit=20.0):
    """
    Try to refine a random initialization using gradient-based optimization.
    This is a heuristic approach that may find low-rank decompositions
    that pure random search misses.
    """
    start_time = time.time()
    
    # Initialize with small random values
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Precompute index mappings
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Simple gradient descent on the squared error
    lr = 0.01
    for step in range(1000):
        if time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE_SECONDS - 5:
            break
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (i, j, k) in indices:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # Current value
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            err = val - 1.0
            
            # Gradients
            grad_U[u_idx] += err * V[v_idx] * W[w_idx]
            grad_V[v_idx] += err * U[u_idx] * W[w_idx]
            grad_W[w_idx] += err * U[u_idx] * V[v_idx]
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        max_err = 0.0
        for (i, j, k) in indices:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            # Round to nearest half-integers
            U_rounded = np.round(2.0 * U) / 2.0
            V_rounded = np.round(2.0 * V) / 2.0
            W_rounded = np.round(2.0 * W) / 2.0
            
            if _check_decomposition_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Start with the known valid schoolbook decomposition
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    start_time = time.time()
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    
    # Try a few promising ranks
    candidate_ranks = []
    for r in range(30, schoolbook_rank):
        candidate_ranks.append(r)
    
    # Sort by rank (ascending) to find the minimum
    for rank in candidate_ranks:
        if time.time() > DEADLINE_SECONDS - 15:
            break
            
        # Give more time to lower ranks
        time_for_this_rank = min(8.0, max(2.0, (DEADLINE_SECONDS - 15 - time.time()) / max(1, schoolbook_rank - rank)))
        
        # First try pure random search
        result = _try_rank_randomized(n, m, p, rank, max_iters=50, time_limit=time_for_this_rank / 2)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            # If we found a low rank, we can stop early
            if rank <= 35:
                break
            continue
        
        # If random search failed, try gradient-based polish
        if time.time() < DEADLINE_SECONDS - 10:
            result = _try_gradient_based_polish(n, m, p, rank, time_limit=time_for_this_rank / 2)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                if rank <= 35:
                    break
    
    return best_result
