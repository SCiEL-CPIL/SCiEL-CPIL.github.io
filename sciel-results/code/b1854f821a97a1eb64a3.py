"""
Strategy: Continuous Optimization (L-BFGS-B) + Discrete Refinement.

1. Precompute the AP incidence matrix M (sparse or dense).
2. Define a smooth surrogate loss: L(x) = || M @ x ||_2^2, where x is in [-1, 1]^N.
3. Use scipy.optimize.minimize (L-BFGS-B) to find a continuous vector x that minimizes L.
   This helps navigate the landscape by using gradient information, potentially finding
   better basins than random discrete starts.
4. Discretize x to s (signs of x_i).
5. Perform discrete local search (greedy flips) to refine s, starting from the 
   discretized continuous solution.
6. Return the best discrete solution found.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 256

def _get_ap_indices():
    """Precompute all arithmetic progressions in [1, N] as a list of index lists."""
    aps = []
    for step in range(1, N // 2 + 1):
        for start in range(1, N - step + 1):
            max_len = (N - start) // step + 1
            if max_len < 2:
                continue
            indices = [start - 1 + k * step for k in range(max_len)]
            for length in range(2, max_len + 1):
                aps.append(indices[:length])
    return aps

def _precompute_ap_matrix(aps):
    """
    Build a dense matrix M where M[i, j] = 1 if index j is in AP i.
    For N=256, the number of APs is roughly 50,000-100,000.
    A dense matrix of size (num_aps, 256) with float32 is ~100MB, which is acceptable.
    """
    num_aps = len(aps)
    M = np.zeros((num_aps, N), dtype=np.float32)
    for i, idxs in enumerate(aps):
        for idx in idxs:
            M[i, idx] = 1.0
    return M

def _objective_and_gradient(x, M):
    """
    Objective: L(x) = sum( (M @ x)^2 )
    Gradient: dL/dx = 2 * M^T @ (M @ x)
    """
    sums = M @ x
    loss = np.sum(sums ** 2)
    grad = 2.0 * (M.T @ sums)
    return loss, grad

def _compute_max_abs_sum(signs, M):
    """Compute max abs sum for a discrete sign vector."""
    sums = M @ signs
    return np.max(np.abs(sums))

def _find_best_flip(signs, M, current_max):
    """
    Find the single flip that reduces the max abs sum the most.
    """
    S = M @ signs
    best_k = -1
    best_max = current_max
    
    for k in range(N):
        delta = 2.0 * signs[k] * M[:, k]
        new_S = S + delta
        new_max = np.max(np.abs(new_S))
        if new_max < best_max:
            best_max = new_max
            best_k = k
            
    return best_k, best_max

def construct():
    start_time = time.time()
    # 90s limit. Reserve 5s for safety and final discrete refinement.
    deadline_continuous = start_time + 60.0
    deadline_discrete = start_time + 85.0
    
    # 1. Precomputation
    aps = _get_ap_indices()
    M = _precompute_ap_matrix(aps)
    
    # 2. Continuous Optimization Phase
    # Initialize with a few different starting points to avoid local minima in continuous space.
    # Common initializations: random, Thue-Morse, Alternating.
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # Random
    for _ in range(3):
        x0 = rng.uniform(-1, 1, size=N).astype(np.float32)
        initializations.append(x0)
        
    # Thue-Morse
    tm = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)], dtype=np.float32)
    initializations.append(tm)
    
    # Alternating
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)], dtype=np.float32)
    initializations.append(alt)
    
    best_x = None
    best_continuous_loss = float('inf')
    
    # Run L-BFGS-B on each initialization until time limit or convergence
    for i, x0 in enumerate(initializations):
        if time.time() > deadline_continuous:
            break
            
        try:
            res = minimize(
                _objective_and_gradient,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            loss = res.fun
            if loss < best_continuous_loss:
                best_continuous_loss = loss
                best_x = res.x
        except Exception:
            continue
            
    # 3. Discretization
    if best_x is not None:
        # Sign function: map >0 to 1, <=0 to -1
        discrete_signs = np.where(best_x > 0, 1.0, -1.0).astype(np.float32)
    else:
        # Fallback to random if optimization failed
        discrete_signs = rng.choice([-1.0, 1.0], size=N).astype(np.float32)
    
    # 4. Discrete Local Search Refinement
    # Start from the discretized continuous solution and refine with greedy flips.
    current_signs = discrete_signs
    current_max = _compute_max_abs_sum(current_signs, M)
    
    # Also keep track of the best result from the continuous phase discretized directly
    # (though the refinement should only improve it)
    
    # Run discrete local search
    # We can afford several iterations here as each flip check is fast (~1ms)
    # and we have ~25s left.
    
    while time.time() < deadline_discrete:
        best_k, new_max = _find_best_flip(current_signs, M, current_max)
        
        if best_k == -1:
            # Local minimum reached
            break
            
        # Apply flip
        current_signs[best_k] *= -1.0
        current_max = new_max
        
        # Optional: Add a simple perturbation strategy if stuck?
        # For now, single flip is the baseline. If we get stuck, we could try a random restart
        # from the continuous solution or a random point, but time is tight.
        # Given the continuous start is likely in a good basin, greedy should converge fast.
        
    # 5. Final Check and Return
    # Ensure we return Python integers
    result = [int(x) for x in current_signs]
    
    return result

# EVOLVE-BLOCK-END
