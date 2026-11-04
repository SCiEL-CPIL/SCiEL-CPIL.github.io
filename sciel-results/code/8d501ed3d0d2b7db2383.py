"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _embedding_init(a, b, n, dim=4):
    """
    Initialize permutation via continuous graph embedding.
    Minimize ||A - P^T B P||_F by optimizing coordinates, then extract permutation.
    """
    # Center and scale matrices for numerical stability
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    
    # Symmetrize
    a_sym = (a_centered + a_centered.T) / 2.0
    b_sym = (b_centered + b_centered.T) / 2.0
    
    # Get eigendecomposition for initialization
    try:
        w_a, v_a = np.linalg.eigh(a_sym)
        w_b, v_b = np.linalg.eigh(b_sym)
        
        # Use most negative eigenvectors
        idx_a = np.argsort(w_a)[:dim]
        idx_b = np.argsort(w_b)[:dim]
        
        X = v_a[:, idx_a].copy()  # facility coordinates
        Y = v_b[:, idx_b].copy()  # location coordinates
        
        # Normalize
        X /= (np.linalg.norm(X) + 1e-10)
        Y /= (np.linalg.norm(Y) + 1e-10)
        
    except Exception:
        X = np.random.randn(n, dim)
        Y = np.random.randn(n, dim)
    
    # Gradient descent to minimize trace(X^T A Y Y^T A^T X) - 2*trace(X^T A Y Y^T B Y Y^T A^T X) + ...
    # Simplified: minimize ||X^T A Y - Y^T B X||_F^2
    # This is equivalent to minimizing the QAP cost in continuous space
    
    lr = 0.01
    for step in range(1000):
        # Compute gradients
        # L = ||X^T A Y - Y^T B X||_F^2
        # dL/dX = -2 * A Y (Y^T B X - X^T A Y)^T ... this gets complex
        # Use a simpler approach: align the two matrices
        
        # Project onto orthogonal complement to remove rotation
        # Use Procrustes-like alignment
        
        # Compute the "mismatch"
        M = X.T @ a_sym @ Y  # n x n
        N = Y.T @ b_sym @ X  # n x n
        
        # Gradient for X: -2 * a_sym @ Y @ (M - N).T
        # Gradient for Y: -2 * b_sym @ X @ (M.T - N)
        
        grad_X = -2.0 * a_sym @ Y @ (M - N).T
        grad_Y = -2.0 * b_sym @ X @ (M.T - N)
        
        # Update
        X = X - lr * grad_X
        Y = Y - lr * grad_Y
        
        # Re-orthogonalize (remove mean)
        X -= np.mean(X, axis=0, keepdims=True)
        Y -= np.mean(Y, axis=0, keepdims=True)
        
        # Normalize to prevent blow-up
        norm_X = np.linalg.norm(X)
        norm_Y = np.linalg.norm(Y)
        if norm_X > 1e10 or norm_Y > 1e10:
            break
        X /= (norm_X + 1e-10)
        Y /= (norm_Y + 1e-10)
    
    # Extract permutation from nearest-neighbor mapping
    # For each facility i, find the location j that minimizes distance in embedding space
    # This is an assignment problem
    
    # Compute distance matrix between X and Y
    # d[i, j] = ||X[i] - Y[j]||^2
    X2 = np.sum(X**2, axis=1)[:, np.newaxis]
    Y2 = np.sum(Y**2, axis=1)[np.newaxis, :]
    d = X2 + Y2 - 2.0 * X @ Y.T
    
    # Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(d)
    p = np.zeros(n, dtype=np.int64)
    p[col_ind] = row_ind
    
    return p

def _row_sum_init(a, b, n):
    """
    Simple initialization based on row sums.
    High-flow facilities go to low-distance locations.
    """
    flow_sums = np.sum(a, axis=1)
    dist_sums = np.sum(b, axis=1)
    
    # Sort facilities by flow (descending) and locations by distance (ascending)
    fac_order = np.argsort(flow_sums)[::-1]
    loc_order = np.argsort(dist_sums)
    
    p = np.zeros(n, dtype=np.int64)
    for i, fac in enumerate(fac_order):
        p[fac] = loc_order[i]
    
    return p

def _hybrid_init(a, b, n):
    """
    Combine multiple initialization strategies and pick the best.
    """
    inits = []
    
    # Embedding-based
    p_emb = _embedding_init(a, b, n)
    inits.append(p_emb)
    
    # Row sum-based
    p_rs = _row_sum_init(a, b, n)
    inits.append(p_rs)
    
    # Random
    p_rand = np.arange(n, dtype=np.int64)
    np.random.shuffle(p_rand)
    inits.append(p_rand)
    
    # Pick the one with lowest initial cost
    best_p = None
    best_cost = np.inf
    for p in inits:
        cost = _compute_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    return best_p

def _fast_delta(a, D, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    D is the current distance matrix b[p, p].
    """
    s1 = np.dot(a[i], D[j]) - np.dot(a[i], D[i])
    s2 = np.dot(a[:, i], D[:, j]) - np.dot(a[:, i], D[:, i])
    s3 = np.dot(a[j], D[i]) - np.dot(a[j], D[j])
    s4 = np.dot(a[:, j], D[:, i]) - np.dot(a[:, j], D[:, j])
    
    total = s1 + s2 + s3 + s4
    
    ki = a[i,i]*(D[j,i]-D[i,i]) + a[i,i]*(D[i,j]-D[i,i]) + a[j,i]*(D[i,i]-D[j,i]) + a[i,j]*(D[i,i]-D[i,j])
    
    kj = a[i,j]*(D[j,j]-D[i,j]) + a[j,i]*(D[j,j]-D[j,i]) + a[j,j]*(D[i,j]-D[j,j]) + a[j,j]*(D[j,i]-D[j,j])
    
    total -= ki + kj
    
    direct = a[i,j]*D[j,i] + a[j,i]*D[i,j] - a[i,j]*D[i,j] - a[j,i]*D[j,i]
    
    total += direct
    
    return total

def _local_search_2opt_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _fast_delta(a, D, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    current_cost += delta
                    improved = True
                    
    return p, current_cost

def _perturb(p, k=5):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Use hybrid initialization
    p_init = _hybrid_init(a, b, n)
    
    # First local search on hybrid init
    p_ls, cost_ls = _local_search_2opt_fast(a, b, p_init, time_budget - 5, start_time)
    best_p = p_ls
    best_cost = cost_ls
    
    # Iterated Local Search
    while time.time() - start_time < time_budget - 5:
        # Perturb
        p_pert = _perturb(best_p, k=5)
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_pert, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
