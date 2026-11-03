import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Calculate the QAP cost for a given permutation."""
    # p is a permutation of indices 0..n-1
    # Cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
    # This can be computed as trace(A @ P @ B @ P.T) where P is the permutation matrix
    # Or simply by indexing.
    # For n=256, direct indexing sum is O(n^2) which is fast.
    # a[i,j] * b[p[i], p[j]]
    # We can vectorize:
    # b_perm = b[np.ix_(p, p)]
    # return np.sum(a * b_perm)
    b_perm = b[np.ix_(p, p)]
    return np.sum(a * b_perm)

def _rank_matching(a, b):
    """Generate initial permutation using rank matching."""
    n = a.shape[0]
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _two_opt(a, b, p, max_iters=10000):
    """Perform 2-opt local search."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Randomly sample pairs to speed up, or check all if n is small
        # For n=256, all pairs is 256*255/2 ~ 32k checks.
        # Each check involves swapping p[i], p[j] and calculating delta cost.
        # Delta cost for swapping i and j:
        # Cost change = sum_{k != i,j} a[i,k]b[p[j],p[k]] + a[k,i]b[p[k],p[j]] + ...
        # It's efficient to precompute or use the formula.
        # Let's just recalculate cost for simplicity first, if too slow, optimize.
        # 32k * 256^2 ops is too slow (2e9).
        # We need incremental cost update.
        
        # Let's implement efficient delta calculation for 2-opt swap (i, j)
        # Let P be the current permutation matrix.
        # Cost = Tr(A P B P^T)
        # Swapping i and j in P means P' = P S where S is swap matrix.
        # This is complex.
        # Alternative: 
        # C_new = C_old - (terms involving i,j in old pos) + (terms in new pos)
        
        # Let's use a simpler heuristic: Random restarts with Greedy Construction or just simple 2-opt with random pairs.
        # Given the time constraint, let's stick to the ADMM/PGD approach which is O(n^3) per step.
        # 1 step PGD: Gradient O(n^3), Hungarian O(n^3).
        # 100 steps = 100 * 2 * 1.6e7 = 3.2e9 ops. Might be too slow in Python/NumPy if not optimized.
        # NumPy matmul is C-optimized. 256x256 matmul is very fast (~1ms).
        # Hungarian is ~10-20ms.
        # So 100 steps is ~2 seconds. We have 120s.
        # We can do ~5000 steps.
        
        pass # Skipping 2-opt implementation for now, relying on PGD steps.
    
    return p, current_cost

def solve(a, b):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = a.shape[0]
    
    # 1. Initialization
    p_init = _rank_matching(a, b)
    best_p = p_init.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Soft assignment matrix X
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), best_p] = 1.0
    
    # Learning rate
    lr = 0.05
    # Decay learning rate over time
    lr_decay = 0.99
    
    while time.time() < deadline:
        # Compute Gradient: G = A X B + A^T X B^T
        # Assuming A, B are symmetric for QAP standard, but let's be general.
        # If A, B symmetric, G = 2 A X B
        # Let's check symmetry quickly or just use general formula.
        # General: d/dX Tr(A X B X^T) = A X B^T + A^T X B
        G = a @ X @ b.T + a.T @ X @ b
        
        # Gradient Step
        X_temp = X - lr * G
        
        # Project to Permutation using Hungarian
        # We want to maximize <P, X_temp>
        # linear_sum_assignment minimizes sum(cost[i, j])
        # So cost = -X_temp
        row_ind, col_ind = linear_sum_assignment(-X_temp)
        p_new = col_ind # p_new[i] = location assigned to facility i
        
        # Evaluate Cost
        cost_new = _qap_cost(a, b, p_new)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new
            # Reset X to the new best permutation to stay close to feasible solutions
            X = np.zeros((n, n), dtype=np.float64)
            X[np.arange(n), best_p] = 1.0
            # Optionally, we could keep X as a mix to allow exploration
            # X = 0.5 * X + 0.5 * P_new_matrix
        else:
            # If no improvement, update X to X_temp (softened) to continue exploration
            # Or dampen the update
            X = 0.5 * X + 0.5 * np.zeros((n, n)) # This decays to 0, bad.
            # Better: X = X_temp, but normalized?
            # Let's just keep X = X_temp and let the projection handle the "hard" constraint.
            # But X_temp might not be doubly stochastic.
            # Let's project X_temp to Doubly Stochastic (Sinkhorn) to keep X valid for next gradient step?
            # For simplicity, let's just set X = P_new (hard projection) even if no improvement?
            # That would make it a pure coordinate descent / local search.
            # To maintain "ADMM" flavor, we should keep X soft.
            # Let's do: X = X_temp
            # And periodically normalize X to be doubly stochastic?
            
            # Actually, if we just set X = P_new (the projected point), it's Projected Gradient Descent.
            # This is a valid and strong method.
            X = np.zeros((n, n), dtype=np.float64)
            X[np.arange(n), p_new] = 1.0
            
        # Decay learning rate
        lr *= lr_decay
        
    return best_p.tolist()
