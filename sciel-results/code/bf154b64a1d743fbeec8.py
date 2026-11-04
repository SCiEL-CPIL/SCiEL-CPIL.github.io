"""
QAP Solver: Gravitational Search Algorithm (GSA) on Soft-Assignment Matrix.

Strategy:
1. Represent the assignment as a soft matrix X (n x n) where X[i, j] is the
   "probability" or "attraction" of facility i being at location j.
2. Initialize X using spectral alignment (matching eigenvectors of A and B).
3. Iterate using GSA physics:
   - Compute gravity force F on each element X[i, j] based on "mass" of other
     elements. Mass is proportional to the flow a[i, k] and distance b[j, l].
   - The force pulls X[i, j] towards configurations that minimize cost.
   - Update X with velocity and inertia (simulated annealing-like cooling).
4. Periodically harden X into a permutation using the Hungarian algorithm or
   greedy rounding, and evaluate the exact cost.
5. Keep the best permutation found.
"""

import numpy as np
import time
from scipy.optimize import linear_sum_assignment
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p (p[i] is location of facility i)."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_init(a, b):
    """
    Initialize soft assignment matrix X using spectral decomposition.
    Align principal eigenvectors of A and B.
    """
    n = a.shape[0]
    # Symmetrize A for stability, though QAP A is not necessarily symmetric.
    # Standard QAP uses A as is. For spectral methods, we often use A_sym = (A + A.T)/2
    # or just A. Let's use A_sym for the "flow" spectrum.
    a_sym = (a + a.T) / 2.0
    
    # Compute eigenvectors
    # We want to align the "high flow" directions with "low distance" directions?
    # Actually, cost is sum a[i,j] b[p[i], p[j]].
    # If we map high flow to low distance, cost decreases.
    
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b)
    except np.linalg.LinAlgError:
        return np.ones((n, n)) / n

    # Sort eigenvectors by eigenvalue
    idx_a = np.argsort(eigvals_a)[::-1]
    idx_b = np.argsort(eigvals_b) # Ascending for B? Or descending?
    
    # Heuristic: High flow (large eigval A) should map to Low distance (small eigval B)?
    # Let's try aligning top A eigenvectors with bottom B eigenvectors.
    # Or simply: X = V_a @ V_b.T
    # Let's try standard alignment: sort both by descending magnitude and align.
    
    # A common approach: X = V_a * V_b (element-wise) or V_a @ V_b.T
    # Let's use V_a @ V_b.T where columns are eigenvectors.
    # We want to maximize trace(X^T B X) or similar? No, we want to minimize.
    
    # Let's just initialize with a random orthogonal matrix or identity-based
    # and let GSA do the work. Spectral init is tricky to get right for QAP.
    # A safer "soft" init: Rank matching.
    
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by flow sum (desc)
    fac_order = np.argsort(-a_row_sums)
    # Sort locations by distance sum (asc) - central locations
    loc_order = np.argsort(b_row_sums)
    
    X = np.zeros((n, n))
    for k in range(n):
        X[fac_order[k], loc_order[k]] = 1.0
        
    # Add noise to break symmetry and allow GSA to move
    X += np.random.randn(n, n) * 0.1
    # Normalize rows to sum to 1 (probability distribution)
    X = np.maximum(X, 0)
    X /= X.sum(axis=1, keepdims=True)
    
    return X

def _harden(X):
    """Convert soft matrix X to a hard permutation p using Hungarian."""
    # Hungarian minimizes cost. We want to assign i to j to minimize sum X? No.
    # We want to find the permutation that best matches the "mass" in X.
    # This is equivalent to finding the permutation that maximizes sum X[i, p[i]]?
    # Or minimizing -sum X[i, p[i]]?
    # Let's use linear_sum_assignment on -X to maximize the sum of selected entries.
    cost_matrix = -X
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(len(row_ind), dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _gsa_step(X, a, b, G, alpha, beta, v, inertia):
    """
    Perform one step of Gravitational Search Algorithm.
    
    X: n x n soft assignment matrix.
    a, b: QAP matrices.
    G: Gravitational constant.
    alpha: Cooling rate (for noise/damping).
    beta: Damping factor.
    v: Velocity matrix.
    inertia: Inertia weight.
    """
    n = X.shape[0]
    
    # Compute "Mass" for each facility-location pair?
    # In GSA, mass is usually a property of the agent. Here, the "agents" are the assignments.
    # Let's define the "force" on X[i, j] based on the gradient of the cost function
    # approximated by the soft assignment.
    
    # Cost(X) approx = sum_{i,j,k,l} a[i,k] b[j,l] X[i,j] X[k,l]
    # This is a quadratic form in the vectorized X.
    # Let x = vec(X). C = x^T M x where M is a large Kronecker sum product.
    # Gradient grad_x C = 2 M x.
    
    # Instead of building the huge M, we can compute the gradient efficiently.
    # Let A = a, B = b.
    # Cost = Tr(X^T A X B^T) ? No.
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]].
    # With soft X: Cost_soft = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    
    # Let's compute the gradient of Cost_soft w.r.t X.
    # d Cost / d X[i, k] = sum_{j, l} a[i, j] b[k, l] X[j, l] + sum_{j, l} a[j, i] b[l, k] X[j, l] (if symmetric)
    # Assuming A is not symmetric, we must be careful.
    # Cost = sum_{i,j} a[i,j] (X^T B X)_{ij} = Tr(X^T B X A^T) ?
    # Let Y = X^T B X. Then Cost = sum_{i,j} a[i,j] Y_{ij} = Tr(A Y) = Tr(A X^T B X).
    # d Tr(A X^T B X) / d X = 2 B X A^T ? No.
    # Let's use matrix calculus.
    # f(X) = Tr(A X^T B X)
    # df = Tr(A dX^T B X) + Tr(A X^T B dX)
    # df = Tr(B X A dX^T) + Tr(A X^T B dX)  (cyclic property)
    # df = Tr((B X A)^T dX) + Tr((A X^T B) dX)
    # df = Tr((A^T X^T B^T + A X^T B) dX)
    # If B is symmetric (distance), B^T = B.
    # Grad = A^T X^T B + A X^T B = (A + A^T) X^T B ? No.
    # Let's check:
    # Term 1: Tr(A dX^T B X) = Tr((B X A)^T dX) -> Grad part 1: B X A^T
    # Term 2: Tr(A X^T B dX) -> Grad part 2: A X^T B
    # Total Grad = B X A^T + A X^T B
    
    # Since B is symmetric:
    grad = b @ X @ a.T + a @ X.T @ b
    
    # GSA Update:
    # v = inertia * v + G * grad + noise
    # X = X + v
    # Normalize X to be a valid soft assignment (rows sum to 1, non-negative)
    
    noise = np.random.randn(*X.shape) * alpha
    v = inertia * v + G * grad + noise
    X = X + v
    
    # Project back to probability simplex (approx)
    X = np.maximum(X, 0)
    # Normalize rows
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = X / row_sums
    
    return X, v

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    best_p = None
    best_cost = np.inf
    
    # Parameters
    G = 1.0  # Initial gravity strength
    alpha = 0.5  # Noise level
    inertia = 0.9
    beta_decay = 0.99
    
    # Initialize
    X = _spectral_init(a, b)
    v = np.zeros_like(X)
    
    # Precompute symmetric A for potential use, though formula handles it
    # a_sym = (a + a.T) / 2.0
    
    iter_count = 0
    last_hardening = 0
    hardening_freq = 10  # Harden every 10 iterations
    
    while time.time() - start_time < time_limit - 5:
        # Adaptive cooling
        G *= beta_decay
        alpha *= beta_decay
        
        # GSA Step
        X, v = _gsa_step(X, a, b, G, alpha, beta_decay, v, inertia)
        
        iter_count += 1
        
        # Hardening and Evaluation
        if iter_count % hardening_freq == 0:
            p = _harden(X)
            cost = _qap_cost(a, b, p)
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
                # If we found a new best, we might want to "reheat" slightly
                # or just continue. Let's continue.
                
        # Check time
        if time.time() - start_time > time_limit - 10:
            break
            
    # Final hardening
    if best_p is None:
        p = _harden(X)
        best_p = p
        best_cost = _qap_cost(a, b, p)
        
    # One last local search polish on the best found permutation?
    # If time permits, do a quick swap search on best_p
    if time.time() - start_time < time_limit - 2:
        p = best_p
        current_cost = best_cost
        a_sym = a + a.T
        improved = True
        while improved and time.time() - start_time < time_limit - 1:
            improved = False
            # Sample random swaps
            for _ in range(1000):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j: continue
                # Swap i and j in p
                p[i], p[j] = p[j], p[i]
                new_cost = _qap_cost(a, b, p)
                if new_cost < current_cost:
                    current_cost = new_cost
                    best_cost = new_cost
                    best_p = p.copy()
                    improved = True
                else:
                    # Revert
                    p[i], p[j] = p[j], p[i]
                    
    return best_p.tolist()
