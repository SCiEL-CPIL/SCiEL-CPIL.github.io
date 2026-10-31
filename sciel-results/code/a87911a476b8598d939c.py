"""Improved program using SLSQP with diverse initializations and local search polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / (2 * N)

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)."""
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    max_val = max_conv * DELTA
    if max_val == 0:
        return 0.0
    return l2_sq / (l1 * max_val)

def _is_sidon_set(indices):
    """Check if indices form a Sidon set (all pairwise differences are unique)."""
    diffs = set()
    n = len(indices)
    for i in range(n):
        for j in range(i):
            d = abs(indices[i] - indices[j])
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _generate_sidon_sets(n, k, num_samples=50):
    """Generate multiple Sidon sets of size k in range(n)."""
    sidon_sets = []
    
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        new_ruler = ruler + [i]
        if _is_sidon_set(new_ruler):
            ruler = new_ruler
            if len(ruler) >= k:
                break
    if len(ruler) >= k:
        sidon_sets.append(ruler[:k])
    
    # Random sampling with rejection
    for _ in range(num_samples):
        attempt = 0
        while attempt < 100:
            candidate = sorted(random.sample(range(n), k))
            if _is_sidon_set(candidate):
                sidon_sets.append(candidate)
                break
            attempt += 1
    
    return sidon_sets

def _local_search_polish(h, support, iterations=1000):
    """Local search to polish weights on support indices."""
    K = len(support)
    w = np.zeros(K)
    for i, idx in enumerate(support):
        w[i] = h[idx]
    
    best_w = w.copy()
    best_ratio = _compute_ratio(h)
    
    for _ in range(iterations):
        # Random coordinate update
        i = random.randint(0, K - 1)
        j = random.randint(0, K - 1)
        if i == j:
            continue
        
        # Try moving epsilon from j to i
        eps = w[j] * 0.1
        if eps <= 0:
            continue
        
        w_new = w.copy()
        w_new[i] += eps
        w_new[j] -= eps
        
        if np.all(w_new >= 0):
            h_new = np.zeros(N)
            for k, idx in enumerate(support):
                h_new[idx] = w_new[k]
            ratio = _compute_ratio(h_new)
            if ratio > best_ratio:
                best_ratio = ratio
                best_w = w_new.copy()
    
    h_opt = np.zeros(N)
    for i, idx in enumerate(support):
        h_opt[idx] = best_w[i]
    return h_opt, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_h = None
    best_ratio = 0.0
    
    # Try different Sidon set sizes
    for target_size in [8, 7, 6, 5]:
        if time.time() > deadline:
            break
        
        # Generate diverse Sidon sets
        sidon_sets = _generate_sidon_sets(N, target_size, num_samples=30)
        
        for support in sidon_sets:
            if time.time() > deadline:
                break
            
            K = len(support)
            if K < 3:
                continue
            
            # Try multiple initializations
            initializations = []
            
            # Uniform
            w_uniform = np.ones(K) / K
            initializations.append(w_uniform)
            
            # Sparse (one dominant)
            for dominant in range(min(3, K)):
                w_sparse = np.zeros(K)
                w_sparse[dominant] = 0.8
                remaining = 0.2 / (K - 1)
                for i in range(K):
                    if i != dominant:
                        w_sparse[i] = remaining
                initializations.append(w_sparse)
            
            # Random
            for _ in range(3):
                w_rand = np.random.dirichlet(np.ones(K))
                initializations.append(w_rand)
            
            for w_init in initializations:
                if time.time() > deadline:
                    break
                
                h_init = np.zeros(N)
                for i, idx in enumerate(support):
                    h_init[idx] = w_init[i]
                
                # Objective function
                def objective(w):
                    h = np.zeros(N)
                    for i, idx in enumerate(support):
                        h[idx] = w[i]
                    return -_compute_ratio(h)
                
                # Constraints: sum(w) = 1, w >= 0
                constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
                bounds = [(0, 1) for _ in range(K)]
                
                # SLSQP optimization
                try:
                    res = minimize(objective, w_init, method='SLSQP', bounds=bounds,
                                  constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
                    
                    if res.success:
                        w_opt = res.x
                        h_opt = np.zeros(N)
                        for i, idx in enumerate(support):
                            h_opt[idx] = w_opt[i]
                        
                        ratio = _compute_ratio(h_opt)
                        
                        # Local search polish
                        h_polished, ratio_polished = _local_search_polish(h_opt, support, iterations=200)
                        
                        if ratio_polished > best_ratio:
                            best_ratio = ratio_polished
                            best_h = h_polished.copy()
                except Exception:
                    pass
    
    if best_h is None:
        # Fallback to single spike
        best_h = np.zeros(N)
        best_h[0] = 1.0
        
    return [float(x) for x in best_h]
