"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _sidon_set(k):
    """Generate a Sidon set (B2 sequence) of size k using greedy construction."""
    used = set()
    seq = []
    candidate = 0
    while len(seq) < k:
        s = set(2 * seq + [candidate])
        s.add(2 * candidate)
        if s.isdisjoint(used):
            seq.append(candidate)
            used |= s
            candidate += 1
        else:
            candidate += 1
    return np.array(seq, dtype=float)

def _compute_ratio(weights, positions):
    """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step function."""
    n = len(weights)
    # L2 norm squared: sum of w_i^2 * (interval length)^2, but since we normalize by ratio, 
    # and all intervals have same length, the length factors cancel.
    l2_sq = np.sum(weights ** 2)
    l1 = np.sum(weights)
    
    # Autoconvolution: for step functions on equal intervals, the autoconvolution
    # at a grid point is sum of w_i * w_j where pos_i + pos_j = target
    # For discrete approximation, compute all pairwise sums
    # positions are indices of steps, autoconvolution peak = max over all sums
    # For a Sidon set, each sum pos_i + pos_j is unique (except i=j vs j=i),
    # so the peak is max over i,j of w_i * w_j for pairs with same sum.
    
    # Compute all pairwise sums and their products
    sums = {}
    for i in range(n):
        for j in range(n):
            s = positions[i] + positions[j]
            if s not in sums:
                sums[s] = 0.0
            sums[s] += weights[i] * weights[j]
    
    l1_inf = max(sums.values()) if sums else 1e-15
    
    if l1_inf < 1e-15 or l1 < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * l1_inf)

def _objective(params, positions):
    """Negative ratio for minimization."""
    # params: log-weights to ensure positivity
    log_w = params
    weights = np.exp(log_w)
    # Normalize so that max weight = 1 to fix scale invariance
    max_w = np.max(weights)
    if max_w < 1e-15:
        return 0.0
    weights = weights / max_w
    return -_compute_ratio(weights, positions)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate a Sidon set of size N
    positions = _sidon_set(N)
    
    best_ratio = 0.0
    best_weights = np.ones(N)
    
    # Try different exponential profiles
    # Parameterize weights as w_i = exp(a * i + b), optimize a and b
    # Since scale-invariant, fix max weight = 1, so w_i = exp(a * (N-1-i)) for a in [0, large]
    
    # Grid search over the skew parameter first
    a_values = np.linspace(0.0, 5.0, 100)
    for a in a_values:
        if time.time() > deadline - 5.0:
            break
        weights = np.exp(-a * np.arange(N))
        max_w = np.max(weights)
        weights = weights / max_w
        ratio = _compute_ratio(weights, positions)
        if ratio > best_ratio:
            best_ratio = ratio
            best_weights = weights.copy()
    
    # Local optimization around the best a found
    if time.time() < deadline - 10.0:
        # Find best a from grid
        a_best = None
        best_ratio_local = 0.0
        for a in a_values:
            weights = np.exp(-a * np.arange(N))
            max_w = np.max(weights)
            if max_w > 0:
                weights = weights / max_w
                ratio = _compute_ratio(weights, positions)
                if ratio > best_ratio_local:
                    best_ratio_local = ratio
                    a_best = a
        
        if a_best is not None:
            # Optimize the full weight vector using log-parameters
            # Initialize with the best exponential profile
            log_w_init = -a_best * np.arange(N)
            
            def obj(params):
                return _objective(params, positions)
            
            result = minimize(obj, log_w_init, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10})
            
            if time.time() < deadline - 2.0:
                log_w_opt = result.x
                weights_opt = np.exp(log_w_opt)
                max_w = np.max(weights_opt)
                if max_w > 1e-15:
                    weights_opt = weights_opt / max_w
                    ratio_opt = _compute_ratio(weights_opt, positions)
                    if ratio_opt > best_ratio:
                        best_ratio = ratio_opt
                        best_weights = weights_opt.copy()
    
    # Return the step heights (nonnegative)
    # The evaluator will use these as the step function heights
    return list(best_weights)
# EVOLVE-BLOCK-END

