# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _sieve_mobius(K):
    """Compute Mobius function mu[1..K] using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu, primes

def _compute_M(mu, K, max_x):
    """Compute M(x) = sum_k mu[k] * floor(x/k) for x in [1, max_x]."""
    A = np.zeros(max_x + 1, dtype=np.float64)
    for d in range(1, K + 1):
        if mu[d] != 0:
            A[d::d] += mu[d]
    return np.cumsum(A)

def _adjust_and_compute(mu, K):
    """Apply the f(1) adjustment and compute the objective a.
    
    Returns (a_value, M_array, adjusted_values)
    """
    # S = sum_k mu(k)/k
    S = np.sum(mu[1:K+1].astype(np.float64) / np.arange(1, K+1, dtype=np.float64))
    
    # Adjusted values: f(1) = mu(1) - S = 1 - S, f(k) = mu(k) for k > 1
    f = np.empty(K + 1, dtype=np.float64)
    f[0] = 0.0
    f[1] = 1.0 - S
    f[2:K+1] = mu[2:K+1].astype(np.float64)
    
    # Compute M(x) with adjusted f
    max_x = 10 * K
    A = np.zeros(max_x + 1, dtype=np.float64)
    for d in range(1, K + 1):
        if f[d] != 0:
            A[d::d] += f[d]
    M = np.cumsum(A)
    
    # Objective: a = -sum_k f(k) * log(k) / k
    if K >= 2:
        ks = np.arange(1, K + 1, dtype=np.float64)
        a = -np.sum(f[1:K+1] * np.log(ks) / ks)
    else:
        a = 0.0
    
    return a, M, f

def _try_lp_refinement(mu, K, time_budget=10.0):
    """Attempt LP-based refinement to reduce M(x) peaks while preserving mu structure.
    
    The idea: keep mu(k) for k > 1 fixed, and adjust f(1) optimally.
    Since f(1) is already set to 1 - S (making sum f(k)/k = 0), we check if
    the constraint M(x) <= 1.0001 is satisfied. If not, we can't improve via
    f(1) alone (it's already determined by the constraint sum f(k)/k = 0).
    
    A more aggressive approach: allow small perturbations to some f(k) values
    to reduce the maximum of M(x). This is a linear program.
    """
    start = time.time()
    
    # First check current feasibility
    a_current, M_current, f_current = _adjust_and_compute(mu, K)
    if np.max(M_current) <= 1.0001:
        return a_current, f_current, True
    
    # The current mu-based solution is infeasible. Try LP to find a feasible solution
    # that's close to mu.
    
    # Simplified LP: minimize deviation from mu subject to M(x) <= 1.0001 and sum f(k)/k = 0
    # Variables: f(1), f(2), ..., f(K)
    # Constraints:
    #   1. sum_k f(k)/k = 0
    #   2. M(x) <= 1.0001 for all x in [1, 10K]
    #   3. -10 <= f(k) <= 10 (clipping bounds)
    # Objective: maximize -sum f(k) log(k)/k  (or minimize sum f(k) log(k)/k)
    
    # This LP is too large (10K constraints, K variables) for most K values.
    # Instead, try a heuristic: reduce f(k) for k where M(x) peaks.
    
    # Alternative: use only the top few M(x) peaks to guide adjustment
    # This is a heuristic, not guaranteed optimal, but fast.
    
    if time.time() - start > time_budget:
        return a_current, f_current, False
    
    # Heuristic: identify x where M(x) is maximal, and reduce contributions to those x
    # M(x) = sum_k f(k) * floor(x/k)
    # For a given x, the largest contributors are small k.
    
    # Try reducing f(1) further? No, f(1) is fixed by the constraint sum f(k)/k = 0.
    # But we can adjust other f(k) and recompute f(1).
    
    # Let's try: for the top T peaks, reduce f(k) for the k that contribute most to those peaks.
    # This is a greedy heuristic.
    
    # Actually, a simpler approach: scale down all mu(k) for k > 1 by a factor,
    # and recompute f(1) to satisfy sum f(k)/k = 0.
    
    best_a = -np.inf
    best_f = f_current.copy()
    best_M = M_current
    
    # Try different scaling factors
    for scale in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        if time.time() - start > time_budget:
            break
        
        f_scaled = np.empty(K + 1, dtype=np.float64)
        f_scaled[0] = 0.0
        f_scaled[1] = 0.0  # Will be set below
        f_scaled[2:K+1] = mu[2:K+1].astype(np.float64) * scale
        
        # Compute S = sum_k f(k)/k for k >= 2
        S_rest = np.sum(f_scaled[2:K+1] / np.arange(2, K+1, dtype=np.float64))
        # Need sum_k f(k)/k = 0, so f(1)/1 + S_rest = 0 => f(1) = -S_rest
        f_scaled[1] = -S_rest
        
        # Clip to [-10, 10]
        f_scaled = np.clip(f_scaled, -10, 10)
        
        # Recompute M
        max_x = 10 * K
        A = np.zeros(max_x + 1, dtype=np.float64)
        for d in range(1, K + 1):
            if f_scaled[d] != 0:
                A[d::d] += f_scaled[d]
        M_scaled = np.cumsum(A)
        
        if np.max(M_scaled) <= 1.0001:
            ks = np.arange(1, K + 1, dtype=np.float64)
            a_scaled = -np.sum(f_scaled[1:K+1] * np.log(ks) / ks)
            if a_scaled > best_a:
                best_a = a_scaled
                best_f = f_scaled.copy()
                best_M = M_scaled
                if scale == 1.0:
                    break
    
    if best_a > -np.inf:
        return best_a, best_f, True
    
    return a_current, f_current, False

def prime_number_theorem_partial_function():
    """
    Construct a partial function f to maximize the objective.
    
    Strategy:
    1. Use the Mobius function mu(k) as the base solution.
    2. Apply the f(1) adjustment to satisfy sum f(k)/k = 0.
    3. Check feasibility (M(x) <= 1.0001 for all x in [1, 10K]).
    4. If infeasible, try scaling down mu(k) for k > 1 to reduce M(x) peaks.
    5. Use the largest feasible K that fits within the time budget.
    """
    start_time = time.time()
    deadline = 590.0
    
    # Try large K first, fall back to smaller if needed
    for K in [200000, 150000, 100000, 50000, 20000, 10000, 5000, 1000, 500]:
        if time.time() > deadline - 30:
            break
        
        # Compute Mobius function
        mu, primes = _sieve_mobius(K)
        
        # Compute M(x) for the adjusted mu
        a_val, M_val, f_val = _adjust_and_compute(mu, K)
        
        if np.max(M_val) <= 1.0001:
            # Feasible! Use this K
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = f_val[1:K + 1].copy()
            return domain, values
        
        # Not feasible with raw mu. Try scaling heuristic.
        time_left = deadline - 30 - (time.time() - start_time)
        if time_left > 5:
            a_scaled, f_scaled, feasible = _try_lp_refinement(mu, K, time_budget=min(time_left, 15.0))
            if feasible:
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = f_scaled[1:K + 1].copy()
                return domain, values
    
    # Fallback: use the known good small solution
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
