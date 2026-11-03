"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _autocorr(a):
    """Compute full autocorrelation of sequence a."""
    n = len(a)
    ac = np.correlate(a, a, mode='full')
    # ac is length 2n-1, centered at n-1
    return ac

def _peak_sidelobe(a):
    """Peak absolute sidelobe of autocorrelation."""
    ac = _autocorr(a)
    # Zero lag is at index N-1
    # Sidelobes are all other indices
    mask = np.ones(len(ac), dtype=bool)
    mask[N-1] = False
    return np.max(np.abs(ac[mask]))

def _l2_sidelobe(a):
    """L2 norm of autocorrelation sidelobes."""
    ac = _autocorr(a)
    mask = np.ones(len(ac), dtype=bool)
    mask[N-1] = False
    return np.sqrt(np.sum(ac[mask]**2))

def _score_metric(a):
    """The raw value we want to minimize. Using peak sidelobe."""
    return _peak_sidelobe(a)

def _soft_peak_sidelobe(a, temperature=10.0):
    """Soft approximation of peak sidelobe using log-sum-exp."""
    ac = _autocorr(a)
    mask = np.ones(len(ac), dtype=bool)
    mask[N-1] = False
    sidelobes = np.abs(ac[mask])
    # Soft max
    max_val = np.max(sidelobes)
    if max_val == 0:
        return 0.0
    # scaled log-sum-exp for soft max
    scaled = sidelobes / temperature
    shifted = scaled - np.max(scaled)
    logsumexp = np.log(np.sum(np.exp(shifted))) + np.max(scaled)
    return logsumexp * temperature

def _soft_l2_sidelobe(a, temperature=5.0):
    """Soft L2 norm using differentiable approximation."""
    ac = _autocorr(a)
    mask = np.ones(len(ac), dtype=bool)
    mask[N-1] = False
    sidelobes = ac[mask]**2
    # Soft L2: use power mean with high power to approximate L2
    # Actually just use L2 directly, it's already differentiable
    return np.sqrt(np.sum(sidelobes))

def _local_search(a, max_iters=50000, time_limit=30.0):
    """Local search by single bit flips."""
    a = a.copy().astype(np.float64)
    best_score = _score_metric(a)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random position
        idx = np.random.randint(N)
        a[idx] = -a[idx]
        new_score = _score_metric(a)
        if new_score < best_score:
            best_score = new_score
        else:
            a[idx] = -a[idx]
    
    return a, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
    # Step 1: Initialize with random ±1
    a = np.random.choice([-1.0, 1.0], size=N)
    best_a = a.copy()
    best_score = _score_metric(a)
    
    # Step 2: Continuous relaxation to get better starting point
    # Minimize L2 norm of sidelobes in continuous space
    def objective(x):
        if time.time() > deadline:
            return 1e10
        return _soft_l2_sidelobe(x, temperature=1.0)
    
    def gradient(x):
        if time.time() > deadline:
            return np.zeros(N)
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros(N)
        base = objective(x)
        for i in range(N):
            x[i] += eps
            f_plus = objective(x)
            x[i] -= 2*eps
            f_minus = objective(x)
            x[i] += eps
            grad[i] = (f_plus - f_minus) / (2*eps)
        return grad
    
    # Use L-BFGS-B for continuous optimization
    x0 = a.copy()
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 50000},
            bounds=[(-1.0, 1.0)] * N
        )
        if result.x is not None:
            x_cont = result.x
            # Round to ±1
            a_rounded = np.where(x_cont >= 0, 1.0, -1.0)
            score_rounded = _score_metric(a_rounded)
            if score_rounded < best_score:
                best_score = score_rounded
                best_a = a_rounded.copy()
    except Exception:
        pass
    
    # Step 3: Local search from best found
    a, best_score = _local_search(best_a, max_iters=100000, time_limit=30.0)
    
    # Step 4: Multiple restarts if time permits
    restarts = 0
    while time.time() < deadline - 5.0 and restarts < 5:
        # New random start
        a_new = np.random.choice([-1.0, 1.0], size=N)
        
        # Quick continuous relaxation
        try:
            result = minimize(
                objective,
                a_new.copy(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 10000},
                bounds=[(-1.0, 1.0)] * N
            )
            if result.x is not None:
                a_rounded = np.where(result.x >= 0, 1.0, -1.0)
                score_rounded = _score_metric(a_rounded)
                if score_rounded < best_score:
                    best_score = score_rounded
                    best_a = a_rounded.copy()
        except Exception:
            pass
        
        # Local search
        a_ls, score_ls = _local_search(best_a, max_iters=50000, time_limit=15.0)
        if score_ls < best_score:
            best_score = score_ls
            best_a = a_ls.copy()
        
        restarts += 1
    
    # Return as integer array of ±1
    return best_a.astype(np.int8)

# EVOLVE-BLOCK-END
