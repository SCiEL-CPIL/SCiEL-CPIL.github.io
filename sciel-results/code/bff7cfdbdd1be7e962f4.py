"""Spectral equiripple approach for minimizing max absolute autoconvolution / (sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize


def _autocorr_full(x):
    """Compute full autocorrelation via FFT (linear, not circular)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = fft(x, size)
    ac = ifft(X * np.conj(X)).real[:2 * n - 1]
    return ac


def _objective(x):
    """Compute max |autocorrelation| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if s == 0:
        return float('inf')
    ac = _autocorr_full(x)
    return np.max(np.abs(ac)) / (s * s)


def _equiripple_spectral_design(N, max_iter=50, tol=1e-6):
    """
    Design a sequence whose autoconvolution is as flat as possible (equiripple),
    by iteratively adjusting the spectral magnitude to equalize the autocorrelation peaks.
    Then round to +/-1 and evaluate.
    """
    rng = np.random.RandomState(123)
    
    # Start with a flat spectral magnitude (white noise)
    # We want to design x such that |autocorr(x)| is roughly constant across lags
    # This is equivalent to making the power spectrum |X(f)|^2 have specific properties
    
    # Initialize x as random signs
    x = rng.choice([-1.0, 1.0], size=N).astype(float)
    
    for it in range(max_iter):
        ac = _autocorr_full(x)
        ac_sidelobes = ac[1:]  # Exclude lag 0
        
        # We want to minimize the peak sidelobe
        # Use a simple iterative method: find the lag with max |ac|,
        # and adjust x to reduce that lag's contribution
        
        # Find the worst lag
        k_worst = np.argmax(np.abs(ac_sidelobes)) + 1
        ac_worst = ac_sidelobes[k_worst]
        
        # Gradient of ac_k wrt x_j:
        # ac_k = sum_{i=0}^{N-1-k} x_i x_{i+k}
        # d(ac_k)/dx_j = x_{j+k} (if j+k < N) + x_{j-k} (if j-k >= 0)
        
        grad = np.zeros(N)
        if k_worst < N:
            grad[0:N-k_worst] += x[k_worst:N]
            grad[k_worst:N] += x[0:N-k_worst]
        
        # Update x in the direction that reduces |ac_k_worst|
        # If ac_worst > 0, we want to decrease ac_k, so move against gradient
        # If ac_worst < 0, we want to increase ac_k, so move with gradient
        direction = -np.sign(ac_worst) * grad
        
        # Small step
        step = 0.1
        x_new = x + step * direction
        
        # Project to +/-1 via sign
        x_signed = np.sign(x_new)
        x_signed[x_signed == 0] = 1.0
        
        # Accept if it improves the objective
        obj_old = np.max(np.abs(ac)) / (np.sum(x) ** 2)
        obj_new = np.max(np.abs(_autocorr_full(x_signed))) / (np.sum(x_signed) ** 2)
        
        if obj_new < obj_old:
            x = x_signed
        else:
            # Try a smaller step or random restart
            if it % 10 == 0:
                x = rng.choice([-1.0, 1.0], size=N).astype(float)
    
    return x


def _randomized_spectral_search(N, n_trials=50, time_budget=60.0):
    """
    Perform multiple randomized spectral designs with different initializations
    and keep the best result.
    """
    start_time = time.time()
    best_score = float('inf')
    best_x = np.ones(N)
    
    rng = np.random.RandomState(42)
    
    for trial in range(n_trials):
        if time.time() > start_time + time_budget:
            break
        
        # Random initialization
        x = rng.choice([-1.0, 1.0], size=N).astype(float)
        
        # Local spectral optimization
        for it in range(100):
            if time.time() > start_time + time_budget:
                break
            
            ac = _autocorr_full(x)
            ac_sidelobes = ac[1:]
            
            # Find top few worst lags
            top_k = min(5, len(ac_sidelobes))
            worst_lags = np.argsort(np.abs(ac_sidelobes))[-top_k:][::-1]
            
            # For each worst lag, compute gradient and update
            grad_total = np.zeros(N)
            for k in worst_lags:
                k = int(k) + 1  # Shift back to original index
                if k < N:
                    grad_k = np.zeros(N)
                    grad_k[0:N-k] += x[k:N]
                    grad_k[k:N] += x[0:N-k]
                    grad_total += np.sign(ac[k]) * grad_k
            
            # Update direction: reduce the worst lags
            step = 0.05
            x_new = x - step * grad_total
            
            # Project to +/-1
            x_signed = np.sign(x_new)
            x_signed[x_signed == 0] = 1.0
            
            # Accept if improved
            obj_old = np.max(np.abs(ac)) / (np.sum(x) ** 2)
            obj_new = np.max(np.abs(_autocorr_full(x_signed))) / (np.sum(x_signed) ** 2)
            
            if obj_new < obj_old:
                x = x_signed
        
        sc = _objective(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    return best_x, best_score


def _local_search_flip(x, n_iter=5000, time_budget=30.0):
    """
    Perform local search by flipping individual bits to reduce the objective.
    """
    start_time = time.time()
    current = x.copy()
    current_score = _objective(current)
    rng = np.random.RandomState(7)
    
    for it in range(n_iter):
        if time.time() > start_time + time_budget:
            break
        
        # Pick a random index to flip
        idx = rng.randint(N)
        
        # Compute the change in autocorrelation efficiently
        # Flipping x[idx] changes ac_k by:
        # For k < idx: ac_k changes by -2 * x[idx] * x[idx+k] (if idx+k < N)
        # For k = idx: ac_k changes by -2 * x[idx]^2 = -2 (but x[idx]^2 = 1)
        # For k > idx: ac_k changes by -2 * x[idx] * x[idx-k] (if idx-k >= 0)
        
        # Simpler: just flip and recompute for small N
        old_val = current[idx]
        current[idx] = -old_val
        
        new_score = _objective(current)
        
        if new_score < current_score:
            current_score = new_score
        else:
            current[idx] = old_val
    
    return current, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_score = float('inf')
    best_x = np.ones(N)
    
    # Evaluate baseline
    sc = _objective(best_x)
    if sc < best_score:
        best_score = sc
    
    # Spectral search with time budget
    if time.time() < deadline - 10.0:
        x_spectral, sc_spectral = _randomized_spectral_search(
            N, n_trials=30, time_budget=40.0
        )
        if sc_spectral < best_score:
            best_score = sc_spectral
            best_x = x_spectral.copy()
    
    # Local search from best found
    if time.time() < deadline - 5.0:
        x_local, sc_local = _local_search_flip(
            best_x, n_iter=3000, time_budget=30.0
        )
        if sc_local < best_score:
            best_score = sc_local
            best_x = x_local.copy()
    
    # Additional local search from random starts
    rng = np.random.RandomState(99)
    for _ in range(3):
        if time.time() > deadline - 5.0:
            break
        
        x_random = rng.choice([-1.0, 1.0], size=N).astype(float)
        x_local, sc_local = _local_search_flip(
            x_random, n_iter=2000, time_budget=20.0
        )
        if sc_local < best_score:
            best_score = sc_local
            best_x = x_local.copy()
    
    return best_x
# EVOLVE-BLOCK-END
