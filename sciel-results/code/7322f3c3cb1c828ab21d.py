"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_max_autocorr(x):
    """Compute the maximum absolute autocorrelation (excluding lag 0)."""
    n = len(x)
    max_val = 0.0
    for lag in range(1, n):
        # Autocorrelation at this lag
        ac = np.dot(x[:n-lag], x[lag:])
        abs_ac = abs(ac)
        if abs_ac > max_val:
            max_val = abs_ac
    return max_val

def _try_improve(x, max_time=100.0):
    """Local search to minimize max absolute autocorrelation."""
    start_time = time.time()
    n = len(x)
    x = x.copy()
    current_score = _compute_max_autocorr(x)
    best_score = current_score
    best_x = x.copy()
    
    # Precompute autocorrelations for efficiency
    def _get_autocorrs(x):
        n = len(x)
        acs = np.zeros(n)
        for lag in range(1, n):
            acs[lag] = np.dot(x[:n-lag], x[lag:])
        return acs
    
    current_acs = _get_autocorrs(x)
    current_max = max(abs(current_acs[1:]))
    
    # Local search: try flipping each position
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > max_time:
                break
            # Try flipping position i
            old_val = x[i]
            x[i] = -old_val
            
            # Recompute affected autocorrelations
            # Only lags that include position i are affected
            new_acs = current_acs.copy()
            for lag in range(1, n):
                # Check if position i is in the range for this lag
                if i < n - lag:
                    # Position i is in the first part
                    new_acs[lag] += 2 * old_val * x[i + lag]
                if i >= lag and i + lag < n:
                    # Position i is in the second part (as x[i-lag] in the product)
                    # Actually, x[j]*x[j+lag], so if j=i, we have x[i]*x[i+lag]
                    # If j+lag=i, we have x[i-lag]*x[i]
                    pass  # Already handled above for j=i case
                    
            # More carefully: ac[lag] = sum_{j=0}^{n-lag-1} x[j]*x[j+lag]
            # Flipping x[i] affects terms where j=i (if i < n-lag) and j+lag=i (if i>=lag and i<n)
            # So terms: x[i]*x[i+lag] (if i < n-lag) and x[i-lag]*x[i] (if i >= lag)
            
            # Let's just recompute properly for affected lags
            affected_lags = []
            for lag in range(1, n):
                # x[i] appears in ac[lag] if:
                # 1. i < n-lag: as x[i] in x[i]*x[i+lag]
                # 2. i >= lag and i < n: as x[i] in x[i-lag]*x[i] where j=i-lag
                if i < n - lag:
                    affected_lags.append(lag)
                if i >= lag and i < n:
                    if lag not in affected_lags:
                        affected_lags.append(lag)
            
            for lag in affected_lags:
                # Recompute ac[lag] from scratch (expensive but correct)
                new_acs[lag] = np.dot(x[:n-lag], x[lag:])
            
            new_max = max(abs(new_acs[1:]))
            
            if new_max < current_max:
                current_max = new_max
                current_acs = new_acs
                improved = True
                if current_max < best_score:
                    best_score = current_max
                    best_x = x.copy()
            else:
                x[i] = old_val  # Revert
    
    return best_x, best_score

def construct():
    """Construct a signed step function to minimize the upper bound on C3."""
    start_time = time.time()
    max_time = 110.0  # Safety margin before 120s limit
    
    # Try multiple random initializations
    best_x = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Start with a few different initial patterns
    initializations = [
        np.ones(N),  # All ones
        np.random.choice([-1, 1], size=N),  # Random
        np.array([1 if i % 2 == 0 else -1 for i in range(N)]),  # Alternating
        np.array([1 if (i // 10) % 2 == 0 else -1 for i in range(N)]),  # Blocks of 10
    ]
    
    for init in initializations:
        if (time.time() - start_time) > max_time:
            break
        
        x, score = _try_improve(init.astype(float), max_time=max(1.0, max_time - (time.time() - start_time)))
        
        if score < best_score:
            best_score = score
            best_x = x
    
    # If we didn't find anything, fall back to all ones
    if best_x is None:
        best_x = np.ones(N)
    
    # Ensure we return a sequence of signed step heights
    return best_x
# EVOLVE-BLOCK-END
