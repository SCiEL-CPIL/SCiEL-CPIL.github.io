import numpy as np
import time

# Precompute the value arrays for the four slopes
_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# For each slope, precompute the bin index for each cell
# D: x - y, range -9..9 -> shift by 9 to get 0..18
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
# H0: x, range 0..9
_h0_vals = _x_flat.astype(np.int64)
# H1: x + y, range 0..18
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
# H2: x + 2y, range 0..27
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

# Number of bins for each
_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array."""
    probs = counts[counts > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _compute_ratio(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    w = P.flatten()
    
    # Compute pooled probabilities for each slope
    d_probs = np.bincount(_d_vals, weights=w, minlength=_N_D)
    h0_probs = np.bincount(_h0_vals, weights=w, minlength=_N_H0)
    h1_probs = np.bincount(_h1_vals, weights=w, minlength=_N_H1)
    h2_probs = np.bincount(_h2_vals, weights=w, minlength=_N_H2)
    
    D = _entropy_from_counts(d_probs)
    H0 = _entropy_from_counts(h0_probs)
    H1 = _entropy_from_counts(h1_probs)
    H2 = _entropy_from_counts(h2_probs)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = _N
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    start_time = time.time()
    timeout = 630
    safety_margin = 15
    
    best_score = _compute_ratio(P)
    best_P = P.copy()
    
    np.random.seed(42)
    
    # Try several initialization strategies
    def _normalize(P):
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-15:
            return None
        return P / s
    
    # Strategy 1: Uniform (already computed)
    # Strategy 2: Diagonal-heavy (maximize D by spreading x-y values)
    # Strategy 3: Random Dirichlet-like
    # Strategy 4: Concentrate on cells that give unique x-y values
    
    # For D = H(x-y), we want to spread mass across different (x-y) values
    # For M = max(H0,H1,H2), we want to minimize the max entropy
    # H0 = H(x): spread x values
    # H1 = H(x+y): spread x+y values  
    # H2 = H(x+2y): spread x+2y values
    
    # Key insight: To maximize D/M, we want D high and M low.
    # D is high when mass is spread across many x-y values.
    # M is low when one of H0, H1, H2 is low.
    # H0 is low if mass is concentrated on few x values.
    # So: spread across x-y values but concentrate on few x values.
    # This means: for each x, spread across y values that give different x-y.
    # But x-y = x-y, so for fixed x, different y give different x-y.
    # So for fixed x, all 10 y values give 10 different x-y values.
    # If we put all mass on one x, D can be at most log2(10) bits.
    # But H0 would be 0, making M potentially small.
    
    # Let's try: put all mass on x=0, spread evenly across y
    # Then D = H(-y) = H(y) = log2(10) ≈ 3.32
    # H0 = H(0) = 0
    # H1 = H(y) = log2(10)
    # H2 = H(2y) = log2(10)
    # M = log2(10), D/M = 1.0
    
    # Better: put mass on multiple x values but in a way that minimizes H0,H1,H2
    # while maximizing D.
    
    # Let's use a structured approach: try to find P that maximizes D/M
    # using the precomputed arrays
    
    def _score_fast(P_flat):
        """Fast score computation assuming P_flat is already normalized and non-negative."""
        d_probs = np.bincount(_d_vals, weights=P_flat, minlength=_N_D)
        h0_probs = np.bincount(_h0_vals, weights=P_flat, minlength=_N_H0)
        h1_probs = np.bincount(_h1_vals, weights=P_flat, minlength=_N_H1)
        h2_probs = np.bincount(_h2_vals, weights=P_flat, minlength=_N_H2)
        
        D = _entropy_from_counts(d_probs)
        H0 = _entropy_from_counts(h0_probs)
        H1 = _entropy_from_counts(h1_probs)
        H2 = _entropy_from_counts(h2_probs)
        
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return 0.0
            
        return D / M
    
    # Multi-start with different initializations
    num_restarts = 30
    time_per_restart = (timeout - safety_margin) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Initialize current P
        if r == 0:
            current_P = P.copy()
        elif r == 1:
            # Concentrate on x=0
            current_P = np.zeros((n, n))
            current_P[0, :] = 1.0 / n
        elif r == 2:
            # Concentrate on a few x values
            current_P = np.zeros((n, n))
            current_P[0, :] = 0.5 / n
            current_P[9, :] = 0.5 / n
        elif r == 3:
            # Diagonal pattern
            current_P = np.zeros((n, n))
            np.fill_diagonal(current_P, 1.0 / n)
        elif r == 4:
            # Anti-diagonal
            current_P = np.zeros((n, n))
            for i in range(n):
                current_P[i, n-1-i] = 1.0 / n
        else:
            # Perturb best found
            current_P = best_P + np.random.normal(0, 0.1, (n, n))
            current_P = _normalize(current_P)
            if current_P is None:
                current_P = P.copy()
                
        current_score = _compute_ratio(current_P)
        
        # Hill climbing with adaptive step size
        step_size = 0.3
        local_time_limit = time_per_restart
        local_start = time.time()
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() - local_start < local_time_limit:
            # Try a few random moves per iteration
            for _ in range(3):
                if time.time() - local_start >= local_time_limit:
                    break
                    
                i, j = np.random.randint(0, n, 2)
                delta = np.random.uniform(-step_size, step_size)
                
                new_P = current_P.copy()
                new_P[i, j] += delta
                
                if new_P[i, j] < -1e-9:
                    continue
                    
                new_P = np.clip(new_P, 0, None)
                s = new_P.sum()
                if s <= 1e-9:
                    continue
                new_P = new_P / s
                
                new_score = _score_fast(new_P.flatten())
                
                if new_score > current_score:
                    current_P = new_P
                    current_score = new_score
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_P = current_P.copy()
                    step_size *= 0.98
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Restart local search from a perturbation of best
                        current_P = best_P + np.random.normal(0, 0.15, (n, n))
                        current_P = _normalize(current_P)
                        if current_P is None:
                            current_P = P.copy()
                        current_score = _compute_ratio(current_P)
                        no_improve_count = 0
                        step_size = 0.3
                        break
                        
    return best_P
