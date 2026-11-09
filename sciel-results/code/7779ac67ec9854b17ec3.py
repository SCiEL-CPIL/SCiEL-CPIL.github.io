# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _entropy_from_values(values, weights):
    """Compute Shannon entropy in bits given values and their weights."""
    from collections import defaultdict
    d = defaultdict(float)
    for v, w in zip(values, weights):
        d[v] += w
    total = sum(d.values())
    if total <= 1e-15:
        return 0.0
    probs = [w / total for w in d.values() if w > 1e-15]
    if not probs:
        return 0.0
    return -sum(p * np.log2(p) for p in probs)


def _compute_ratio(P):
    """Compute D/M ratio for a 10x10 distribution P[y,x]."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    # D = H of (x - y)
    d_vals = x_flat - y_flat
    D = _entropy_from_values(d_vals, w)
    
    # M = max(H_0, H_1, H_2)
    h0_vals = x_flat
    h0 = _entropy_from_values(h0_vals, w)
    
    h1_vals = x_flat + y_flat
    h1 = _entropy_from_values(h1_vals, w)
    
    h2_vals = x_flat + 2 * y_flat
    h2 = _entropy_from_values(h2_vals, w)
    
    M = max(h0, h1, h2)
    
    if M <= 1e-9:
        return 0.0
    
    return D / M


def _optimize_P():
    """Optimize P to maximize D/M ratio."""
    deadline = time.time() + 580  # 580s to leave safety margin
    
    # Start from uniform
    P0 = np.ones((10, 10), dtype=np.float64) / 100.0
    best_P = P0.copy()
    best_score = _compute_ratio(best_P)
    
    # Try multiple random initializations
    rng = np.random.default_rng(42)
    
    for init_iter in range(5):
        if time.time() > deadline:
            break
        
        # Create a random initial distribution
        if init_iter == 0:
            P_init = P0.copy()
        else:
            # Random Dirichlet-like distribution
            P_init = rng.dirichlet(np.ones(100))
            P_init = P_init.reshape(10, 10)
        
        # Optimize using L-BFGS-B
        def objective(params):
            P = np.exp(params)  # Ensure positivity
            P = P / P.sum()     # Normalize
            return -_compute_ratio(P)  # Minimize negative ratio
        
        # Initialize with log of P (with small epsilon to avoid log(0))
        params0 = np.log(P_init + 1e-10)
        
        try:
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success or _compute_ratio(np.exp(result.x) / np.exp(result.x).sum()) > best_score:
                P_candidate = np.exp(result.x)
                P_candidate = P_candidate / P_candidate.sum()
                score = _compute_ratio(P_candidate)
                if score > best_score:
                    best_score = score
                    best_P = P_candidate
        except Exception:
            pass
        
        # Also try a few gradient-free local improvements
        for local_iter in range(20):
            if time.time() > deadline:
                break
            
            # Perturb the current best
            P_perturbed = best_P.copy()
            mask = rng.random((10, 10)) < 0.3
            P_perturbed[mask] *= (1 + 0.1 * (rng.random(np.sum(mask)) - 0.5))
            P_perturbed = np.clip(P_perturbed, 0, None)
            s = P_perturbed.sum()
            if s > 1e-15:
                P_perturbed = P_perturbed / s
            
            score = _compute_ratio(P_perturbed)
            if score > best_score:
                best_score = score
                best_P = P_perturbed
    
    return best_P


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    P = _optimize_P()
    P = np.clip(P, 0, None)
    s = P.sum()
    if s > 1e-15:
        P = P / s
    return P


# EVOLVE-BLOCK-END
