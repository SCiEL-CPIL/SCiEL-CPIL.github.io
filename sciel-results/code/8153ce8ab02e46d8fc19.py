# EVOLVE-BLOCK-START
import numpy as np
import random

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 probability distribution P[y, x] that maximizes the ratio
    H(X-Y) / max(H(X), H(X+Y), H(X+2Y)).
    
    This implementation uses Simulated Annealing to optimize the distribution.
    It operates on log-weights to ensure positivity and uses multiple restarts.
    """
    n = 10
    
    def compute_entropy(probs):
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def score_distribution(P):
        # P is 10x10
        # X: sum over y
        p_x = P.sum(axis=0)
        
        # X+Y
        p_xy = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                p_xy[x + y] += P[y, x]
                
        # X+2Y
        p_x2y = np.zeros(3*n - 1)
        for y in range(n):
            for x in range(n):
                p_x2y[x + 2*y] += P[y, x]
                
        # X-Y
        p_xm_y = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                p_xm_y[x - y + (n-1)] += P[y, x]

        H_x = compute_entropy(p_x)
        H_xy = compute_entropy(p_xy)
        H_x2y = compute_entropy(p_x2y)
        H_xm_y = compute_entropy(p_xm_y)
        
        M = max(H_x, H_xy, H_x2y)
        if M < 1e-9:
            return 0.0
        return H_xm_y / M

    def log_weights_to_probs(log_w):
        # Stable softmax
        log_w_max = np.max(log_w)
        w = np.exp(log_w - log_w_max)
        total = w.sum()
        if total == 0:
            return np.ones(n*n) / (n*n)
        return w / total

    def evaluate_log_weights(log_w):
        P = log_weights_to_probs(log_w)
        return score_distribution(P)

    best_score = -np.inf
    best_log_w = None
    
    # Simulated Annealing parameters
    num_restarts = 20
    max_iter = 1000
    T_start = 1.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max_iter)
    
    random.seed(42)
    np.random.seed(42)
    
    for restart in range(num_restarts):
        # Initialize
        if restart == 0:
            # Start with uniform
            log_w = np.zeros(n*n)
        else:
            # Start with random
            log_w = np.random.randn(n*n) * 0.5
            
        current_score = evaluate_log_weights(log_w)
        if current_score > best_score:
            best_score = current_score
            best_log_w = log_w.copy()
            
        T = T_start
        
        for i in range(max_iter):
            # Perturb
            # Pick a random subset of indices to perturb for better mixing?
            # Or just one?
            # Let's pick 5 random indices
            indices = np.random.choice(n*n, 5, replace=False)
            noise = np.random.randn(5) * 0.5
            new_log_w = log_w.copy()
            new_log_w[indices] += noise
            
            new_score = evaluate_log_weights(new_log_w)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                log_w = new_log_w
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_log_w = log_w.copy()
            
            T *= cooling
            
    # Convert best log weights to P
    if best_log_w is None:
        P = np.ones((n, n), dtype=np.float64) / (n*n)
    else:
        P = log_weights_to_probs(best_log_w).reshape(n, n)
        
    # Final safety checks
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n*n)
    else:
        P /= s
        
    return P
# EVOLVE-BLOCK-END
