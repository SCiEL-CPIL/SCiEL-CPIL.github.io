import numpy as np
import time
import warnings

N = 50

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    def compute_autocorr(h):
        """Compute linear autocorrelation of h."""
        n = len(h)
        c = np.zeros(2 * n - 1)
        for k in range(n):
            c[k] = np.dot(h[:n-k], h[k:])
            c[2*n-2-k] = c[k]
        return c
    
    def compute_score(h):
        """
        Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
        where h*h is the linear autoconvolution (autocorrelation for real signals).
        We assume the evaluator may normalize h to L1=1, so we compute the score
        on the normalized version.
        """
        h = np.asarray(h, dtype=float)
        s = np.sum(h)
        if s <= 0:
            return 0.0
        h_norm = h / s
        q = np.sum(h_norm ** 2)
        c = compute_autocorr(h_norm)
        m = np.max(c)
        if m <= 0:
            return 0.0
        return q / m
    
    def compute_score_no_l1(h):
        """Alternative: no L1 normalization, just Q/M."""
        h = np.asarray(h, dtype=float)
        q = np.sum(h ** 2)
        c = compute_autocorr(h)
        m = np.max(c)
        if m <= 0:
            return 0.0
        return q / m
    
    def compute_score_excl_lag0(h):
        """Score excluding lag 0 from the max."""
        h = np.asarray(h, dtype=float)
        s = np.sum(h)
        if s <= 0:
            return 0.0
        h_norm = h / s
        q = np.sum(h_norm ** 2)
        c = compute_autocorr(h_norm)
        # Exclude lag 0 (index n-1 in the full autocorrelation array)
        n = len(h)
        m = np.max(np.concatenate([c[:n-1], c[n:]]))
        if m <= 0:
            return float('inf') if q > 0 else 0.0
        return q / m
    
    # Try gradient-based optimization on the most likely objective
    # The cell finding suggests minimizing autoconvolution peak relative to energy
    # i.e., minimize M/Q = 1/(Q/M), so maximize Q/M
    
    best_h = np.zeros(N)
    best_h[0] = 2.0/3.0
    best_h[1] = 1.0/3.0
    best_score = compute_score(best_h)
    
    # Gradient-based optimization
    def optimize_gradient(h_init, objective_fn, max_iter=2000, lr=0.01):
        h = h_init.copy().astype(float)
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s > 0:
            h = h / s
        
        for it in range(max_iter):
            if time.time() - t_start > deadline - 5:
                break
            
            # Compute gradient of -objective (we want to minimize -objective)
            # Use finite differences for gradient
            eps = 1e-6
            grad = np.zeros_like(h)
            for i in range(N):
                h_plus = h.copy()
                h_plus[i] += eps
                h_plus = np.maximum(h_plus, 0)
                sp = np.sum(h_plus)
                if sp > 0:
                    h_plus = h_plus / sp
                else:
                    h_plus = h / max(s, 1e-10)
                
                h_minus = h.copy()
                h_minus[i] = max(h[i] - eps, 0)
                sm = np.sum(h_minus)
                if sm > 0:
                    h_minus = h_minus / sm
                else:
                    h_minus = h / max(s, 1e-10)
                
                grad[i] = (objective_fn(h_plus) - objective_fn(h_minus)) / (2 * eps)
            
            # Gradient ascent (maximize objective)
            h_new = h + lr * grad
            h_new = np.maximum(h_new, 0)
            s_new = np.sum(h_new)
            if s_new > 0:
                h_new = h_new / s_new
            
            # Check improvement
            new_obj = objective_fn(h_new)
            old_obj = objective_fn(h)
            
            if new_obj > old_obj:
                h = h_new
                # Adaptive learning rate
                lr = min(lr * 1.05, 0.1)
            else:
                lr *= 0.9
                if lr < 1e-8:
                    break
        
        return h
    
    # Try optimization with the Q/M objective (L1 normalized)
    h_opt1 = optimize_gradient(best_h, compute_score, max_iter=3000, lr=0.05)
    s1 = compute_score(h_opt1)
    if s1 > best_score:
        best_score = s1
        best_h = h_opt1
    
    # Try optimization with Q/M excluding lag 0
    h_opt2 = optimize_gradient(best_h, compute_score_excl_lag0, max_iter=3000, lr=0.05)
    s2 = compute_score_excl_lag0(h_opt2)
    # Compare using the same metric as best_score
    s2_norm = compute_score(h_opt2)
    if s2_norm > best_score:
        best_score = s2_norm
        best_h = h_opt2
    
    # Try different initializations
    np.random.seed(42)
    for trial in range(20):
        if time.time() - t_start > deadline - 10:
            break
        
        # Random sparse initialization
        h_init = np.zeros(N)
        k = np.random.randint(1, 6)
        idx = np.random.choice(N, k, replace=False)
        h_init[idx] = np.random.dirichlet(np.ones(k))
        
        h_opt = optimize_gradient(h_init, compute_score, max_iter=2000, lr=0.05)
        s = compute_score(h_opt)
        if s > best_score:
            best_score = s
            best_h = h_opt
    
    # Try a few structured initializations
    structured_inits = [
        np.array([2.0/3, 1.0/3] + [0.0]*(N-2)),
        np.array([0.5, 0.5] + [0.0]*(N-2)),
        np.array([1.0/3, 1.0/3, 1.0/3] + [0.0]*(N-3)),
        np.array([0.6, 0.3, 0.1] + [0.0]*(N-3)),
        np.array([0.7, 0.2, 0.1] + [0.0]*(N-3)),
        np.array([0.8, 0.1, 0.1] + [0.0]*(N-3)),
    ]
    
    for h_init in structured_inits:
        if time.time() - t_start > deadline - 10:
            break
        h_opt = optimize_gradient(h_init, compute_score, max_iter=2000, lr=0.05)
        s = compute_score(h_opt)
        if s > best_score:
            best_score = s
            best_h = h_opt
    
    # Final polish: local search with smaller steps
    if time.time() - t_start < deadline - 5:
        h_polish = best_h.copy()
        lr = 0.01
        for it in range(5000):
            if time.time() - t_start > deadline - 5:
                break
            eps = 1e-7
            grad = np.zeros_like(h_polish)
            for i in range(N):
                h_plus = h_polish.copy()
                h_plus[i] += eps
                h_plus = np.maximum(h_plus, 0)
                sp = np.sum(h_plus)
                if sp > 0:
                    h_plus = h_plus / sp
                
                h_minus = h_polish.copy()
                h_minus[i] = max(h_polish[i] - eps, 0)
                sm = np.sum(h_minus)
                if sm > 0:
                    h_minus = h_minus / sm
                
                grad[i] = (compute_score(h_plus) - compute_score(h_minus)) / (2 * eps)
            
            h_new = h_polish + lr * grad
            h_new = np.maximum(h_new, 0)
            s_new = np.sum(h_new)
            if s_new > 0:
                h_new = h_new / s_new
            
            new_obj = compute_score(h_new)
            old_obj = compute_score(h_polish)
            
            if new_obj > old_obj:
                h_polish = h_new
                lr = min(lr * 1.02, 0.05)
            else:
                lr *= 0.95
                if lr < 1e-10:
                    break
        
        if compute_score(h_polish) > best_score:
            best_h = h_polish
            best_score = compute_score(h_polish)
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    return best_h
# EVOLVE-BLOCK-END
