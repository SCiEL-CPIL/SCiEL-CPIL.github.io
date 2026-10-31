import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(h):
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 30s safety margin before 150s kill
    
    # Precompute grid and basis functions
    x = np.arange(N)
    
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)
    
    # Precompute basis functions for various K
    def precompute_basis(K):
        basis = []
        for k in range(1, K + 1):
            cos_k = np.cos(2 * np.pi * k * x / N)
            sin_k = np.sin(2 * np.pi * k * x / N)
            basis.append((cos_k, sin_k))
        return basis
    
    def get_h_from_params(params, K, basis):
        h = np.full(N, 0.5)
        for k in range(K):
            cos_k, sin_k = basis[k]
            h += params[2*k] * cos_k
            h += params[2*k+1] * sin_k
        return h
    
    def compute_overlap_and_grad(h_clipped, alpha):
        shifted = np.zeros(N)
        shifted[:-1] = h_clipped[1:]
        overlap = h_clipped * (1.0 - shifted)
        
        # Smooth max gradient weights
        x_max = np.max(overlap)
        e = np.exp(alpha * (overlap - x_max))
        p = e / np.sum(e)
        
        # Gradient of overlap w.r.t h
        h_next = np.roll(h_clipped, -1)
        h_next[-1] = 0.0
        h_prev = np.roll(h_clipped, 1)
        h_prev[0] = 0.0
        
        grad_h = p * (1.0 - h_next) - np.roll(p, 1) * h_prev
        return overlap, grad_h, p
    
    def epigraph_objective(params, K, basis, alpha):
        """Minimize t subject to smooth_max(overlap) <= t, with bounds penalty."""
        h = get_h_from_params(params[:2*K], K, basis)
        t = params[2*K]
        
        h_clipped = np.clip(h, 0.0, 1.0)
        overlap, _, _ = compute_overlap_and_grad(h_clipped, alpha)
        sm = _smooth_max(overlap, alpha)
        
        # Penalty for bounds violation
        penalty = 50.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
        
        # Epigraph: minimize max(sm, t) + penalty, but we want to push t down
        # Actually, let's just minimize sm + penalty and use t as a slack variable
        # Simpler: just minimize smooth_max + penalty, no epigraph needed for L-BFGS-B
        return sm + penalty
    
    def epigraph_grad(params, K, basis, alpha):
        h = get_h_from_params(params[:2*K], K, basis)
        h_clipped = np.clip(h, 0.0, 1.0)
        overlap, grad_h, p = compute_overlap_and_grad(h_clipped, alpha)
        
        # Gradient of smooth max w.r.t h is grad_h (as computed)
        # Mask for clip
        mask = (h >= 0.0) & (h <= 1.0)
        total_grad_h = grad_h * mask
        
        # Penalty gradient
        grad_penalty = np.zeros(N)
        grad_penalty[h > 1.0] = 100.0 * (h[h > 1.0] - 1.0)
        grad_penalty[h < 0.0] = -100.0 * h[h < 0.0]
        
        total_grad_h += grad_penalty
        
        # Chain rule to params
        grad_params = np.zeros(len(params))
        for k in range(K):
            cos_k, sin_k = basis[k]
            grad_params[2*k] = np.dot(total_grad_h, cos_k)
            grad_params[2*k+1] = np.dot(total_grad_h, sin_k)
        
        # Gradient w.r.t t is 0 (t is not used in objective directly in this formulation)
        # If we were using epigraph properly, grad w.r.t t would be 1
        # But since we're just minimizing smooth_max, t is unused. Let's remove t.
        
        return grad_params
    
    def fix_sum(h):
        """Project h to have sum exactly N/2 while staying in [0,1]."""
        h = h.copy()
        for _ in range(20):
            diff = N/2.0 - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        # Final correction: if sum is still off, adjust the element closest to 0.5
        diff = N/2.0 - np.sum(h)
        if abs(diff) > 1e-10:
            # Find index where we can adjust without violating bounds
            idx = np.argmin(np.abs(h - 0.5))
            h[idx] += diff
            h[idx] = np.clip(h[idx], 0.0, 1.0)
            # If still off, distribute among multiple elements
            diff2 = N/2.0 - np.sum(h)
            if abs(diff2) > 1e-10:
                # Adjust elements with most slack
                slack = np.minimum(h, 1.0 - h)
                top_idx = np.argsort(slack)[-int(abs(diff2)*N*2):]
                per_elem = diff2 / len(top_idx)
                for i in top_idx:
                    h[i] += per_elem
                    h[i] = np.clip(h[i], 0.0, 1.0)
        return h
    
    # Try different numbers of Fourier modes with multiple restarts
    for K in [4, 6, 8, 10]:
        if time.time() > deadline - 30:
            break
        
        basis = precompute_basis(K)
        n_params = 2 * K
        
        # Multiple restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            if restart == 0:
                np.random.seed(42 + K)
                h0 = np.random.randn(n_params) * 0.15
            elif restart == 1:
                np.random.seed(100 + K)
                h0 = np.random.randn(n_params) * 0.1
            else:
                np.random.seed(200 + K)
                h0 = np.random.randn(n_params) * 0.2
            
            try:
                res = minimize(
                    lambda p: epigraph_objective(p, K, basis, alpha=150.0),
                    h0,
                    jac=lambda p: epigraph_grad(p, K, basis, alpha=150.0),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_candidate = get_h_from_params(res.x, K, basis)
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                h_candidate = fix_sum(h_candidate)
                
                sc = _score(h_candidate)
                if sc < score_best:
                    score_best = sc
                    h_best = h_candidate.copy()
                    
            except Exception as e:
                pass
    
    # Final polish: local search on the best solution
    if time.time() < deadline - 10:
        # Try direct local optimization on h with projected gradient
        h = h_best.copy()
        
        for alpha in [300.0, 500.0, 800.0]:
            if time.time() > deadline - 10:
                break
            
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                h_clipped = np.clip(h, 0.0, 1.0)
                overlap, grad_h, _ = compute_overlap_and_grad(h_clipped, alpha)
                
                # Projected gradient step
                h_new = h - lr * grad_h
                h_new = np.clip(h_new, 0.0, 1.0)
                h_new = fix_sum(h_new)
                
                sc_new = _score(h_new)
                if sc_new < score_best:
                    score_best = sc_new
                    h_best = h_new.copy()
                    h = h_new.copy()
                else:
                    # Reduce learning rate
                    lr *= 0.9
                    if lr < 1e-6:
                        break
    
    return h_best
