import numpy as np
import time

N = 100

def _score(h):
    """Compute the maximum overlap of h with its shifted 1-h."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=500.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _fix_sum(h, target=None):
    """Project h to have exact sum of N/2 while staying in [0,1]."""
    if target is None:
        target = N / 2.0
    h = np.array(h, dtype=float)
    for _ in range(100):
        diff = target - np.sum(h)
        h += diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        if abs(target - np.sum(h)) < 1e-15:
            break
    # Final exact correction
    diff = target - np.sum(h)
    if abs(diff) > 0:
        # Distribute the remaining difference
        h += diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        # If still not exact, adjust one element
        diff2 = target - np.sum(h)
        if abs(diff2) > 1e-15:
            # Find an element that can absorb the difference
            for i in range(len(h)):
                if diff2 > 0 and h[i] < 1.0 - abs(diff2):
                    h[i] += diff2
                    break
                elif diff2 < 0 and h[i] > abs(diff2):
                    h[i] += diff2
                    break
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    x = np.arange(N)
    
    # Start with uniform
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)
    
    # Strategy 1: Two-level profiles with smooth transition
    # h[i] = 0.5 + A * tanh(B*(i-C))
    
    def make_profile(A, B, C):
        """Create profile h[i] = 0.5 + A * tanh(B*(i-C))"""
        i = np.arange(N)
        h = 0.5 + A * np.tanh(B * (i - C))
        return h
    
    def optimize_profile(B):
        """For fixed B, optimize A and C to satisfy sum constraint and minimize score."""
        best_score = float('inf')
        best_AC = None
        
        for A in np.linspace(0.05, 0.5, 30):
            # For this A and B, find C such that sum(h) = N/2
            C0 = (N - 1) / 2.0
            h = make_profile(A, B, C0)
            sum_h = np.sum(h)
            
            if abs(sum_h - N/2.0) > 1e-10:
                # Binary search for C
                C_lo = 0.0
                C_hi = N - 1.0
                for _ in range(60):
                    C_mid = (C_lo + C_hi) / 2.0
                    h_mid = make_profile(A, B, C_mid)
                    sum_mid = np.sum(h_mid)
                    if sum_mid > N/2.0:
                        C_hi = C_mid
                    else:
                        C_lo = C_mid
                C0 = (C_lo + C_hi) / 2.0
            
            h = make_profile(A, B, C0)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_AC = (A, C0)
        
        return best_AC
    
    # Try different steepness values B
    for B in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 20:
            break
        
        best_params = optimize_profile(B)
        if best_params is not None:
            A, C = best_params
            h_candidate = make_profile(A, B, C)
            h_candidate = _fix_sum(h_candidate)
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
    
    # Strategy 2: Two-level step function
    # h[i] = v1 for i < k, v2 for i >= k
    
    for k in range(1, N):
        if time.time() > deadline - 10:
            break
        
        v1_min = max(0.0, 1.0 - N/(2.0*k))
        v1_max = min(1.0, N/(2.0*k))
        
        if v1_min >= v1_max:
            continue
        
        for v1 in np.linspace(v1_min, v1_max, 20):
            v2 = (N/2.0 - v1*k) / (N-k)
            if v2 < -1e-10 or v2 > 1.0 + 1e-10:
                continue
            v2 = np.clip(v2, 0.0, 1.0)
            
            h = np.zeros(N)
            h[:k] = v1
            h[k:] = v2
            h = _fix_sum(h)
            sc = _score(h)
            if sc < score_best:
                score_best = sc
                h_best = h.copy()
    
    # Strategy 3: Direct vector optimization with smooth max objective
    # This is the main improvement: directly optimize all 100 values
    
    if time.time() < deadline - 5:
        from scipy.optimize import minimize
        
        # Use smooth max with adaptive alpha
        def objective_and_grad(h):
            h_clipped = np.clip(h, 0.0, 1.0)
            
            shifted = np.zeros(N)
            shifted[:-1] = h_clipped[1:]
            overlap = h_clipped * (1.0 - shifted)
            
            # Smooth max
            alpha = 100.0
            x_max = np.max(overlap)
            val = x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (overlap - x_max))))
            
            # Penalty for constraint violation
            penalty = 100.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
            
            total = val + penalty
            
            # Gradient
            # Gradient of smooth max
            weights = np.exp(alpha * (overlap - x_max))
            weights /= np.sum(weights)
            
            # d(overlap)/d(h_clipped)
            # overlap[i] = h_clipped[i] * (1 - h_clipped[i+1]) for i < N-1
            # overlap[N-1] = h_clipped[N-1] * (1 - 0) = h_clipped[N-1]
            
            grad_overlap = np.zeros(N)
            # d(overlap[i])/d(h[i]) = 1 - h[i+1] for i < N-1, = 1 for i=N-1
            grad_overlap[:-1] = 1.0 - h_clipped[1:]
            grad_overlap[N-1] = 1.0
            
            # d(overlap[i])/d(h[i+1]) = -h[i] for i < N-1
            grad_overlap[:-1] += -h_clipped[:-1]
            
            grad_val = np.dot(weights, grad_overlap) / alpha
            
            # Gradient of penalty
            grad_penalty = 200.0 * (np.maximum(0, h - 1.0) - np.maximum(0, -h))
            
            grad_total = grad_val + grad_penalty
            
            return total, grad_total
        
        # Initialize from best h found so far
        h0 = h_best.copy()
        
        try:
            res = minimize(
                objective_and_grad,
                h0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            
            h_candidate = res.x
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
                
        except Exception:
            pass
    
    # Strategy 4: Local search polish - try small perturbations around best
    if time.time() < deadline - 1:
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            
            h_trial = h_best.copy()
            # Random perturbation
            idx = np.random.randint(0, N, 3)
            h_trial[idx] += np.random.normal(0, 0.01, 3)
            h_trial = np.clip(h_trial, 0.0, 1.0)
            h_trial = _fix_sum(h_trial)
            
            sc = _score(h_trial)
            if sc < score_best:
                score_best = sc
                h_best = h_trial.copy()
    
    return h_best
