import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    N = len(h)
    max_ov = 0.0
    for s in range(1, N):
        shifted = np.roll(h, -s)
        ov = np.sum(h * (1 - shifted)) / N
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _objective(h):
    return _compute_max_overlap(h)

def construct():
    deadline = time.time() + 140.0  # safety margin
    
    # Start with the seed
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try some structured patterns
    patterns = []
    
    # Pattern 1: 0/1 alternating
    p1 = np.zeros(N)
    p1[::2] = 1.0
    if np.sum(p1) == N/2:
        patterns.append(p1)
    
    # Pattern 2: quadratic residue
    # ... etc
    
    # Use scipy to optimize
    from scipy.optimize import minimize
    
    def obj_and_grad(h_flat):
        h = h_flat  # already in [0,1] if we project
        return _compute_max_overlap(h)
    
    # Use a smooth approximation or just direct
    x0 = np.full(N, 0.5)
    
    # Constraints: sum(h) = N/2, 0 <= h <= 1
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    bounds = [(0, 1)] * N
    
    # This is non-smooth (max of linear functions), so use SLSQP or similar
    # Actually, let's try a different approach: parametrize as h = 0.5 + 0.5*sin(...) or use a smooth parameterization
    
    # Let's try: h[i] = 0.5 + a*sin(2*pi*k*i/N + phase) for various k
    # This automatically satisfies sum = N/2
    
    best_found = best_score
    
    for k in range(1, N//2 + 1):
        for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
            h = 0.5 + 0.5 * np.sin(2*np.pi*k*np.arange(N)/N + phase)
            h = np.clip(h, 0, 1)
            # Re-normalize
            if np.sum(h) != N/2:
                h = h / np.sum(h) * N/2
                h = np.clip(h, 0, 1)
                # This might not sum exactly, so adjust
                diff = N/2 - np.sum(h)
                # Distribute diff
                h += diff / N
                h = np.clip(h, 0, 1)
            
            sc = _compute_max_overlap(h)
            if sc < best_found:
                best_found = sc
                best_h = h.copy()
    
    # Try scipy optimization
    if time.time() < deadline - 30:
        try:
            # Use a smooth objective: max overlap is max of linear functions, use smooth max
            def smooth_max_overlap(h, t=1.0):
                N = len(h)
                ovs = []
                for s in range(1, N):
                    shifted = np.roll(h, -s)
                    ov = np.sum(h * (1 - shifted)) / N
                    ovs.append(ov)
                ovs = np.array(ovs)
                # Smooth max: log-sum-exp approximation
                return np.max(ovs)  # Just use hard max for now
            
            # Parameterize: h[i] = sigmoid(x[i]) to ensure [0,1]
            # But we need sum = N/2
            
            # Alternative: use projected gradient
            h = best_h.copy()
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute subgradient of max overlap
                max_s = None
                max_ov = -1
                for s in range(1, N):
                    shifted = np.roll(h, -s)
                    ov = np.sum(h * (1 - shifted)) / N
                    if ov > max_ov:
                        max_ov = ov
                        max_s = s
                
                shifted = np.roll(h, -max_s)
                # Gradient of overlap at max_s: d/dh_i [h_i(1-h_{i+s})] = (1-h_{i+s}) - h_i*h_{i-s}
                # Wait: overlap = sum_i h[i](1-h[(i+s)%N])/N
                # d/dh[i] = (1-h[(i+s)%N])/N
                # d/dh[(i+s)%N] = -h[i]/N
                grad = np.zeros(N)
                for i in range(N):
                    grad[i] += (1 - h[(i+max_s)%N]) / N
                    grad[(i+max_s)%N] += -h[i] / N
                grad /= N  # Already divided by N in the sum... let me recheck
                
                # overlap(s) = (1/N) sum_i h[i](1-h[i+s])
                # d overlap/d h[i] = (1/N)(1-h[i+s]) + sum over j where j+s=i of h[j]*(-1)/N
                # = (1/N)(1-h[i+s]) - (1/N)h[i-s]
                
                grad = np.zeros(N)
                for i in range(N):
                    grad[i] = (1 - h[(i+max_s)%N])/N - h[(i-max_s)%N]/N
                
                # Projected gradient step
                h_new = h - lr * grad
                # Project to sum = N/2, 0<=h<=1
                h_new = np.clip(h_new, 0, 1)
                diff = N/2 - np.sum(h_new)
                h_new += diff/N
                h_new = np.clip(h_new, 0, 1)
                
                sc = _compute_max_overlap(h_new)
                if sc < best_found:
                    best_found = sc
                    best_h = h_new.copy()
                h = h_new
        except:
            pass
    
    # Ensure exact sum
    diff = N/2 - np.sum(best_h)
    best_h += diff/N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
