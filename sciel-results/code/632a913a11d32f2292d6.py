import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # Initial guess: ones
    h0 = np.ones(N)
    
    # We want to minimize max(A(h))
    # Use a soft-max approximation
    # f(h) = log(sum(exp(beta * A(h)))) / beta
    # A(h) is quadratic.
    
    # To handle the constraint sum(h)=1, we can parameterize h or use constraints.
    # Let's use constraints.
    
    def objective(h):
        # Compute autocorrelation
        # h is length N. conv is length 2N-1.
        # We only care about the max.
        # For speed, we can compute only the central part?
        # No, max can be anywhere.
        # But for smooth h, max is likely central.
        
        # Use FFT for speed?
        # N=400 is small. Direct conv is fine.
        conv = np.convolve(h, h, mode='full')
        
        # Soft max
        beta = 10.0
        # Shift to avoid overflow
        max_conv = np.max(conv)
        if max_conv > 100:
             return max_conv # Just return the max if it's huge
        exp_conv = np.exp(beta * (conv - max_conv))
        soft_max = max_conv + np.log(np.sum(exp_conv)) / beta
        return soft_max

    def jac(h):
        # Gradient of soft max is hard to compute analytically with conv.
        # Use numerical gradient? Too slow for 400 dims?
        # 400 * 2 evaluations per step. 100 steps. 80k evals.
        # Each eval is conv of 400. Fast.
        # Let's try numerical.
        eps = 1e-5
        grad = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            grad[i] = (objective(h_plus) - objective(h)) / eps
        return grad

    # This is too slow.
    # Let's use a simpler heuristic.
    
    # Start with h = ones.
    h = np.ones(N)
    S = np.sum(h)
    
    # Try to reduce the central peak by flipping signs of elements far from center?
    # Or use a random perturbation and local search.
    
    best_h = h.copy()
    best_val = np.inf
    
    # Calculate value for seed
    def get_val(h):
        S = np.sum(h)
        if S == 0: return np.inf
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        return abs(2 * N * max_c / (S * S))
        
    best_val = get_val(best_h)
    
    # Local search: flip signs of single elements
    # But we must maintain S != 0.
    # If we flip h[i] to -h[i], S changes by -2h[i].
    
    # Let's do a few iterations of "flip the element that reduces the max most"
    # This is greedy.
    
    for _ in range(100):
        improved = False
        for i in range(N):
            h_new = h.copy()
            h_new[i] = -h_new[i]
            val_new = get_val(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new
                h = h_new
                improved = True
        if not improved:
            break
            
    return best_h
