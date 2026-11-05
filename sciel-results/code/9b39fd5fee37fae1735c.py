import numpy as np

N = 400

def construct():
    def evaluate(h):
        S = np.sum(h)
        if abs(S) < 1e-10:
            return np.inf
        C = np.convolve(h, h)
        M = np.max(C)
        return abs(2 * N * M / (S**2))
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Basis functions
    i = np.arange(N)
    x_norm = 2 * i / (N - 1) - 1 # -1 to 1
    
    bases = [
        np.ones(N),
        x_norm**2,
        np.cos(2 * np.pi * i / N),
        np.cos(4 * np.pi * i / N),
        np.sin(2 * np.pi * i / N),
    ]
    
    # We will search in the space spanned by these bases (excluding the constant term which is fixed to 1 to keep sum ~ N)
    # h = 1 + sum_j theta_j * (bases[j] - mean(bases[j]))
    # Centering the bases ensures they don't shift the mean significantly, keeping S stable.
    
    centered_bases = []
    for b in bases[1:]:
        b_c = b - np.mean(b)
        centered_bases.append(b_c)
        
    # Simple grid search over 2 parameters (first two non-constant bases)
    # theta1 for x^2, theta2 for cos(2pi i/N)
    
    # We can do a nested loop
    # theta1 in [-0.5, 0.5], theta2 in [-0.5, 0.5]
    
    # To be safe and fast, we'll do a coarse grid then a fine grid.
    
    def get_h(theta1, theta2):
        h = np.ones(N)
        h += theta1 * centered_bases[0]
        h += theta2 * centered_bases[1]
        return h
        
    # Coarse grid
    for t1 in np.linspace(-0.5, 0.5, 10):
        for t2 in np.linspace(-0.5, 0.5, 10):
            h = get_h(t1, t2)
            val = evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
    # Fine grid around the best found (implicitly by just doing a denser grid if we wanted, 
    # but 10x10 is 100 evals. 20x20 is 400. 50x50 is 2500. All are fast.)
    
    # Let's do a 50x50 grid directly.
    for t1 in np.linspace(-0.5, 0.5, 50):
        for t2 in np.linspace(-0.5, 0.5, 50):
            h = get_h(t1, t2)
            val = evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
    # Add a third parameter?
    # h = 1 + t1*b1 + t2*b2 + t3*b3
    # 50^3 = 125,000 evals. Might be too slow?
    # 125,000 * 160,000 ops = 20 billion ops. Too slow.
    # Stick to 2 parameters or use gradient descent on the 3 parameters.
    
    # Let's use Nelder-Mead on the 3 parameters.
    from scipy.optimize import minimize
    
    def objective(params):
        h = np.ones(N)
        h += params[0] * centered_bases[0]
        h += params[1] * centered_bases[1]
        if len(centered_bases) > 2:
            h += params[2] * centered_bases[2]
        return evaluate(h)
        
    # Initial guess from best grid point? 
    # Or just start at 0,0,0 (seed)
    # Since we already found a good point, let's use it as start?
    # The grid search above updates best_h. 
    # We can extract the params? Hard.
    # Just run minimize from 0,0,0 and 0.1,0,0 etc.
    
    # Actually, the grid search is robust. Let's just do a finer grid on the 2 params.
    # 100x100 = 10,000 evals.
    # 10,000 * 160,000 = 1.6 billion ops. ~1-2 seconds in Python/NumPy? 
    # np.convolve is optimized. 10k calls should be fine within 120s.
    
    for t1 in np.linspace(-0.5, 0.5, 100):
        for t2 in np.linspace(-0.5, 0.5, 100):
            h = get_h(t1, t2)
            val = evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
    return best_h
