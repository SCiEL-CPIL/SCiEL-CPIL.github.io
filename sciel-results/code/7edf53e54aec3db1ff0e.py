"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.array(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with seed
    best_h = [1.0] * N
    best_val = compute_value(best_h)
    
    # Try various analytical patterns that might improve the ratio
    # The key insight: we want to maximize l2 / (norm_1 * norm_inf)
    # l2 involves adjacent products, norm_1 is L1 norm, norm_inf is max
    
    # For uniform heights, all g values are equal, so:
    # norm_inf = g[0] (max), norm_1 = sum(g)/(m+1)
    # If we make g more peaked, norm_inf increases but norm_1 stays similar
    # If we spread g more evenly, norm_inf decreases but norm_1 stays similar
    
    # Let's try: concentrate mass in center to reduce norm_inf relative to spread
    # Or: create a pattern where g has many equal large values
    
    # Try triangular distribution
    for trial in range(500):
        if time.time() > deadline:
            break
            
        # Different strategies
        strategy = trial % 5
        
        if strategy == 0:
            # Gaussian-like
            x = np.linspace(-1, 1, N)
            sigma = 0.3 + 0.7 * (trial // 5 % 100) / 100.0
            h = np.exp(-x**2 / (2 * sigma**2))
            
        elif strategy == 1:
            # Cosine bump
            x = np.linspace(-1, 1, N)
            width = 0.5 + 0.5 * (trial // 5 % 100) / 100.0
            h = np.maximum(0, np.cos(np.pi * x / (2 * width)))
            
        elif strategy == 2:
            # Uniform but with edge decay
            decay = (trial // 5 % 100) / 100.0
            x = np.linspace(0, 1, N)
            h = 1.0 - decay * np.abs(x - 0.5) * 2
            
        elif strategy == 3:
            # Bimodal
            center1 = 0.3 + 0.2 * (trial // 5 % 100) / 100.0
            center2 = 0.7 - 0.2 * (trial // 5 % 100) / 100.0
            x = np.linspace(0, 1, N)
            h = np.exp(-((x - center1)**2) / 0.02) + np.exp(-((x - center2)**2) / 0.02)
            
        else:
            # Plateau with sharp edges
            edge_width = 5 + 15 * (trial // 5 % 100) / 100.0
            h = np.ones(N)
            for i in range(int(edge_width)):
                if i < N:
                    h[i] = i / edge_width
                    h[N-1-i] = i / edge_width
        
        h = np.maximum(0, h)
        if np.all(h == 0):
            continue
            
        val = compute_value(h)
        if val > best_val:
            best_val = val
            best_h = h.tolist()
    
    # Local refinement using gradient-free optimization
    if time.time() < deadline - 10:
        # Try Nelder-Mead style local search
        step_size = 0.1
        for iteration in range(5000):
            if time.time() > deadline:
                break
                
            improved = False
            idx = np.random.randint(0, N)
            delta = np.random.choice([-1, 1]) * step_size * np.random.rand()
            
            new_h = best_h.copy()
            new_h[idx] = max(0, new_h[idx] + delta)
            
            val = compute_value(new_h)
            if val > best_val:
                best_val = val
                best_h = new_h
                improved = True
            
            if not improved and iteration % 100 == 0:
                step_size *= 0.95
    
    return best_h
# EVOLVE-BLOCK-END
