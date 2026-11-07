import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _eval(h):
    """Compute the exact value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return 0.0
        
    g = np.convolve(h, h)
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def _smooth_eval(h, k):
    """
    Smoothed objective function for L-BFGS-B.
    Uses log-sum-exp for soft max of norm_inf.
    Returns -value (to minimize).
    """
    h = np.maximum(h, 0)
    if np.all(h == 0):
        return 0.0
        
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # Soft max for norm_inf
    max_g = np.max(g)
    if k == 0:
        norm_inf = max_g
    else:
        shifted = k * (g - max_g)
        shifted = np.clip(shifted, -50, 50)
        exp_sum = np.sum(np.exp(shifted))
        norm_inf = max_g + (1.0 / k) * np.log(exp_sum)
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return -value

def _lbfgs_smooth(h0, k, maxiter=500):
    """Run L-BFGS-B with smoothed objective."""
    try:
        res = minimize(
            _smooth_eval,
            h0,
            args=(k,),
            method='L-BFGS-B',
            bounds=[(0, 100)] * N,
            options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        return res.x
    except Exception:
        return h0

def _lbfgs_exact(h0, maxiter=300):
    """Run L-BFGS-B with exact objective (using subgradient approximation)."""
    # For exact, we can't use gradient directly due to non-differentiability.
    # Instead, we use a very high k in the smooth version as a proxy.
    return _lbfgs_smooth(h0, k=5000, maxiter=maxiter)

def _cd(h, max_steps=100):
    """Coordinate descent refinement."""
    h = np.maximum(h, 0).copy()
    current_val = _eval(h)
    
    for step in range(max_steps):
        improved = False
        for i in range(N):
            old_h = h[i]
            # Try a few perturbations
            for delta in [0.01, 0.05, 0.1, 0.5, -0.01, -0.05, -0.1, -0.5]:
                new_h = h[i] + delta
                if new_h < 0:
                    continue
                h[i] = new_h
                val = _eval(h)
                if val > current_val:
                    current_val = val
                    improved = True
                else:
                    h[i] = old_h
                    break
        if not improved:
            break
    return h

def _make_initial_guesses():
    """Generate diverse initial points."""
    np.random.seed(42)
    initial_points = []
    
    # 1. Uniform
    initial_points.append(np.ones(N))
    
    # 2. Random Uniform
    for _ in range(15):
        initial_points.append(np.random.rand(N))
        
    # 3. Sparse Spikes
    for _ in range(15):
        h = np.zeros(N)
        idx = np.random.choice(N, size=np.random.randint(3, 10), replace=False)
        h[idx] = 1.0
        initial_points.append(h)
        
    # 4. Blocky Patterns
    for _ in range(15):
        h = np.zeros(N)
        start = np.random.randint(0, N - 5)
        length = np.random.randint(5, 20)
        h[start:start+length] = 1.0
        initial_points.append(h)
        
    # 5. Gaussian Bumps
    for _ in range(15):
        x = np.linspace(-1, 1, N)
        center = np.random.uniform(-0.8, 0.8)
        width = np.random.uniform(0.1, 0.6)
        h = np.exp(-((x - center)**2) / (2 * width**2))
        initial_points.append(h)
        
    # 6. Cosine Waves
    for _ in range(10):
        x = np.linspace(0, 1, N)
        freq = np.random.uniform(1, 5)
        phase = np.random.uniform(0, 2*np.pi)
        h = np.abs(np.cos(2*np.pi*freq*x + phase))
        initial_points.append(h)
        
    # 7. Triangular
    for _ in range(5):
        x = np.linspace(0, 1, N)
        center = np.random.uniform(0.2, 0.8)
        width = np.random.uniform(0.2, 0.5)
        h = np.maximum(0, 1 - np.abs(x - center) / width)
        initial_points.append(h)
        
    # 8. Two bumps
    for _ in range(5):
        x = np.linspace(-1, 1, N)
        c1 = np.random.uniform(-0.8, -0.2)
        c2 = np.random.uniform(0.2, 0.8)
        w1 = np.random.uniform(0.1, 0.4)
        w2 = np.random.uniform(0.1, 0.4)
        h = np.exp(-((x - c1)**2) / (2 * w1**2)) + np.exp(-((x - c2)**2) / (2 * w2**2))
        initial_points.append(h)
        
    np.random.shuffle(initial_points)
    return initial_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_h = np.ones(N)
    best_val = _eval(best_h)
    
    initial_points = _make_initial_guesses()
    
    # k values for soft max annealing - more stages, sharper
    k_values = [5, 20, 100, 500, 2000, 10000]
    
    for i, h0 in enumerate(initial_points):
        if time.time() > deadline - 15:
            break
            
        h0 = np.maximum(h0, 0)
        
        # Normalize h0 to have reasonable magnitude
        if np.max(h0) > 0:
            h0 = h0 / np.max(h0) * 1.0
            
        current_h = h0.copy()
        
        # Anneal k: start smooth, get sharper
        for k in k_values:
            if time.time() > deadline - 5:
                break
                
            current_h = _lbfgs_smooth(current_h, k, maxiter=300)
        
        # Coordinate descent polish
        if time.time() < deadline - 3:
            current_h = _cd(current_h, max_steps=50)
        
        # Final high-precision polish with very high k
        if time.time() < deadline - 2:
            current_h = _lbfgs_smooth(current_h, k=50000, maxiter=200)
        
        # Final check
        current_h = np.maximum(current_h, 0)
        val = _eval(current_h)
        if val > best_val:
            best_val = val
            best_h = current_h.copy()
            
    return best_h.tolist()
