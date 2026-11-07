"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _evaluate(h):
    """Compute the task value for given heights h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h, mode='full')
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def construct():
    """Return 50 nonnegative step heights."""
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin
    
    # Start with the known best 4-point symmetric shape repeated
    # [1, 1.8, 1.8, 1] pattern
    base = np.array([1.0, 1.8, 1.8, 1.0])
    # Tile to length 50
    h0 = np.tile(base, 13)[:50]
    # Adjust last few to maintain symmetry if needed
    h0 = h0.copy()
    # Make it symmetric
    h0 = (h0 + h0[::-1]) / 2.0
    
    best_h = h0.copy()
    best_val = _evaluate(best_h)
    
    # Try some structured patterns first
    patterns = []
    
    # Uniform
    patterns.append(np.ones(50))
    
    # Triangular (peak in middle)
    tri = np.linspace(0, 1, 25)
    tri = np.concatenate([tri, tri[:-1][::-1]])
    patterns.append(tri)
    
    # Sine wave
    x = np.linspace(0, 2*np.pi, 50)
    patterns.append(np.maximum(np.sin(x), 0))
    
    # Two peaks
    x = np.linspace(0, 1, 50)
    two_peak = np.exp(-10*(x-0.25)**2) + np.exp(-10*(x-0.75)**2)
    patterns.append(two_peak)
    
    # Cosine window
    x = np.linspace(-1, 1, 50)
    patterns.append(0.5 + 0.5*np.cos(np.pi*x))
    
    # Plateau
    plateau = np.zeros(50)
    plateau[10:40] = 1.0
    patterns.append(plateau)
    
    # The 4-point pattern tiled
    patterns.append(h0.copy())
    
    # Evaluate all patterns
    for p in patterns:
        val = _evaluate(p)
        if val > best_val:
            best_val = val
            best_h = p.copy()
    
    if time.time() > deadline:
        return best_h.tolist()
    
    # Gradient-free optimization: use a simple coordinate ascent + random perturbation
    # Try scipy's differential evolution if available and time permits
    try:
        from scipy.optimize import differential_evolution, minimize
    except ImportError:
        differential_evolution = None
        minimize = None
    
    # Try a few local optimizations from different starting points
    start_points = [best_h.copy()]
    # Add some random perturbations
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = best_h * (1 + 0.3 * rng.randn(50))
        perturbed = np.maximum(perturbed, 0)
        start_points.append(perturbed)
    
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        
        # Simple coordinate descent
        h = sp.copy()
        improved = True
        max_iters = 200
        it = 0
        while improved and it < max_iters and time.time() < deadline - 5:
            improved = False
            it += 1
            for i in range(N):
                # Try increasing
                h_old = h[i]
                h[i] = h_old * 1.2
                if h[i] > 10:
                    h[i] = 10
                val = _evaluate(h)
                if val > best_val:
                    best_val = val
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = h_old
                
                # Try decreasing
                h[i] = h_old * 0.8
                val = _evaluate(h)
                if val > best_val:
                    best_val = val
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = h_old
            
            if not improved:
                break
    
    # Try a few more random restarts with local search
    if time.time() < deadline - 15:
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random start with some structure
            h = np.maximum(0.5 + 0.5*rng.randn(50), 0)
            h = (h + h[::-1]) / 2  # symmetrize
            
            # Quick coordinate ascent
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                i = rng.randint(N)
                h_old = h[i]
                h[i] = h_old * (1 + 0.3 * rng.randn())
                h[i] = max(h[i], 0)
                val = _evaluate(h)
                if val > best_val:
                    best_val = val
                    best_h = h.copy()
                else:
                    h[i] = h_old
    
    # Final: try scipy minimize if time allows
    if minimize is not None and time.time() < deadline - 10:
        try:
            def neg_value(h_flat):
                h = np.maximum(h_flat, 0)
                return -_evaluate(h)
            
            res = minimize(neg_value, best_h, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.x is not None:
                h_opt = np.maximum(res.x, 0)
                val_opt = _evaluate(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0)
    best_h = np.where(np.isfinite(best_h), best_h, 0)
    
    return best_h.tolist()

# EVOLVE-BLOCK-END
