import numpy as np

N = 50

def compute_value(h):
    """
    Computes the value for a given height array h.
    h must be non-negative.
    """
    if np.any(h < 0):
        return -np.inf
    if np.sum(h) == 0:
        return 0.0
        
    # Normalize to sum 1 to avoid scaling issues, though value is scale invariant
    # But keeping sum 1 helps with numerical stability in exp/log if we were using them, 
    # here we just use raw h.
    # Actually, let's just compute directly.
    
    g = np.convolve(h, h, mode='full')
    
    # y = [0, g[0], ..., g[m-1], 0]
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # w is spacing. 
    # The points are np.linspace(-0.5, 0.5, m + 2).
    # Length is m+2. Number of intervals is m+1.
    # w = 1.0 / (m + 1).
    
    m = len(g)
    w = 1.0 / (m + 1)
    
    # Construct y implicitly to save memory/time
    # y[0] = 0
    # y[1..m] = g[0..m-1]
    # y[m+1] = 0
    
    # Sum term:
    # i=0: y[0]=0, y[1]=g[0]. Term: (0 + 0 + g[0]^2) = g[0]^2
    # i=1..m-1: y[i]=g[i-1], y[i+1]=g[i]. Term: g[i-1]^2 + g[i-1]g[i] + g[i]^2
    # i=m: y[m]=g[m-1], y[m+1]=0. Term: g[m-1]^2 + 0 + 0 = g[m-1]^2
    
    # Total sum S = g[0]^2 + g[m-1]^2 + sum_{i=1}^{m-1} (g[i-1]^2 + g[i-1]g[i] + g[i]^2)
    # S = sum(g^2) + sum(g^2) + sum(g[:-1]*g[1:])?
    # Let's expand:
    # sum_{i=0}^{m-1} g[i]^2 is in the first part?
    # Term i=0: g[0]^2
    # Term i=m: g[m-1]^2
    # Middle terms: sum_{k=0}^{m-2} (g[k]^2 + g[k]g[k+1] + g[k+1]^2)
    # Sum of all terms:
    # g[0]^2 (from i=0)
    # + sum_{k=0}^{m-2} g[k]^2 + g[k]g[k+1] + g[k+1]^2
    # + g[m-1]^2 (from i=m)
    
    # Coefficient of g[j]^2:
    # If j=0: from i=0 (1) + from middle k=0 (1) = 2.
    # If j=m-1: from middle k=m-2 (1) + from i=m (1) = 2.
    # If 0 < j < m-1: from middle k=j-1 (1) + from middle k=j (1) = 2.
    
    # So sum of squares part is 2 * sum(g^2).
    # Cross term part is sum_{k=0}^{m-2} g[k]g[k+1].
    
    # So l2 = (w/3) * (2 * sum(g^2) + sum(g[:-1]*g[1:]))
    
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:])
    
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def construct():
    np.random.seed(42)
    
    best_h = np.ones(N) / N
    best_val = compute_value(best_h)
    
    # Define several initial shapes
    # 1. Uniform
    h0 = np.ones(N) / N
    
    # 2. Center Peak (Gaussian-like)
    x = np.linspace(-1, 1, N)
    h1 = np.exp(-x**2 * 5)
    h1 = h1 / np.sum(h1)
    
    # 3. Two Peaks
    h2 = np.exp(-(x-0.5)**2 * 10) + np.exp(-(x+0.5)**2 * 10)
    h2 = h2 / np.sum(h2)
    
    # 4. Random
    h3 = np.random.rand(N)
    h3 = h3 / np.sum(h3)
    
    # 5. Edge Peak
    h4 = np.exp(-x * 5)
    h4 = h4 / np.sum(h4)
    
    starts = [h0, h1, h2, h3, h4]
    
    for h_start in starts:
        h = h_start.copy()
        current_val = compute_value(h)
        
        # Hill Climbing
        step_size = 0.1
        for i in range(5000): # Iterations
            # Pick a random index to perturb
            idx = np.random.randint(N)
            
            # Propose a new height
            # Use a multiplicative perturbation to stay positive
            # Or additive with clipping
            # Let's try additive on normalized h, then renormalize
            # Or simpler: h_new = h * (1 + noise)
            
            # Let's use a simple random walk in log-space or multiplicative
            # h[idx] = h[idx] * (1 + 0.2 * (np.random.rand() - 0.5))
            # This can go to 0 or negative if not careful.
            
            # Better: Perturb one element, keep others same, renormalize to sum 1.
            # This ensures sum=1 and h>=0 if we clamp.
            
            h_new = h.copy()
            # Add random noise to one component
            h_new[idx] += np.random.normal(0, 0.05)
            # Clamp to non-negative
            h_new = np.maximum(h_new, 0)
            
            # Check if any mass was lost, if so, renormalize
            s = np.sum(h_new)
            if s < 1e-9:
                # If it collapsed, restart this candidate or skip
                continue
            h_new = h_new / s
            
            new_val = compute_value(h_new)
            
            if new_val > current_val:
                h = h_new
                current_val = new_val
                # Optionally increase step size slightly or keep constant
            else:
                # If stuck, maybe try a larger step or different index
                # Simple hill climbing doesn't backtrack, so just continue
                pass
        
        if current_val > best_val:
            best_val = current_val
            best_h = h
    
    return list(best_h)
