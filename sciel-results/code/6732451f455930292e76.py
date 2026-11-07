import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    h = np.asarray(h)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    # y is [0, g[0], ..., g[m-1], 0]
    # Length of y is m + 2
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    # Points are np.linspace(-0.5, 0.5, m + 2)
    # Number of intervals is m + 1
    w = 1.0 / (m + 1)
    
    # l2 calculation
    # sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # i goes from 0 to m. y indices 0..m+1.
    # Terms: y[i], y[i+1]
    # Vectorized:
    yi = y[:-1] # length m+1
    yip1 = y[1:] # length m+1
    l2 = (w / 3.0) * np.sum(yi**2 + yi * yip1 + yip1**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    # Initial candidates
    x = np.linspace(-0.25, 0.25, N)
    
    seeds = []
    # 1. Uniform (Seed)
    seeds.append(np.ones(N))
    # 2. Gaussian
    seeds.append(np.exp(-10 * x**2))
    # 3. Cosine window
    seeds.append(0.5 + 0.5 * np.cos(2 * np.pi * x))
    # 4. Bimodal
    seeds.append(np.exp(-50 * (x - 0.1)**2) + np.exp(-50 * (x + 0.1)**2))
    # 5. Plateau
    seeds.append(np.where(np.abs(x) < 0.1, 1.0, 0.0))
    # 6. Random smooth
    rng = np.random.default_rng(42)
    r = rng.normal(0, 1, N)
    # Smooth it
    from scipy.ndimage import gaussian_filter1d
    r_smooth = gaussian_filter1d(r, sigma=2)
    r_smooth = np.abs(r_smooth)
    seeds.append(r_smooth)

    best_h = None
    best_val = 0.0

    for h0 in seeds:
        if time.time() > deadline:
            break
            
        # Normalize h0 so that max is 1? The value is scale invariant?
        # Let's check:
        # g scales with h^2.
        # l2 scales with h^4.
        # norm_1 scales with h^2.
        # norm_inf scales with h^2.
        # value = (C * h^4) / (C1 * h^2 * C2 * h^2) = Constant.
        # So the value is independent of the scale of h.
        # We can fix one parameter or normalize to avoid scaling issues during optimization.
        # However, minimize might wander to 0 or infinity.
        # It's better to optimize on the shape.
        # We can fix sum(h) = 1 or max(h) = 1.
        # Let's just let it optimize, but initialize with max(h)=1.
        if np.max(h0) > 0:
            h0 = h0 / np.max(h0)
        
        def neg_value(h_vec):
            # Ensure non-negative
            h_vec = np.maximum(h_vec, 0)
            return -compute_value(h_vec)

        try:
            res = minimize(neg_value, h0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            h_opt = np.maximum(res.x, 0)
            val = compute_value(h_opt)
            
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception:
            pass

    if best_h is None:
        return [1.0] * N
        
    return list(best_h)

# EVOLVE-BLOCK-START
# (The construct function is defined above, but the task requires the entry point to be named construct 
# and the block structure is usually expected. I will wrap it properly or just define it. 
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program had:
# N = 50
# # EVOLVE-BLOCK-START
# def construct():
#     return [1.0] * N
# # EVOLVE-BLOCK-END
# I will place the code inside the block or ensure construct is the global function.
# To be safe, I will define compute_value globally and construct in the block.
# EVOLVE-BLOCK-END

# Re-defining to match the expected structure if the harness looks for the block, 
# but standard python execution just needs the function. 
# The prompt says: "Keep the program's entry-point function name(s) ... EXACTLY as in the current program above"
# Current program:
# """Initial program..."""
# N = 50
# # EVOLVE-BLOCK-START
# def construct():
#     return [1.0] * N
# # EVOLVE-BLOCK-END

# So I will output the full file with the block.
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    # Initial candidates
    x = np.linspace(-0.25, 0.25, N)
    
    seeds = []
    # 1. Uniform (Seed)
    seeds.append(np.ones(N))
    # 2. Gaussian
    seeds.append(np.exp(-10 * x**2))
    # 3. Cosine window
    seeds.append(0.5 + 0.5 * np.cos(2 * np.pi * x))
    # 4. Bimodal
    seeds.append(np.exp(-50 * (x - 0.1)**2) + np.exp(-50 * (x + 0.1)**2))
    # 5. Plateau
    seeds.append(np.where(np.abs(x) < 0.1, 1.0, 0.0))
    # 6. Random smooth
    rng = np.random.default_rng(42)
    r = rng.normal(0, 1, N)
    # Smooth it
    from scipy.ndimage import gaussian_filter1d
    r_smooth = gaussian_filter1d(r, sigma=2)
    r_smooth = np.abs(r_smooth)
    seeds.append(r_smooth)

    best_h = None
    best_val = 0.0

    for h0 in seeds:
        if time.time() > deadline:
            break
            
        # Normalize h0 so that max is 1
        if np.max(h0) > 0:
            h0 = h0 / np.max(h0)
        
        def neg_value(h_vec):
            # Ensure non-negative
            h_vec = np.maximum(h_vec, 0)
            return -compute_value(h_vec)

        try:
            res = minimize(neg_value, h0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            h_opt = np.maximum(res.x, 0)
            val = compute_value(h_opt)
            
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception:
            pass

    if best_h is None:
        return [1.0] * N
        
    return list(best_h)
# EVOLVE-BLOCK-END
