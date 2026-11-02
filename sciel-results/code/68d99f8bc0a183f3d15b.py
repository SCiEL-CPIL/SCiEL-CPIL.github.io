import numpy as np
import time
import sys

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _optimize_single_coordinate(h, i):
    """
    Given h, find the optimal value for h[i] that maximizes the ratio,
    keeping all other h[j] (j != i) fixed.
    
    Let x = h[i].
    Let A = sum_{j != i} h[j]^2
    Let B = sum_{j != i} h[j]
    
    L2_sq = A + x^2
    L1 = B + x
    
    Convolution:
    (h*h)_k = sum_{j} h_j h_{k-j}
    Terms involving h_i (which is x):
    If we write h = h_fixed + x * e_i, then
    h*h = h_fixed*h_fixed + x*(h_fixed*e_i + e_i*h_fixed) + x^2*(e_i*e_i)
    
    Let C_k = (h_fixed*h_fixed)_k
    Let D_k = (h_fixed*e_i + e_i*h_fixed)_k
    Let E_k = (e_i*e_i)_k
    
    Note: e_i is a unit vector at index i.
    (h_fixed * e_i)_k = h_fixed[k-i] if 0 <= k-i < N else 0
    (e_i * h_fixed)_k = h_fixed[k-i] if 0 <= k-i < N else 0
    So D_k = 2 * h_fixed[k-i] (if valid index)
    
    (e_i * e_i)_k is 1 if k-i == i => k=2i, else 0.
    So E_k is 1 at k=2i, 0 elsewhere.
    
    Conv_k(x) = C_k + x * D_k + x^2 * E_k
    
    We need to maximize R(x) = (A + x^2) / ((B + x) * max_k(C_k + x*D_k + x^2*E_k))
    
    Since x >= 0, and B > 0 (assuming we don't start with all zeros),
    we can scan x over a reasonable range or use a root finding method on the derivative.
    However, the function is rational and potentially non-convex.
    
    A simpler, robust approach for coordinate descent in this context:
    Since N=50 is small, we can discretize x or use a golden section search
    over a bounded interval for x.
    
    What is a good bound for x?
    If x is very large, L2 ~ x^2, L1 ~ x, Conv ~ x^2 (at peak 2i).
    R ~ x^2 / (x * x^2) = 1/x -> 0.
    If x is 0, R = A / (B * max(C)).
    
    The maximum likely occurs at a moderate x.
    Let's estimate a bound.
    If we set x such that L2 is dominated by x, i.e., x^2 ~ A, x ~ sqrt(A).
    Let's search x in [0, 2 * sqrt(A + 1)] to be safe.
    """
    h = np.asarray(h, dtype=float)
    if h[i] == 0 and np.sum(h) == 0:
        return 0.0
        
    A = np.sum(h**2) - h[i]**2
    B = np.sum(h) - h[i]
    
    # Precompute C, D, E vectors for the convolution part
    # C = h_fixed * h_fixed
    h_fixed = h.copy()
    h_fixed[i] = 0.0
    
    C = np.convolve(h_fixed, h_fixed, mode='full')
    
    # D_k = 2 * h_fixed[k-i]
    D = np.zeros(2*N - 1)
    # Indices k such that 0 <= k-i < N => i <= k < i+N
    start_idx = i
    end_idx = i + N
    # h_fixed indices are 0..N-1.
    # D[k] = 2 * h_fixed[k-i]
    # k-i ranges from 0 to N-1.
    D[start_idx:end_idx] = 2.0 * h_fixed
    
    # E_k = 1 at k=2i
    E = np.zeros(2*N - 1)
    if 2*i < 2*N - 1:
        E[2*i] = 1.0
        
    # We want to maximize f(x) = (A + x^2) / ((B + x) * max(C + x*D + x^2*E))
    
    # Golden Section Search
    # Range: [0, x_max]
    # Estimate x_max:
    # If B is small, x can be large.
    # Let's try x_max = max(1.0, 2.0 * np.sqrt(A + 1.0) + B)
    x_max = max(1.0, 2.0 * np.sqrt(A + 1.0) + B)
    
    # If B is 0 (all other zeros), the problem is simpler:
    # h has only h[i] = x.
    # L2 = x^2, L1 = x, Conv peak = x^2.
    # Ratio = x^2 / (x * x^2) = 1/x. Maximized at x -> 0? No, ratio -> infinity?
    # Wait. If h is a single spike, conv is a single spike.
    # L2 = x^2, L1 = x, SupConv = x^2.
    # Ratio = x^2 / (x * x^2) = 1/x.
    # To maximize, we want x small? But h must be non-negative.
    # If x -> 0, ratio -> infinity?
    # Let's check the definition.
    # If h = [0, 1, 0], L2=1, L1=1, Conv=[0,1,0], Sup=1. Ratio = 1.
    # If h = [0, 0.1, 0], L2=0.01, L1=0.1, Conv=[0,0.01,0], Sup=0.01. Ratio = 0.01/(0.1*0.01) = 1.
    # It is scale invariant.
    # So if B=0, the value is 1.0.
    # We should avoid B=0 in the search unless we handle it.
    
    if B < 1e-12:
        # If only one element is non-zero, ratio is 1.
        # If we add a second element, we can potentially exceed 1.
        # For this coordinate step, if we are the only one, any x gives 1.
        # Let's return current x or a small value to allow other coords to act.
        return h[i] if h[i] > 0 else 1e-6

    def ratio_func(x):
        if x < 0:
            x = 0.0
        l2_val = A + x*x
        l1_val = B + x
        if l1_val < 1e-15:
            return 0.0
        conv_vals = C + x * D + x*x * E
        sup_conv = np.max(conv_vals)
        if sup_conv < 1e-15:
            return 0.0
        return l2_val / (l1_val * sup_conv)
        
    # Golden Section Search for maximum
    gr = (np.sqrt(5) - 1) / 2  # ~0.618
    a = 0.0
    b = x_max
    
    # Check endpoints
    fa = ratio_func(a)
    fb = ratio_func(b)
    
    # If the function is increasing over the whole range, the max is at b.
    # But we know it goes to 0 at infinity.
    # Let's ensure b is large enough.
    # If fb > fa, maybe we need a larger b?
    # Actually, for x large, ratio ~ 1/x. So it decreases.
    # There must be a local max.
    
    # Standard Golden Section
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = ratio_func(c)
    fd = ratio_func(d)
    
    # We want to maximize
    for _ in range(100):  # 100 iterations is plenty for precision
        if fc < fd:
            # Max is in [c, b]
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = ratio_func(d)
        else:
            # Max is in [a, d]
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = ratio_func(c)
            
        if abs(b - a) < 1e-10:
            break
            
    x_opt = (a + b) / 2.0
    return x_opt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Initialization: Try a few different sparse starts
    # 1. Uniform
    h_uniform = np.ones(N) / N
    # 2. Sparse random
    np.random.seed(42)
    
    initializations = []
    
    # Add uniform
    initializations.append(h_uniform.copy())
    
    # Add a few random sparse ones
    for k in [3, 5, 10, 20]:
        idxs = np.random.choice(N, k, replace=False)
        h_sparse = np.zeros(N)
        h_sparse[idxs] = 1.0 / k
        initializations.append(h_sparse)
        
    # Add a "Sidon-like" set
    sidon_S = [0, 1, 4, 10, 18, 23, 30]
    h_sidon = np.zeros(N)
    for idx in sidon_S:
        h_sidon[idx] = 1.0 / len(sidon_S)
    initializations.append(h_sidon)
    
    # Coordinate Descent
    # We will iterate through the coordinates multiple times.
    # To speed up, we can randomly permute the order of coordinates in each pass.
    
    num_passes = 2000  # Max passes, limited by time
    coords = np.arange(N)
    
    current_h = None
    
    for init_idx, h_init in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
            
        h = h_init.copy()
        
        # Run coordinate descent on this start
        # We'll do a fixed number of passes or until time runs out
        # Let's allocate time per initialization if possible, or just run until deadline.
        # Since we have ~5 inits, we can do ~400 passes each if we split time evenly.
        # But better to just run as many as possible and keep the global best.
        
        # Actually, let's just run one long loop, and if we find a new best, 
        # we can restart or continue. 
        # A simpler strategy: Run coordinate descent on each init for a fixed number of steps, 
        # then pick the best.
        
        steps_per_init = 500
        for step in range(steps_per_init):
            if time.time() > deadline - 2.0:
                break
                
            # Randomize order to avoid cycles
            np.random.shuffle(coords)
            
            improved = False
            for i in coords:
                if time.time() > deadline - 1.0:
                    break
                    
                old_val = h[i]
                new_val = _optimize_single_coordinate(h, i)
                
                # Ensure non-negative
                if new_val < 0:
                    new_val = 0.0
                    
                h[i] = new_val
                
                if abs(new_val - old_val) > 1e-12:
                    improved = True
                    
            # Compute ratio
            ratio = _compute_ratio(h)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h.copy()
                
            # Convergence check: if no coordinate changed significantly, break
            if not improved:
                break
                
    # Final verification
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
        best_h = best_h.copy()
        
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(_compute_ratio(h))
