import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
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

def _generate_sidon_set(size, max_retries=100):
    """
    Greedily construct a Sidon-like set (low additive energy) of given size.
    We want indices such that differences are mostly unique.
    """
    # Try to pick indices with large gaps
    # Simple strategy: pick indices spaced by ~sqrt(N)
    # Or use a random greedy approach:
    # Start with empty set, add index if it doesn't create too many duplicate differences.
    
    best_set = None
    best_energy = float('inf')
    
    for _ in range(max_retries):
        S = set()
        # Random start
        S.add(np.random.randint(0, N))
        
        while len(S) < size:
            # Try to add a random new index
            added = False
            for _ in range(100):
                idx = np.random.randint(0, N)
                if idx in S:
                    continue
                new_S = S | {idx}
                # Calculate additive energy (number of duplicate differences)
                # For small sets, this is fast
                diffs = []
                for i in new_S:
                    for j in new_S:
                        if i != j:
                            diffs.append(i - j)
                # Count frequencies
                from collections import Counter
                counts = Counter(diffs)
                # Energy is sum of (count-1) for counts > 1? Or just max count?
                # We want to minimize the max count of any difference (excluding 0)
                max_count = max(counts.values()) if counts else 0
                if max_count < 2: # Perfect Sidon (distinct differences)
                    S = new_S
                    added = True
                    break
                elif max_count < 3: # Allow some collisions
                    # Prefer if we reduce the max count
                    if best_set is None or max_count < 2:
                         # This logic is flawed for "best" tracking in loop
                         pass
            
            if not added:
                # Try a different random start or break
                break
        
        if len(S) == size:
            # Calculate the "peak" off-0 correlation
            h_test = np.zeros(N)
            for i in S:
                h_test[i] = 1.0
            conv = np.convolve(h_test, h_test, mode='full')
            # Peak is at center (index N-1)
            peak = conv[N-1]
            # Max off-peak
            off_peak = max(np.max(conv[:N-1]), np.max(conv[N:]))
            # We want to minimize off_peak
            if off_peak < best_energy:
                best_energy = off_peak
                best_set = S
                
    if best_set is None:
        # Fallback to evenly spaced
        best_set = set(np.linspace(0, N-1, size, dtype=int))
        
    return sorted(best_set)

def _optimize_amplitudes(support, h_init=None, time_limit=10.0):
    """
    Optimize the amplitudes of h on the given support.
    """
    if not support:
        return np.zeros(N), 0.0
        
    h0 = np.zeros(N)
    for i in support:
        h0[i] = 1.0
        
    if h_init is not None:
        h0 = h_init.copy()
        
    def objective(amps):
        h = np.zeros(N)
        for i, a in zip(support, amps):
            h[i] = a
        return -_compute_ratio(h)
        
    def jac(amps):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(amps)
        f0 = objective(amps)
        for i in range(len(amps)):
            amps_plus = amps.copy()
            amps_plus[i] += eps
            f_plus = objective(amps_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    bounds = [(0.0, 10.0)] * len(support)
    
    try:
        res = minimize(objective, h0[support], method='L-BFGS-B', bounds=bounds, 
                       jac=jac, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        h_opt = np.zeros(N)
        for i, a in zip(support, res.x):
            h_opt[i] = a
            
        return h_opt, -res.fun
    except Exception:
        return h0, _compute_ratio(h0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Try different support sizes
    # Max Sidon size for N=50 is ~7
    # Try sizes 1 to 10
    np.random.seed(42)
    
    for size in range(1, 11):
        if time.time() > deadline - 10.0:
            break
            
        # Generate a few candidate supports
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
                
            support = _generate_sidon_set(size)
            
            # Optimize amplitudes
            h_opt, ratio = _optimize_amplitudes(support)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
                
    # Also try a simple uniform vector as baseline
    h_uniform = np.ones(N) / N
    ratio_uniform = _compute_ratio(h_uniform)
    if ratio_uniform > best_ratio:
        best_ratio = ratio_uniform
        best_h = h_uniform.copy()
        
    # Coordinate descent polish on the best found
    # Use the existing coordinate descent logic but simplified
    def _optimize_single_coordinate(h, i):
        A = np.sum(h**2) - h[i]**2
        B = np.sum(h) - h[i]
        
        if B < 1e-12:
            return h[i]
            
        h_fixed = h.copy()
        h_fixed[i] = 0.0
        
        C = np.convolve(h_fixed, h_fixed, mode='full')
        
        D = np.zeros(2*N - 1)
        start_idx = i
        end_idx = i + N
        D[start_idx:end_idx] = 2.0 * h_fixed
        
        E = np.zeros(2*N - 1)
        if 2*i < 2*N - 1:
            E[2*i] = 1.0
            
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
            
        # Golden Section Search
        gr = (np.sqrt(5) - 1) / 2
        a = 0.0
        b = max(1.0, 2.0 * np.sqrt(A + 1.0) + B)
        
        fa = ratio_func(a)
        fb = ratio_func(b)
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = ratio_func(c)
        fd = ratio_func(d)
        
        for _ in range(50):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = ratio_func(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = ratio_func(c)
                
            if abs(b - a) < 1e-9:
                break
                
        x_opt = (a + b) / 2.0
        return x_opt

    coords = np.arange(N)
    for _ in range(10): # Reduced passes to save time
        if time.time() > deadline - 2.0:
            break
        np.random.shuffle(coords)
        for i in coords:
            new_val = _optimize_single_coordinate(best_h, i)
            best_h[i] = new_val
            
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
        
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(_compute_ratio(h))
