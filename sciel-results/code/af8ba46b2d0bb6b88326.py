import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_value(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def compute_corr_max(h):
    """Compute max correlation value (without normalization)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c))

def project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if abs(s - target) > 1e-10:
        diff = target - s
        free = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def binary_local_search(h_bin, max_iters=5000):
    """Local search on binary array: swap pairs to reduce max correlation."""
    h_bin = h_bin.copy()
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    best_val = compute_corr_max(h_bin)
    
    for it in range(max_iters):
        if remaining() < 5:
            break
        improved = False
        # Try random swaps
        np.random.shuffle(ones)
        np.random.shuffle(zeros)
        for i in range(min(50, len(ones))):
            for j in range(min(50, len(zeros))):
                oi = ones[i]
                oz = zeros[j]
                if oi == oz:
                    continue
                # Swap
                h_bin[oi], h_bin[oz] = h_bin[oz], h_bin[oi]
                new_val = compute_corr_max(h_bin)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    # Update lists
                    ones[ones == oi] = oz
                    zeros[zeros == oz] = oi
                    improved = True
                else:
                    # Revert
                    h_bin[oi], h_bin[oz] = h_bin[oz], h_bin[oi]
        if not improved:
            break
    return h_bin, best_val

def construct():
    i = np.arange(N)
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)  # 0.5
    
    # === Phase 1: Binary discrete search ===
    # Start with alternating pattern
    h_bin_init = np.zeros(N)
    h_bin_init[::2] = 1.0  # 50 ones, 50 zeros
    h_bin_init = h_bin_init.copy()
    
    # Also try other initial patterns
    np.random.seed(42)
    binary_starts = [h_bin_init.copy()]
    
    # Random binary starts with 50 ones
    for _ in range(5):
        h_rand = np.zeros(N)
        ones_idx = np.random.choice(N, 50, replace=False)
        h_rand[ones_idx] = 1.0
        binary_starts.append(h_rand)
    
    # Block patterns
    for block_size in [1, 2, 5, 10]:
        h_block = np.zeros(N)
        for start in range(0, N, 2 * block_size):
            h_block[start:start+block_size] = 1.0
        binary_starts.append(h_block)
    
    best_binary = None
    best_binary_val = float('inf')
    
    for h0 in binary_starts:
        if remaining() < 10:
            break
        h_bin, val = binary_local_search(h0, max_iters=2000)
        if val < best_binary_val:
            best_binary_val = val
            best_binary = h_bin.copy()
    
    if best_binary is not None:
        h_from_binary = best_binary.copy()
        val_binary = compute_value(h_from_binary)
        if val_binary < best_val:
            best_val = val_binary
            best_h = h_from_binary
    
    # === Phase 2: Continuous polish from best binary solution ===
    if best_binary is not None:
        x0 = best_binary - 0.5
    else:
        x0 = np.zeros(N)
    
    def obj_x(x):
        h = x + 0.5
        c = np.correlate(h, 1.0 - h, mode="full")
        return float(np.max(c))
    
    # Also add the original continuous starts
    starts = [x0.copy()]
    for k in range(1, 6):
        for A in [0.1, 0.2, 0.3, 0.4, 0.5]:
            starts.append(A * np.sin(2 * np.pi * k * i / N))
            starts.append(A * np.cos(2 * np.pi * k * i / N))
    
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.randn(N)
        x_rand -= np.mean(x_rand)
        x_rand /= max(np.max(np.abs(x_rand)), 1e-10) * 2
        starts.append(x_rand)
    
    bounds = [(-0.5, 0.5)] * N
    
    for x0 in starts:
        if remaining() < 5:
            break
        
        x0 = np.clip(x0, -0.5, 0.5)
        s = np.sum(x0)
        if abs(s) > 1e-10:
            x0 = x0 - s / N
            x0 = np.clip(x0, -0.5, 0.5)
            for _ in range(10):
                s = np.sum(x0)
                if abs(s) < 1e-10:
                    break
                x0 = x0 - s / N
                x0 = np.clip(x0, -0.5, 0.5)
        
        try:
            res = minimize(
                obj_x,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = res.x
            h_opt = x_opt + 0.5
            h_opt = project(h_opt)
            val = compute_value(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # === Phase 3: Final polish with tighter tolerance ===
    if remaining() > 10:
        x0 = best_h - 0.5
        x0 = np.clip(x0, -0.5, 0.5)
        s = np.sum(x0)
        x0 = x0 - s / N
        x0 = np.clip(x0, -0.5, 0.5)
        
        try:
            res = minimize(
                obj_x,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_opt = res.x
            h_opt = x_opt + 0.5
            h_opt = project(h_opt)
            val = compute_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h
