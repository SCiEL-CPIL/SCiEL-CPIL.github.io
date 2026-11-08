import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution (full)
    conv = np.convolve(h, h, mode='full')
    sup_norm = np.max(conv)
    if sup_norm == 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def normalize_l1(h):
    s = np.sum(h)
    if s == 0:
        return h
    return h / s

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 140s to leave 10s safety margin
    best_val = 0.0
    best_h = np.ones(N)
    
    # Start from a few initial shapes
    initial_shapes = [
        np.ones(N),
        np.exp(-np.linspace(0, 1, N)**2 * 2),
        np.abs(np.sin(np.linspace(0, np.pi, N))),
        np.concatenate([np.ones(N//2), np.zeros(N - N//2)]),
        np.array([1.0 if i % 2 == 0 else 0.1 for i in range(N)]),
    ]
    
    rng = np.random.RandomState(42)
    
    # Evaluate initial shapes
    for h0 in initial_shapes:
        h0 = np.maximum(h0, 0)
        h0n = normalize_l1(h0)
        val = compute_ratio(h0n)
        if val > best_val:
            best_val = val
            best_h = h0n.copy()
    
    # Simulated annealing
    h = best_h.copy()
    T = 0.1
    T_min = 1e-8
    alpha = 0.9995
    current_val = best_val
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Random perturbation
        mask = rng.random(N) < 0.3
        noise = rng.normal(0, 0.1, N) * mask
        h_new = np.maximum(h + noise, 0)
        
        if np.sum(h_new) == 0:
            h_new = h.copy()
            h_new[rng.randint(N)] += 1e-10
        
        h_newn = normalize_l1(h_new)
        new_val = compute_ratio(h_newn)
        
        # Acceptance criterion
        if new_val >= current_val or rng.random() < np.exp((new_val - current_val) / max(T, 1e-12)):
            h = h_newn
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_h = h.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Return best found, scaled small in case no normalization
    # But primarily the shape matters if normalized
    return list(best_h)
# EVOLVE-BLOCK-END
