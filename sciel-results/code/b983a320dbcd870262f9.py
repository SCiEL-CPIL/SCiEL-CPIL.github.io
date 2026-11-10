"""Find optimal 24-coefficient ±1 polynomial minimizing sup norm."""
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 24
    sqrt_n = np.sqrt(n)
    
    # Precompute evaluation points on unit circle
    N_pts = 4096
    theta = np.linspace(0, 2*np.pi, N_pts, endpoint=False)
    z = np.exp(1j * theta)
    
    # Z[j, k] = z_j^k
    Z = np.ones((N_pts, n), dtype=complex)
    for k in range(1, n):
        Z[:, k] = Z[:, k-1] * z
    
    def sup_norm(c):
        """Compute max |P(z)| for coefficient vector c."""
        vals = Z @ c
        return np.max(np.abs(vals))
    
    # Start with Golay-Rudin-Shapiro
    p, q = np.array([1]), np.array([1])
    while len(p) < n:
        p = np.concatenate([p, q])
        q = np.concatenate([p, -q])
    c_best = p[:n].copy()
    best_val = sup_norm(c_best)
    
    # Local search: try single flips, then pairs
    def local_search(c, max_iters=10000):
        c = c.copy()
        cur_val = sup_norm(c)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try single flips
            for k in range(n):
                c[k] = -c[k]
                new_val = sup_norm(c)
                if new_val < cur_val - 1e-12:
                    cur_val = new_val
                    improved = True
                    break
                else:
                    c[k] = -c[k]
            if improved:
                continue
            # Try pair flips
            pair_improved = False
            for k1 in range(n):
                if pair_improved:
                    break
                for k2 in range(k1+1, n):
                    c[k1] = -c[k1]
                    c[k2] = -c[k2]
                    new_val = sup_norm(c)
                    if new_val < cur_val - 1e-12:
                        cur_val = new_val
                        pair_improved = True
                        break
                    else:
                        c[k1] = -c[k1]
                        c[k2] = -c[k2]
            if not pair_improved:
                break
        return c, cur_val
    
    c_best, best_val = local_search(c_best)
    
    # Random restarts with local search
    rng = np.random.default_rng(42)
    n_restarts = 0
    for restart in range(500):
        if time.time() > deadline - 5.0:
            break
        
        c = rng.choice([-1, 1], size=n)
        c, val = local_search(c, max_iters=5000)
        if val < best_val - 1e-12:
            best_val = val
            c_best = c
        n_restarts += 1
    
    # More aggressive local search from best
    if time.time() < deadline - 10.0:
        c_best, best_val = local_search(c_best, max_iters=50000)
    
    return c_best.tolist()
