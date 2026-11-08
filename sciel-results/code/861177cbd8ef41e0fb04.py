import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Local optimization over weight ratios for 2-point and 3-point clusters.
    """
    t0 = time.time()
    deadline = 100.0  # Safety margin well under 120s
    
    def compute_score(h):
        """Compute the ratio: L2^2 / (L1 * sup_norm_autoconv)."""
        h = np.array(h, dtype=float)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution (linear)
        conv = np.convolve(h, h, mode='full')
        conv_inf = np.max(conv)
        if conv_inf < 1e-15:
            return 0.0
        return l2_sq / (l1 * conv_inf)
    
    def local_optimize(h_init, max_iter=200, tol=1e-10):
        """Nelder-Mead style local optimization on the nonnegative simplex."""
        h = np.array(h_init, dtype=float)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return h, 0.0
        h = h / l1  # normalize to L1=1
        
        # Use log-barrier for positivity: work in log-space
        eps = 1e-12
        h_safe = np.maximum(h, eps)
        h_safe = h_safe / np.sum(h_safe)
        
        # Coordinate descent with line search
        best_h = h_safe.copy()
        best_s = compute_score(best_h)
        
        for it in range(max_iter):
            if time.time() - t0 > deadline:
                break
            improved = False
            for i in range(N):
                if best_h[i] < 1e-8:
                    continue
                # Try increasing h[i] by taking from another index
                for j in range(N):
                    if j == i or best_h[j] < 1e-8:
                        continue
                    # Try moving delta from j to i
                    for delta in [best_h[j]*0.1, best_h[j]*0.3, best_h[j]*0.5, best_h[j]*0.7, best_h[j]*0.9]:
                        h_trial = best_h.copy()
                        h_trial[i] += delta
                        h_trial[j] -= delta
                        if h_trial[j] < 0:
                            continue
                        h_trial = h_trial / np.sum(h_trial)
                        s_trial = compute_score(h_trial)
                        if s_trial > best_s + 1e-12:
                            best_h = h_trial.copy()
                            best_s = s_trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        return best_h, best_s
    
    best_score = 0.0
    best_h = None
    
    # Generate diverse initial configurations
    candidates = []
    
    # 2-point clusters with various weight ratios
    for r in np.linspace(0.05, 10.0, 150):
        if time.time() - t0 > deadline:
            break
        a = r
        b = 1.0
        total = a + b
        h = np.zeros(N)
        h[0] = a / total
        h[1] = b / total
        candidates.append(h)
    
    # 3-point clusters
    for r1 in np.linspace(0.05, 5.0, 40):
        if time.time() - t0 > deadline:
            break
        for r2 in np.linspace(0.05, 5.0, 40):
            if time.time() - t0 > deadline:
                break
            a = r1
            b = 1.0
            c = r2
            total = a + b + c
            h = np.zeros(N)
            h[0] = a / total
            h[1] = b / total
            h[2] = c / total
            candidates.append(h)
    
    # 4-point clusters
    for r1 in np.linspace(0.05, 3.0, 20):
        if time.time() - t0 > deadline:
            break
        for r2 in np.linspace(0.05, 3.0, 20):
            if time.time() - t0 > deadline:
                break
            for r3 in np.linspace(0.05, 3.0, 20):
                if time.time() - t0 > deadline:
                    break
                a = r1
                b = 1.0
                c = 1.0
                d = r3
                total = a + b + c + d
                h = np.zeros(N)
                h[0] = a / total
                h[1] = b / total
                h[2] = c / total
                h[3] = d / total
                candidates.append(h)
    
    # 5-point clusters (more spread out)
    for r1 in np.linspace(0.1, 2.0, 15):
        if time.time() - t0 > deadline:
            break
        for r2 in np.linspace(0.1, 2.0, 15):
            if time.time() - t0 > deadline:
                break
                a = r1
                b = 1.0
                c = 1.0
                d = 1.0
                e = r2
                total = a + b + c + d + e
                h = np.zeros(N)
                h[0] = a / total
                h[1] = b / total
                h[2] = c / total
                h[3] = d / total
                h[4] = e / total
                candidates.append(h)
    
    # Evaluate all candidates and keep top ones for local optimization
    scored_candidates = []
    for h in candidates:
        s = compute_score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local optimize the top candidates
    top_k = min(20, len(scored_candidates))
    for s_init, h_init in scored_candidates[:top_k]:
        if time.time() - t0 > deadline:
            break
        h_opt, s_opt = local_optimize(h_init, max_iter=100)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    if best_h is None:
        # Fallback: known good 2-point 2:1
        best_h = [0.0] * N
        best_h[0] = 2.0 / 3.0
        best_h[1] = 1.0 / 3.0
    else:
        best_h = best_h.tolist()
    
    return best_h
# EVOLVE-BLOCK-END
