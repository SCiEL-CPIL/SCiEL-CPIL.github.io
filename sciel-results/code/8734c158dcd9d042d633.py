import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s
    
    # Initial guess: Uniform
    h = np.full(N, 0.5)
    
    # Precompute indices for linear shifts
    # O_k = sum_{i=k}^{N-1} h_i * (1 - h_{i-k})
    
    def compute_overlaps(h):
        overlaps = np.zeros(N)
        for k in range(N):
            # h[i] * (1 - h[i-k]) for i >= k
            # h[i-k] is h[0..N-1-k]
            # h[i] is h[k..N-1]
            prod = h[k:] * (1 - h[:N-k])
            overlaps[k] = np.sum(prod)
        return overlaps
    
    def soft_max(objs, beta):
        # Numerically stable log-sum-exp
        m = np.max(objs)
        if m > 1000:
            return m
        return m + np.log(np.sum(np.exp(objs - m))) / beta
        
    def objective(h, beta):
        overlaps = compute_overlaps(h)
        return soft_max(overlaps, beta)
        
    def grad_objective(h, beta):
        overlaps = compute_overlaps(h)
        # Gradient of soft max
        # d/dh sum_{i} h_i (1-h_{i-k})
        # Term for h_j:
        # In O_k, h_j appears if j >= k.
        # Term: h_j (1 - h_{j-k})
        # Deriv w.r.t h_j: (1 - h_{j-k})
        # Also h_j appears as h_{j+k} in O_k? No, O_k uses h_i and h_{i-k}.
        # Let's change variable.
        # O_k = sum_{i=k}^{N-1} h_i - h_i h_{i-k}
        # d O_k / d h_j:
        # If j >= k: term h_j (1 - h_{j-k}). Deriv: 1 - h_{j-k}.
        # If j + k < N: h_j is part of h_{i-k} where i = j+k.
        # Term in O_k: h_{j+k} (1 - h_j). Deriv w.r.t h_j: -h_{j+k}.
        # So d O_k / d h_j = (1 - h_{j-k}) * [j>=k] - h_{j+k} * [j+k < N]
        
        grads = np.zeros_like(h)
        weights = np.exp(overlaps - np.max(overlaps))
        weights /= np.sum(weights)
        
        for k in range(N):
            w = weights[k]
            if w < 1e-10: continue
            
            # Part 1: h_j (1 - h_{j-k}) for j >= k
            # d/dh_j = 1 - h_{j-k}
            # j ranges k to N-1
            # h_{j-k} ranges 0 to N-1-k
            g1 = 1 - h[:N-k]
            grads[k:] += w * g1
            
            # Part 2: - h_{j+k} (1 - h_j) for j+k < N
            # d/dh_j = -h_{j+k}
            # j ranges 0 to N-1-k
            # h_{j+k} ranges k to N-1
            g2 = -h[k:]
            grads[:N-k] += w * g2
            
        return grads / beta  # Gradient of (1/beta) log sum exp is (1/beta) * weighted sum of grads?
        # Actually d (1/beta) log sum exp(f_i) = (1/beta) * sum w_i df_i.
        # So yes, divide by beta.

    def project(h):
        # Project onto 0 <= h <= 1, sum h = 50
        # Use bisection on Lagrange multiplier
        lo, hi = 0.0, 1.0
        target = 50.0
        
        for _ in range(50):
            mid = (lo + hi) / 2
            proj = np.clip(h - mid, 0, 1)
            s = np.sum(proj)
            if s > target:
                lo = mid
            else:
                hi = mid
        proj = np.clip(h - (lo+hi)/2, 0, 1)
        # Correct sum exactly
        diff = target - np.sum(proj)
        proj += diff / N
        # Re-clip if necessary (should be small)
        return np.clip(proj, 0, 1)

    # FISTA
    h_curr = h.copy()
    h_prev = h.copy()
    t_k = 1.0
    beta = 10.0  # Start with moderate beta
    step_size = 1.0 / (2.0 * N)  # Lipschitz constant approx
    
    best_h = h_curr.copy()
    best_val = compute_overlaps(best_h).max()
    
    for it in range(500):
        if time.time() > deadline:
            break
            
        # Increase beta slowly
        if it % 50 == 0:
            beta *= 1.2
            step_size = 1.0 / (2.0 * N * beta) # Adjust step size?
            # Actually L for soft max is approx L_obj * beta.
            # L_obj is bounded by N.
            # Let's keep step_size conservative.
            
        grad = grad_objective(h_curr, beta)
        
        # Gradient step
        h_next = h_curr - step_size * grad
        
        # Projection
        h_next = project(h_next)
        
        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        h_curr = h_next + ((t_k - 1) / t_new) * (h_next - h_prev)
        h_curr = project(h_curr)
        
        h_prev = h_next
        t_k = t_new
        
        # Check best
        val = compute_overlaps(h_curr).max()
        if val < best_val:
            best_val = val
            best_h = h_curr.copy()
            
        if it % 100 == 0:
            print(f"Iter {it}, Beta {beta}, Max Overlap {best_val}")

    # Final polish: check uniform
    uniform_val = 25.0
    if best_val > uniform_val:
        return np.full(N, 0.5)
        
    return best_h
