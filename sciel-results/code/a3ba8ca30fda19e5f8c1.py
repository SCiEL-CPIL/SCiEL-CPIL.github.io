# EVOLVE-BLOCK-START
import numpy as np


def _exact_1d_union_integral(base_pos, N):
    """
    Compute the exact integral of the union measure over c in [0,1] for N intervals.
    Interval i: [base_pos[i] + c*i/N, base_pos[i] + 1/N + c*i/N]
    
    The union measure is piecewise linear in c. Breakpoints occur when:
    - Left endpoint of i = left endpoint of j: base_pos[i] + c*i/N = base_pos[j] + c*j/N
      => c = (base_pos[j] - base_pos[i]) / ((i-j)/N)
    - Left endpoint of i = right endpoint of j: base_pos[i] + c*i/N = base_pos[j] + 1/N + c*j/N
      => c = (base_pos[j] + 1/N - base_pos[i]) / ((i-j)/N)
    - Right endpoint of i = left endpoint of j: base_pos[i] + 1/N + c*i/N = base_pos[j] + c*j/N
      => c = (base_pos[j] - base_pos[i] - 1/N) / ((i-j)/N)
    - Right endpoint of i = right endpoint of j: base_pos[i] + 1/N + c*i/N = base_pos[j] + 1/N + c*j/N
      => c = (base_pos[j] - base_pos[i]) / ((i-j)/N)
    
    We collect all breakpoints in [0,1], sort them, and compute the union measure
    at the midpoint of each subinterval (where the set of merged intervals is constant).
    """
    n = len(base_pos)
    breakpoints = [0.0, 1.0]
    
    for i in range(n):
        for j in range(i + 1, n):
            di = i - j
            if di == 0:
                continue
            # Four types of coincidences
            # L_i = L_j: c = (base_pos[j] - base_pos[i]) / (di/N) = N*(base_pos[j]-base_pos[i])/di
            c1 = N * (base_pos[j] - base_pos[i]) / di
            # L_i = R_j: c = N*(base_pos[j] + 1/N - base_pos[i]) / di
            c2 = N * (base_pos[j] + 1.0/N - base_pos[i]) / di
            # R_i = L_j: c = N*(base_pos[j] - base_pos[i] - 1/N) / di
            c3 = N * (base_pos[j] - base_pos[i] - 1.0/N) / di
            # R_i = R_j: c = N*(base_pos[j] - base_pos[i]) / di (same as c1)
            
            for c_val in [c1, c2, c3]:
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)
    
    breakpoints = sorted(set(breakpoints))
    
    # Compute union measure at midpoint of each subinterval
    total_integral = 0.0
    for k in range(len(breakpoints) - 1):
        c_lo = breakpoints[k]
        c_hi = breakpoints[k + 1]
        c_mid = 0.5 * (c_lo + c_hi)
        
        # Compute all intervals at c_mid
        lefts = base_pos + c_mid * np.arange(n) / N
        rights = lefts + 1.0 / N
        
        # Merge intervals
        order = np.argsort(lefts)
        l_sorted = lefts[order]
        r_sorted = rights[order]
        
        merged_len = 0.0
        cl = l_sorted[0]
        cr = r_sorted[0]
        for m in range(1, n):
            if l_sorted[m] <= cr:
                cr = max(cr, r_sorted[m])
            else:
                merged_len += cr - cl
                cl = l_sorted[m]
                cr = r_sorted[m]
        merged_len += cr - cl
        
        total_integral += merged_len * (c_hi - c_lo)
    
    return total_integral


def _optimize_1d_exact(N, n_iter=1000, lr_init=0.01):
    """
    Optimize 1-D base positions using the exact integral and numerical gradients.
    """
    # Initial guess: folding pattern
    base = np.zeros(N)
    for i in range(N):
        base[i] = 1.0 / (2 * N) - i / N + (1.0 / (2 * N)) * (i % 2)
    base -= base.min()
    
    pos = base.copy()
    best_pos = pos.copy()
    best_vol = _exact_1d_union_integral(pos, N)
    
    lr = lr_init
    for it in range(n_iter):
        grad = np.zeros(N)
        eps = 1e-6
        for i in range(N):
            pos[i] += eps
            vp = _exact_1d_union_integral(pos, N)
            pos[i] -= 2 * eps
            vm = _exact_1d_union_integral(pos, N)
            pos[i] += eps
            grad[i] = (vp - vm) / (2 * eps)
        
        pos -= lr * grad
        pos -= pos.min()
        
        vol = _exact_1d_union_integral(pos, N)
        if vol < best_vol:
            best_vol = vol
            best_pos = pos.copy()
        
        lr *= 0.995
    
    return best_pos, best_vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with N = 8.
    
    The union volume factors as V = A_x * A_y. We minimize each 1-D union
    measure independently using a folding construction refined by gradient
    descent with exact integral evaluation.
    """
    N = 8
    
    x_1d, vol_x = _optimize_1d_exact(N, n_iter=500, lr_init=0.005)
    y_1d, vol_y = _optimize_1d_exact(N, n_iter=500, lr_init=0.005)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_1d[i_flat]
    base_y = y_1d[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
