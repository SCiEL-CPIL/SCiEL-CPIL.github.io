"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    def compute_overlap_penalty(xs, ys, rs):
        """Vectorized overlap penalty computation."""
        n = len(rs)
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        return penalty

    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        # Check bounds
        if np.any(rs < 1e-9):
            return
        if np.any(xs - rs < -1e-6) or np.any(xs + rs > w + 1e-6):
            return
        if np.any(ys - rs < -1e-6) or np.any(ys + rs > h + 1e-6):
            return
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < (min_dist - 1e-7) ** 2:
                    return
        val = np.sum(rs)
        if val > best_value:
            best_value = val
            best_result = x.copy()

    def sa_joint_optimize(x0, w0, max_iters=3000):
        """Simulated annealing jointly optimizing w, positions, and radii.
        x0: array of shape (3*N,) for [x1,y1,r1, x2,y2,r2, ...]
        w0: initial width (h = 2 - w)
        Returns (x_final, w_final)
        """
        xs = x0[0::3].copy()
        ys = x0[1::3].copy()
        rs = x0[2::3].copy()
        w = w0
        h = 2.0 - w
        
        T = 0.08
        cooling = 0.9995
        rng = np.random.default_rng(42)
        
        def cost(xs, ys, rs, w, h):
            # Boundary violations
            bpen = 0.0
            for i in range(N):
                if xs[i] < rs[i]:
                    bpen += (rs[i] - xs[i]) ** 2
                if xs[i] > w - rs[i]:
                    bpen += (xs[i] - (w - rs[i])) ** 2
                if ys[i] < rs[i]:
                    bpen += (rs[i] - ys[i]) ** 2
                if ys[i] > h - rs[i]:
                    bpen += (ys[i] - (h - rs[i])) ** 2
            overlap = compute_overlap_penalty(xs, ys, rs)
            return bpen + overlap - np.sum(rs)
        
        current_cost = cost(xs, ys, rs, w, h)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = rng.random()
            
            if move_type < 0.15:
                # Perturb aspect ratio
                old_w = w
                w += rng.normal(0, T * 0.5)
                w = np.clip(w, 0.4, 1.6)
                h = 2.0 - w
                # Re-clamp circles to new bounds
                for i in range(N):
                    rs[i] = np.clip(rs[i], 1e-4, min(w, h) / 2.0)
                    xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                    ys[i] = np.clip(ys[i], rs[i], h - rs[i])
                new_cost = cost(xs, ys, rs, w, h)
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    w = old_w
                    h = 2.0 - w
                    # Restore clamped values (re-derive from old bounds)
                    for i in range(N):
                        rs[i] = np.clip(rs[i], 1e-4, min(w, h) / 2.0)
                        xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                        ys[i] = np.clip(ys[i], rs[i], h - rs[i])
                    current_cost = cost(xs, ys, rs, w, h)
                    
            else:
                # Perturb a random circle
                idx = rng.integers(0, N)
                old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
                scale = T * 0.3
                xs[idx] += rng.normal(0, scale)
                ys[idx] += rng.normal(0, scale)
                rs[idx] += rng.normal(0, scale * 0.5)
                
                # Clamp to bounds
                rs[idx] = np.clip(rs[idx], 1e-4, min(w, h) / 2.0)
                xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
                ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
                
                new_cost = cost(xs, ys, rs, w, h)
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
                
            T *= cooling
        
        x_final = np.zeros(3 * N)
        x_final[0::3] = xs
        x_final[1::3] = ys
        x_final[2::3] = rs
        return x_final, w

    # Phase 1: Joint SA optimization with multiple initial aspect ratios
    init_widths = [0.7, 0.85, 1.0, 1.15, 1.3, 1.45]
    
    for w_init in init_widths:
        if time.time() > deadline:
            break
        h_init = 2.0 - w_init
        
        # Initialize with grid
        a = int(np.ceil(np.sqrt(N * w_init / h_init)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
            
        x0 = np.zeros(3 * N)
        for i in range(N):
            col = i % a
            row = i // a
            x0[3 * i] = (col + 0.5) * w_init / a
            x0[3 * i + 1] = (row + 0.5) * h_init / b
            x0[3 * i + 2] = min(w_init / a, h_init / b) * 0.45
        
        # Joint SA optimization
        x_j, w_j = sa_joint_optimize(x0, w_init, max_iters=2500)
        h_j = 2.0 - w_j
        check_and_update(x_j, w_j, h_j)
        
        # Second pass with perturbation
        if time.time() < deadline - 5:
            rng2 = np.random.default_rng(123 + int(w_init * 100))
            x0b = x_j.copy()
            for i in range(N):
                x0b[3*i] += rng2.normal(0, 0.02)
                x0b[3*i+1] += rng2.normal(0, 0.02)
                x0b[3*i+2] += rng2.normal(0, 0.01)
            x_j2, w_j2 = sa_joint_optimize(x0b, w_j, max_iters=1500)
            h_j2 = 2.0 - w_j2
            check_and_update(x_j2, w_j2, h_j2)

    # Phase 2: Fixed-w refinement on best aspect ratio found
    if best_result is not None and time.time() < deadline - 10:
        best_w = None
        # Re-derive w from best_result is tricky; use the w from last check
        # Instead, do a quick SA at several widths near the best
        for w_refine in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            if time.time() > deadline - 3:
                break
            h_refine = 2.0 - w_refine
            a = int(np.ceil(np.sqrt(N * w_refine / h_refine)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            x0r = np.zeros(3 * N)
            for i in range(N):
                col = i % a
                row = i // a
                x0r[3 * i] = (col + 0.5) * w_refine / a
                x0r[3 * i + 1] = (row + 0.5) * h_refine / b
                x0r[3 * i + 2] = min(w_refine / a, h_refine / b) * 0.45
            
            x_rf, _ = sa_joint_optimize(x0r, w_refine, max_iters=800)
            h_rf = 2.0 - w_refine
            check_and_update(x_rf, w_refine, h_rf)

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros(3 * N)
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[3 * idx] = (i + 0.5) * w / a
                best_result[3 * idx + 1] = (j + 0.5) * h / b
                best_result[3 * idx + 2] = r
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3 * i:3 * i + 3]
    
    return result
# EVOLVE-BLOCK-END
