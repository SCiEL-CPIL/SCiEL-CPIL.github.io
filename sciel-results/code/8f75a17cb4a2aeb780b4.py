"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 110.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h):
        if np.any(r < -1e-9):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
        # Check overlaps vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        # Upper triangle check
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.any((dist[mask] < r_sum[mask] - 1e-9)):
            return False
        return True

    def compute_energy(x, y, r, w, h):
        """
        Compute energy (penalty) for a configuration. Lower is better.
        Energy = -sum(r) + penalty_terms
        """
        sum_r = np.sum(r)
        
        # Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlaps = np.maximum(r_sum - dist, 0)
        overlap_pen = np.sum(overlaps[mask] ** 2)
        
        # Boundary penalty
        bound_pen = 0.0
        bound_pen += np.sum(np.maximum(r - x, 0) ** 2)
        bound_pen += np.sum(np.maximum(x + r - w, 0) ** 2)
        bound_pen += np.sum(np.maximum(r - y, 0) ** 2)
        bound_pen += np.sum(np.maximum(y + r - h, 0) ** 2)
        
        # Radius positivity penalty
        neg_r = np.maximum(-r, 0)
        r_pen = np.sum(neg_r ** 2)
        
        penalty = 100.0 * (overlap_pen + bound_pen + r_pen)
        
        return -sum_r + penalty

    def simulated_annealing(x, y, r, w, h, rng, max_iters=5000, T0=0.1, T1=0.001, alpha=0.999):
        """
        Simulated annealing to optimize circle packing.
        Moves: 
        1. Move a circle's position
        2. Change a circle's radius
        3. Swap two circles' positions
        4. Uniform scale all radii
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        current_energy = compute_energy(x, y, r, w, h)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_energy = current_energy
        
        T = T0
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Generate a random move
            move_type = rng.integers(0, 4)
            i = rng.integers(0, N)
            
            old_x, old_y, old_r = x[i], y[i], r[i]
            
            if move_type == 0:
                # Move position
                step = 0.05 * T + 0.01
                dx = rng.normal(0, step)
                dy = rng.normal(0, step)
                x[i] = np.clip(x[i] + dx, 0, w)
                y[i] = np.clip(y[i] + dy, 0, h)
                
            elif move_type == 1:
                # Change radius
                step = 0.02 * T + 0.005
                delta_r = rng.normal(0, step)
                r[i] = max(0, r[i] + delta_r)
                
            elif move_type == 2:
                # Swap positions with another circle
                j = rng.integers(0, N)
                if j != i:
                    x[i], x[j] = x[j], x[i]
                    y[i], y[j] = y[j], y[i]
                    r[i], r[j] = r[j], r[i]
                    
            else:
                # Uniform scale all radii
                scale = 1.0 + rng.normal(0, 0.02 * T + 0.005)
                r = np.maximum(0, r * scale)
            
            new_energy = compute_energy(x, y, r, w, h)
            delta = new_energy - current_energy
            
            # Accept or reject
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            else:
                # Reject move
                x[i], y[i], r[i] = old_x, old_y, old_r
            
            # Cool down
            T *= alpha
        
        return best_x, best_y, best_r

    def hex_init(w, h, est_r, seed):
        rng = np.random.default_rng(seed)
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            
            if row % 2 == 1:
                x += 0.5 * w / n_cols
                if x > w - est_r:
                    x -= w / n_cols
                elif x < est_r:
                    x += w / n_cols

            xs[i] = x
            ys[i] = y

        jitter_scale = 0.2 * est_r
        xs += rng.normal(0, jitter_scale, N)
        ys += rng.normal(0, jitter_scale, N)
        
        xs = np.clip(xs, est_r * 0.5, w - est_r * 0.5)
        ys = np.clip(ys, est_r * 0.5, h - est_r * 0.5)
        
        return xs, ys

    # Aspect ratios to try
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.6, 1.3, 0.75, 1.1, 0.9, 1.8, 0.55]
    aspect_ratios.sort(key=lambda ar: abs(ar - 1.0))

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        est_r = np.sqrt(0.9 * w * h / (N * np.pi))
        init_r = est_r * 0.7

        # Determine restarts
        time_left = deadline - time.time()
        max_restarts = max(1, min(6, int(time_left / 8)))

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = restart * 1000 + int(ar * 100)
            rng = np.random.default_rng(seed)
            xs, ys = hex_init(w, h, init_r, seed)
            
            r0 = np.full(N, init_r)
            
            # Simulated annealing
            x_sa, y_sa, r_sa = simulated_annealing(xs, ys, r0, w, h, rng, max_iters=3000, T0=0.05, T1=0.001, alpha=0.9995)
            
            # Validate and score
            if validate(x_sa, y_sa, r_sa, w, h):
                score = np.sum(r_sa)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x_sa, y_sa, r_sa])

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END
