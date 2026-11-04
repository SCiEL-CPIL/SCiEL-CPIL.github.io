"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # We will use Simulated Annealing to maximize the sum of radii
    # for circles packed in a rectangle of perimeter 4.
    # We will try a few aspect ratios.
    
    best_score = -1
    best_solution = None
    
    # Aspect ratios to try: w/h
    # Perimeter 4 => w + h = 2.
    # If ar = w/h, then h = 2/(1+ar), w = 2*ar/(1+ar).
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66]
    
    # Pre-calculate indices for upper triangle
    iu, ju = np.triu_indices(N, 1)
    
    for ar in aspect_ratios:
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initialize in a grid pattern
        side = int(np.ceil(np.sqrt(N)))
        rows = side
        cols = side
        while rows * cols < N:
            if rows <= cols:
                rows += 1
            else:
                cols += 1
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        for i in range(N):
            row = i // cols
            col = i % cols
            xs[i] = (col + 0.5) * w / cols
            ys[i] = (row + 0.5) * h / rows
            # Initial radius small enough to not overlap in grid
            max_r_cell = min(w / cols, h / rows) / 2.0
            rs[i] = max_r_cell * 0.5
            
        # Simulated Annealing
        T_start = 0.5
        T_end = 1e-6
        steps = 50000
        
        def get_cost(xs, ys, rs, w, h):
            cost = -np.sum(rs)
            
            # Boundary penalty
            vx = np.maximum(0, rs - xs) + np.maximum(0, xs - (w - rs))
            vy = np.maximum(0, rs - ys) + np.maximum(0, ys - (h - rs))
            penalty_boundary = np.sum(vx**2) + np.sum(vy**2)
            
            # Overlap penalty
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = rs[:, None] + rs[None, :]
            
            overlaps = np.maximum(0, min_dist[iu, ju] - dist[iu, ju])
            penalty_overlap = np.sum(overlaps**2)
            
            weight = 100.0
            cost += weight * (penalty_boundary + penalty_overlap)
            
            return cost

        current_cost = get_cost(xs, ys, rs, w, h)
        best_cost_in_sa = current_cost
        best_xs = xs.copy()
        best_ys = ys.copy()
        best_rs = rs.copy()
        
        for step in range(steps):
            T = T_start * (T_end / T_start) ** (step / steps)
            
            i = np.random.randint(N)
            
            ox, oy, orr = xs[i], ys[i], rs[i]
            
            # Scale perturbation with temperature
            scale = 0.05 * (T / T_start + 0.1)
            dx = np.random.uniform(-scale, scale)
            dy = np.random.uniform(-scale, scale)
            dr = np.random.uniform(-scale, scale)
            
            xs[i] += dx
            ys[i] += dy
            rs[i] += dr
            
            if rs[i] < 0.01:
                rs[i] = 0.01
            
            new_cost = get_cost(xs, ys, rs, w, h)
            
            delta = new_cost - current_cost
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost_in_sa:
                    best_cost_in_sa = new_cost
                    best_xs = xs.copy()
                    best_ys = ys.copy()
                    best_rs = rs.copy()
            else:
                if np.random.rand() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    xs[i] = ox
                    ys[i] = oy
                    rs[i] = orr
        
        # Check validity of best solution
        valid = True
        if np.any(best_xs < best_rs - 1e-5) or np.any(best_xs > w - best_rs + 1e-5):
            valid = False
        if np.any(best_ys < best_rs - 1e-5) or np.any(best_ys > h - best_rs + 1e-5):
            valid = False
        
        dx = best_xs[:, None] - best_xs[None, :]
        dy = best_ys[:, None] - best_ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = best_rs[:, None] + best_rs[None, :]
        if np.any(min_dist[iu, ju] - dist[iu, ju] > 1e-5):
            valid = False
            
        if valid:
            score = np.sum(best_rs)
            if score > best_score:
                best_score = score
                best_solution = np.column_stack([best_xs, best_ys, best_rs])
                
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
# EVOLVE-BLOCK-END
