"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 150.0  # evaluator kills after 150s; keep safety margin


def _validate(x, W, H):
    """Check feasibility: circles inside rectangle and non-overlapping."""
    n = len(x) // 3
    coords = x.reshape(n, 3)
    cx, cy, r = coords[:, 0], coords[:, 1], coords[:, 2]
    # containment
    if np.any(cx - r < -1e-9) or np.any(cx + r > W + 1e-9):
        return False
    if np.any(cy - r < -1e-9) or np.any(cy + r > H + 1e-9):
        return False
    # non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True


def _objective(x, W, H):
    """Negative sum of radii (to minimize)."""
    n = len(x) // 3
    r = x[2::3]
    return -np.sum(r)


def _constraints_factory(W, H):
    n = N
    # Containment constraints (4 per circle)
    # Non-overlap constraints (n*(n-1)/2)
    def con_fun(x):
        coords = x.reshape(n, 3)
        cx, cy, r = coords[:, 0], coords[:, 1], coords[:, 2]
        # containment
        c = []
        c.append(cx - r)          # >= 0
        c.append(W - cx - r)      # >= 0
        c.append(cy - r)          # >= 0
        c.append(H - cy - r)      # >= 0
        # non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                c.append(dist - r[i] - r[j])
        return np.array(c)

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, W))   # cx
        bounds.append((0, H))   # cy
        bounds.append((1e-6, min(W, H) / 2))  # r
    return con_fun, bounds


def _run_slsqp(x0, W, H, time_limit):
    """Run SLSQP optimization."""
    con_fun, bounds = _constraints_factory(W, H)
    cons = [{'type': 'ineq', 'fun': con_fun}]
    
    result = minimize(
        _objective, x0, args=(W, H),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
    )
    return result.x


def _grid_init(W, H, n_cols, n_rows):
    """Create initial grid positions for N circles."""
    positions = []
    for i in range(n_cols):
        for j in range(n_rows):
            if len(positions) >= N:
                break
            cx = (i + 0.5) * W / n_cols
            cy = (j + 0.5) * H / n_rows
            # Initial radius: half the minimum cell dimension minus small margin
            r = 0.5 * min(W / n_cols, H / n_rows) * 0.95
            positions.append([cx, cy, r])
    # Pad if needed
    while len(positions) < N:
        positions.append([W / 2, H / 2, 0.05])
    return np.array(positions[:N]).flatten()


def construct():
    start_time = time.time()
    best_x = None
    best_val = -np.inf

    # Try multiple aspect ratios
    aspect_ratios = [
        (1.0, 1.0),      # square
        (1.2, 8/6),      # W:H = 1.2:1  -> W=1.2, H=0.8
        (0.8, 1.2),      # W:H = 1:1.2
        (1.5, 2/3),      # W=1.5, H=0.5
        (0.5, 1.5),      # W=0.5, H=1.5
        (1.3, 0.7/1.3*2/2),  # more elongated
    ]
    # Normalize: W + H = 2
    aspect_ratios = [
        (1.0, 1.0),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.6, 0.4),
        (0.4, 1.6),
    ]

    for W, H in aspect_ratios:
        if time.time() - start_time > DEADLINE - 20:
            break

        # Try different grid configurations
        grid_configs = [
            (5, 5),   # 5x5 grid, use 21
            (4, 6),   # 4x6 grid, use 21
            (6, 4),   # 6x4 grid, use 21
            (7, 3),   # 7x3 grid, use 21
            (3, 7),   # 3x7 grid, use 21
        ]

        for n_cols, n_rows in grid_configs:
            if time.time() - start_time > DEADLINE - 20:
                break

            x0 = _grid_init(W, H, n_cols, n_rows)
            
            try:
                result_x = _run_slsqp(x0, W, H, DEADLINE - 20)
                if _validate(result_x, W, H):
                    val = np.sum(result_x[2::3])
                    if val > best_val:
                        best_val = val
                        best_x = result_x
            except Exception:
                pass

    # If no valid solution found, fall back to simple grid
    if best_x is None:
        W, H = 1.0, 1.0
        best_x = _grid_init(W, H, 5, 5)

    # Format output: array of shape (N, 3)
    result = best_x.reshape(N, 3)
    return result


if __name__ == '__main__':
    result = construct()
    print(f"Shape: {result.shape}")
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
