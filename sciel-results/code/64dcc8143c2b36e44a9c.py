"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    best_score = -np.inf
    best_result = None

    def _make_hex_init():
        """Create hexagonal lattice initialization."""
        points = []
        for row in range(10):
            for col in range(10):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                points.append((x, y))
        pts = np.array(points)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        return init_pts

    def _make_grid_init():
        """Create grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([0.5/side + i/side, 0.5/side + j/side])
        return np.array(centers[:N])

    def _make_random_init(seed):
        """Create random initialization."""
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    def _compute_initial_radius(pts):
        """Compute initial radius based on minimum pairwise distance."""
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        return min_dist / 2 * 0.85

    def _evaluate(centers, radii):
        """Compute sum of radii if valid, else -inf."""
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9:
                return -np.inf
            if y - r < -1e-9 or y + r > 1 + 1e-9:
                return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    return -np.inf
        return np.sum(radii)

    def _pbd_simulate(pos0, r0, n_iters=2000, expand_rate=0.0005):
        """
        Position-Based Dynamics simulation:
        - Start with given positions and radii
        - Each iteration: expand all radii slightly, then project to resolve overlaps
        - Returns final positions and radii
        """
        pos = pos0.copy().astype(float)
        r = r0.copy().astype(float)
        
        for it in range(n_iters):
            # Expand radii
            r += expand_rate
            
            # Constraint projection: multiple passes
            for _pass in range(3):
                # Pairwise circle-circle constraints
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pos[j, 0] - pos[i, 0]
                        dy = pos[j, 1] - pos[i, 1]
                        d = np.sqrt(dx * dx + dy * dy)
                        min_d = r[i] + r[j]
                        if d < min_d and d > 1e-12:
                            # Push apart
                            overlap = min_d - d
                            nx = dx / d
                            ny = dy / d
                            # Weighted by inverse mass (equal here)
                            pos[i, 0] -= 0.5 * overlap * nx
                            pos[i, 1] -= 0.5 * overlap * ny
                            pos[j, 0] += 0.5 * overlap * nx
                            pos[j, 1] += 0.5 * overlap * ny
                        elif d <= 1e-12:
                            # Degenerate: push in arbitrary direction
                            pos[i, 0] -= 0.5 * min_d * 0.01
                            pos[j, 0] += 0.5 * min_d * 0.01
                
                # Boundary constraints: clamp positions
                for i in range(N):
                    # Left wall
                    if pos[i, 0] < r[i]:
                        pos[i, 0] = r[i]
                    # Right wall
                    if pos[i, 0] > 1.0 - r[i]:
                        pos[i, 0] = 1.0 - r[i]
                    # Bottom wall
                    if pos[i, 1] < r[i]:
                        pos[i, 1] = r[i]
                    # Top wall
                    if pos[i, 1] > 1.0 - r[i]:
                        pos[i, 1] = 1.0 - r[i]
                    
                    # If radius exceeds half the box, shrink it
                    max_r = 0.5
                    if r[i] > max_r:
                        r[i] = max_r
            
            # Check if we're making progress (optional early stop)
            if it % 500 == 0 and it > 0:
                pass  # Could add convergence check
        
        return pos, r

    def _validity_check(arr):
        """Check if circles are valid: inside [0,1]^2 and non-overlapping."""
        if arr.shape != (N, 3):
            return False
        x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
        if np.any(r <= 0):
            return False
        if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
            return False
        if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
            return False
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - 1e-7:
                    return False
        return True

    # Define initializations
    inits = []
    inits.append(('hex', _make_hex_init()))
    inits.append(('grid', _make_grid_init()))
    for seed in range(15):
        inits.append(('rand', _make_random_init(seed + 42)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if time.time() > deadline - 5:
            break

        init_r = _compute_initial_radius(init_pts)
        
        # Run PBD simulation with varying expand rates
        for expand_rate in [0.0003, 0.0005, 0.0008]:
            if time.time() > deadline - 3:
                break
            
            n_iters = 3000
            pos, r = _pbd_simulate(init_pts, np.full(N, init_r), 
                                    n_iters=n_iters, expand_rate=expand_rate)
            
            # Clamp to valid range
            for i in range(N):
                pos[i, 0] = np.clip(pos[i, 0], r[i], 1.0 - r[i])
                pos[i, 1] = np.clip(pos[i, 1], r[i], 1.0 - r[i])
                r[i] = min(r[i], 0.5)
            
            score = _evaluate(pos, r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([pos, r])

    # If no valid result found, return grid fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final validity check and fix if needed
    if not _validity_check(best_result):
        # Reduce radii until valid
        for _ in range(50):
            best_result[:, 2] *= 0.95
            if _validity_check(best_result):
                break
    
    return best_result
