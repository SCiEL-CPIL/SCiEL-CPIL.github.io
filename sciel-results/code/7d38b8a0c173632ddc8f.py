import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = len(x_flat)

    start_time = time.time()
    time_limit = 580.0  # Safety margin before 630s kill

    def _make_densified(path):
        """Build the full densified path: start + path + end, with linear interpolation."""
        full = [(0.0, 0.0, 0.0)]
        for p in path:
            full.append((float(p[0]), float(p[1]), float(p[2])))
        full.append((-2.0, 0.0, 90.0))
        dens = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m):
                t = (k + 1) / m
                dens.append((
                    p1[0] + (p2[0] - p1[0]) * t,
                    p1[1] + (p2[1] - p1[1]) * t,
                    p1[2] + (p2[2] - p1[2]) * t,
                ))
        dens.append(full[-1])
        return dens

    def evaluate_path_fast(path):
        """Fast evaluation using precomputed grid and vectorized rotation."""
        dens = _make_densified(path)
        n_dens = len(dens)
        # Precompute all rotations
        angles_rad = np.array([np.deg2rad(d[2]) for d in dens])
        cos_arr = np.cos(angles_rad)
        sin_arr = np.sin(angles_rad)
        cx_arr = np.array([d[0] for d in dens])
        cy_arr = np.array([d[1] for d in dens])
        
        # For each placement, compute mask and take intersection
        # Process in batches to limit memory
        batch_size = 20
        mask = np.ones(n_points, dtype=bool)
        for start in range(0, n_dens, batch_size):
            end = min(start + batch_size, n_dens)
            for i in range(start, end):
                ca = angles_rad[i]
                cos_a = cos_arr[i]
                sin_a = sin_arr[i]
                dx = x_flat - cx_arr[i]
                dy = y_flat - cy_arr[i]
                u = dx * cos_a + dy * sin_a
                v = -dx * sin_a + dy * cos_a
                in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
                in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
                mask &= (in_arm1 | in_arm2)
                if not np.any(mask):
                    return 0.0
        return float(np.sum(mask)) * 0.0001

    def evaluate_path(path):
        """Full evaluation: returns area score."""
        return evaluate_path_fast(path)

    # Multiple initial paths to try
    initial_paths = [
        [
            (-0.1, 0.1, 15.0),
            (-0.2, 0.2, 20.0),
            (-0.3, 0.3, 34.0),
            (-0.3, 0.3, 40.0),
            (-0.4, 0.3, 50.0),
            (-0.5, 0.2, 55.0),
            (-0.6, 0.2, 75.0),
            (-0.7, 0.3, 86.0),
        ],
        [
            (-0.05, 0.05, 10.0),
            (-0.15, 0.15, 25.0),
            (-0.25, 0.25, 40.0),
            (-0.35, 0.25, 55.0),
            (-0.45, 0.2, 70.0),
            (-0.55, 0.25, 82.0),
            (-0.65, 0.3, 88.0),
        ],
        [
            (-0.1, 0.0, 12.0),
            (-0.2, 0.1, 28.0),
            (-0.3, 0.2, 42.0),
            (-0.35, 0.2, 55.0),
            (-0.45, 0.15, 68.0),
            (-0.55, 0.2, 80.0),
            (-0.65, 0.25, 88.0),
            (-0.72, 0.28, 90.0),
        ],
    ]

    best_score = 0.0
    best_path = None
    current_path = None

    # Evaluate all initial paths and pick the best
    for ip in initial_paths:
        s = evaluate_path(ip)
        if s > best_score:
            best_score = s
            best_path = ip[:]
            current_path = ip[:]

    # Phase 1: Random local search with adaptive step sizes and restarts
    np.random.seed(42)
    epsilon_xy = 0.12
    epsilon_a = 2.5
    no_improve = 0
    max_no_improve = 25
    phase1_time_limit = 400.0  # Spend ~400s on random search
    restart_count = 0
    max_restarts = 3

    while time.time() - start_time < min(time_limit, phase1_time_limit):
        idx = np.random.randint(len(current_path))
        p = current_path[idx]

        new_x = p[0] + np.random.normal(0, epsilon_xy)
        new_y = p[1] + np.random.normal(0, epsilon_xy)
        new_a = p[2] + np.random.normal(0, epsilon_a)
        new_a = np.clip(new_a, 0.0, 90.0)

        new_p = (new_x, new_y, new_a)
        test_path = current_path[:]
        test_path[idx] = new_p

        score = evaluate_path(test_path)

        if score > best_score:
            best_score = score
            best_path = test_path
            current_path = test_path
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                epsilon_xy *= 0.8
                epsilon_a *= 0.8
                no_improve = 0
                if epsilon_xy < 0.008 and epsilon_a < 0.4:
                    # Restart from best with fresh noise
                    current_path = best_path[:]
                    epsilon_xy = 0.25
                    epsilon_a = 4.0
                    restart_count += 1
                    if restart_count >= max_restarts:
                        # Try a different initial path
                        for ip in initial_paths:
                            if ip is not best_path:
                                current_path = ip[:]
                                break
                        else:
                            current_path = best_path[:]
                        restart_count = 0

    # Phase 2: Coordinate-descent polish with multiple scales
    polish_deadline = time_limit - 30.0  # Leave 30s buffer for final polish

    # Multi-scale polish: start coarse, get finer
    scales = [
        (0.04, 0.8),
        (0.02, 0.4),
        (0.01, 0.2),
        (0.005, 0.1),
        (0.002, 0.05),
        (0.001, 0.02),
        (0.0005, 0.01),
    ]

    for polish_xy, polish_a in scales:
        if time.time() > polish_deadline:
            break
        # Multiple passes at each scale
        for pass_iter in range(3):
            if time.time() > polish_deadline:
                break
            improved = False
            for idx in range(len(best_path)):
                if time.time() > polish_deadline:
                    break
                p = best_path[idx]
                best_local = best_score
                best_local_p = p

                # Try perturbations in x
                for dx in (-polish_xy, -polish_xy * 0.5, polish_xy * 0.5, polish_xy):
                    test_p = (p[0] + dx, p[1], p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                # Try perturbations in y
                for dy in (-polish_xy, -polish_xy * 0.5, polish_xy * 0.5, polish_xy):
                    test_p = (p[0], p[1] + dy, p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                # Try perturbations in a
                for da in (-polish_a, -polish_a * 0.5, polish_a * 0.5, polish_a):
                    new_a = np.clip(p[2] + da, 0.0, 90.0)
                    test_p = (p[0], p[1], new_a)
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                if best_local > best_score:
                    best_path[idx] = best_local_p
                    best_score = best_local
                    improved = True

            if not improved:
                break

    # Phase 3: Final high-precision polish with very small steps
    if time.time() < polish_deadline:
        fine_xy = 0.0002
        fine_a = 0.005
        for pass_iter in range(5):
            if time.time() > polish_deadline:
                break
            improved = False
            for idx in range(len(best_path)):
                if time.time() > polish_deadline:
                    break
                p = best_path[idx]
                best_local = best_score
                best_local_p = p

                for dx in (-fine_xy, fine_xy):
                    test_p = (p[0] + dx, p[1], p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                for dy in (-fine_xy, fine_xy):
                    test_p = (p[0], p[1] + dy, p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                for da in (-fine_a, fine_a):
                    new_a = np.clip(p[2] + da, 0.0, 90.0)
                    test_p = (p[0], p[1], new_a)
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                if best_local > best_score:
                    best_path[idx] = best_local_p
                    best_score = best_local
                    improved = True

            if not improved:
                break

    return np.array(best_path)
