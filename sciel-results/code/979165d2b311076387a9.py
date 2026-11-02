"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles.
Strategy: Simulated annealing on centers + aspect ratio, with LP-based radius polish.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import Delaunay
import time

N = 21
DEADLINE = 138.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    best_score = 0.0
    best_result = None

    def get_delaunay_edges(centers):
        """Get Delaunay triangulation edges for circle centers."""
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        edge = tuple(sorted((simplex[i], simplex[j])))
                        edges.add(edge)
            return list(edges)
        except Exception:
            return [(i, j) for i in range(N) for j in range(i + 1, N)]

    def solve_radii_lp(centers, w, h, edges):
        """Solve LP for maximum sum of radii given fixed centers and adjacency."""
        c = -np.ones(N)
        x = centers[:, 0]
        y = centers[:, 1]

        A_ub_list = []
        b_ub_list = []

        for i in range(N):
            for b in [x[i], w - x[i], y[i], h - y[i]]:
                row = np.zeros(N)
                row[i] = 1.0
                A_ub_list.append(row)
                b_ub_list.append(b)

        for (i, j) in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(dist)

        if not A_ub_list:
            return np.zeros(N)

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        bounds = [(0, None)] * N

        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                return np.maximum(result.x, 0.0)
            else:
                return np.zeros(N)
        except Exception:
            return np.zeros(N)

    def evaluate(w, centers):
        """Evaluate a configuration: return (score, radii, valid)."""
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return 0.0, np.zeros(N), False

        edges = get_delaunay_edges(centers)
        r = solve_radii_lp(centers, w, h, edges)

        # Validate
        valid = True
        for i in range(N):
            if r[i] < -1e-9:
                valid = False
                break
            if centers[i, 0] - r[i] < -1e-6 or centers[i, 0] + r[i] > w + 1e-6:
                valid = False
                break
            if centers[i, 1] - r[i] < -1e-6 or centers[i, 1] + r[i] > h + 1e-6:
                valid = False
                break

        if valid:
            for (i, j) in edges:
                dist = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                if dist < r[i] + r[j] - 1e-6:
                    valid = False
                    break

        score = np.sum(r) if valid else 0.0
        return score, r, valid

    def make_initial(w, h, seed=0):
        """Create initial centers in a grid pattern."""
        rng = np.random.RandomState(seed)
        side = int(np.ceil(np.sqrt(N)))
        centers = np.zeros((N, 2))
        for idx in range(N):
            i = idx % side
            j = idx // side
            centers[idx, 0] = (i + 0.5) * w / side + rng.randn() * 0.02
            centers[idx, 1] = (j + 0.5) * h / side + rng.randn() * 0.02
        centers[:, 0] = np.clip(centers[:, 0], 0.01, w - 0.01)
        centers[:, 1] = np.clip(centers[:, 1], 0.01, h - 0.01)
        return centers

    # Simulated annealing
    rng = np.random.RandomState(42)

    # Try a few different aspect ratios as starting points
    aspect_starts = [1.0, 1.5, 0.8, 2.0, 1.2]

    for ar_idx, ar in enumerate(aspect_starts):
        if time.time() > DEADLINE:
            break

        w = ar * 2.0 / (ar + 1.0)
        h = 2.0 - w

        centers = make_initial(w, h, seed=ar_idx * 7)
        score, r, valid = evaluate(w, centers)

        if score > best_score:
            best_score = score
            best_result = (w, centers.copy(), r.copy())

        # Annealing parameters
        T_start = 0.05
        T_end = 0.0001
        # Number of steps based on available time
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        # Estimate: each eval takes ~2ms, so we can do ~remaining/0.002 steps
        # But we want to spread across all aspect ratios
        steps_per_ar = int(remaining / (len(aspect_starts) - ar_idx) / 0.003)
        steps_per_ar = max(500, min(steps_per_ar, 50000))

        if steps_per_ar < 100:
            continue

        alpha = (T_end / T_start) ** (1.0 / steps_per_ar)
        T = T_start

        current_w = w
        current_centers = centers.copy()
        current_score = score
        best_local_score = score
        best_local_w = w
        best_local_centers = centers.copy()

        for step in range(steps_per_ar):
            if time.time() > DEADLINE:
                break

            # Choose move type
            move_type = rng.random()

            if move_type < 0.7:
                # Move one circle
                idx = rng.randint(N)
                step_size = T * 2.0
                delta = rng.randn(2) * step_size
                new_centers = current_centers.copy()
                new_centers[idx, 0] = np.clip(current_centers[idx, 0] + delta[0], 0.01, current_w - 0.01)
                new_centers[idx, 1] = np.clip(current_centers[idx, 1] + delta[1], 0.01, 2.0 - current_w - 0.01)
                new_w = current_w
            elif move_type < 0.85:
                # Move aspect ratio
                delta_w = rng.randn() * T
                new_w = np.clip(current_w + delta_w, 0.2, 1.8)
                new_centers = current_centers.copy()
                # Rescale centers to new dimensions
                new_h = 2.0 - new_w
                old_h = 2.0 - current_w
                new_centers[:, 0] = current_centers[:, 0] * (new_w / current_w)
                new_centers[:, 1] = current_centers[:, 1] * (new_h / old_h)
                new_centers[:, 0] = np.clip(new_centers[:, 0], 0.01, new_w - 0.01)
                new_centers[:, 1] = np.clip(new_centers[:, 1], 0.01, new_h - 0.01)
            else:
                # Move two circles
                idx1, idx2 = rng.choice(N, size=2, replace=False)
                step_size = T * 2.0
                delta1 = rng.randn(2) * step_size
                delta2 = rng.randn(2) * step_size
                new_centers = current_centers.copy()
                new_centers[idx1, 0] = np.clip(current_centers[idx1, 0] + delta1[0], 0.01, current_w - 0.01)
                new_centers[idx1, 1] = np.clip(current_centers[idx1, 1] + delta1[1], 0.01, 2.0 - current_w - 0.01)
                new_centers[idx2, 0] = np.clip(current_centers[idx2, 0] + delta2[0], 0.01, current_w - 0.01)
                new_centers[idx2, 1] = np.clip(current_centers[idx2, 1] + delta2[1], 0.01, 2.0 - current_w - 0.01)
                new_w = current_w

            new_score, new_r, new_valid = evaluate(new_w, new_centers)

            # Accept if better, or with probability exp((new_score - current_score)/T)
            if new_score >= current_score:
                current_score = new_score
                current_w = new_w
                current_centers = new_centers.copy()
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_w = new_w
                    best_local_centers = new_centers.copy()
            else:
                delta_s = current_score - new_score
                if delta_s < T * 5:  # Only consider if not too bad
                    prob = np.exp(-delta_s / T)
                    if rng.random() < prob:
                        current_score = new_score
                        current_w = new_w
                        current_centers = new_centers.copy()

            T *= alpha

        if best_local_score > best_score:
            best_score = best_local_score
            best_result = (best_local_w, best_local_centers.copy(), None)

    # Final polish on best result
    if best_result is not None and time.time() < DEADLINE - 5:
        w, centers, r = best_result
        h = 2.0 - w
        edges = get_delaunay_edges(centers)
        r_opt = solve_radii_lp(centers, w, h, edges)
        score = np.sum(r_opt)
        if score > best_score:
            best_score = score
            best_result = (w, centers.copy(), r_opt.copy())

    # Build final result
    if best_result is not None:
        w, centers, r = best_result
        h = 2.0 - w
        result_array = np.zeros((N, 3))
        for i in range(N):
            x = np.clip(centers[i, 0], r[i], w - r[i])
            y = np.clip(centers[i, 1], r[i], h - r[i])
            r_i = max(0.001, min(r[i], min(w, h) / 2 - 0.001))
            result_array[i] = [x, y, r_i]
        return result_array
    else:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for idx in range(N):
            i = idx % side
            j = idx // side
            r = 0.5 / side - 0.001
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            cells.append([x, y, r])
        return np.array(cells)
