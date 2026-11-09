# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 13
# Precompute all 286 triangle index triples
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """Hard Heilbronn score: A_min / A_hull."""
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 1e-12:
        return 0.0
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas) / A_hull

def simulated_annealing(init_pts, time_limit, rng, T_start=1.0, T_end=0.001, cooling=0.9999):
    """
    Simulated Annealing on the hard objective.
    Accepts worse moves with probability exp(-delta_score / T).
    """
    current_pts = init_pts.copy()
    current_score = compute_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T = T_start
    n_points = len(current_pts)
    
    while time.time() < time_limit and T > T_end:
        # Pick a random point to move
        idx = rng.integers(n_points)
        
        # Generate a random perturbation (Gaussian)
        # Scale perturbation with temperature? Or fixed?
        # Let's use a fixed scale that might be too large initially.
        # Better: scale perturbation with T? No, T is abstract here.
        # Let's use a step size that decays slightly or is constant.
        step_size = 0.1 * (0.1 + 0.9 * (T / T_start))
        
        delta = rng.normal(0, step_size, size=2)
        
        old_x, old_y = current_pts[idx, 0], current_pts[idx, 1]
        new_x = old_x + delta[0]
        new_y = old_y + delta[1]
        
        # Check bounds? The problem doesn't specify bounds, but scaling matters.
        # Since the score is scale-invariant (ratio), we can normalize after?
        # No, we just maximize the ratio. The absolute scale doesn't matter for the score,
        # but it affects the "energy" landscape if we used absolute areas.
        # Here score is ratio, so it's scale invariant.
        # However, if points drift too far, numerical issues might arise.
        # Let's keep them within a reasonable box, e.g., [-5, 5].
        if new_x < -5 or new_x > 5 or new_y < -5 or new_y > 5:
            T *= cooling
            continue
            
        current_pts[idx, 0] = new_x
        current_pts[idx, 1] = new_y
        
        new_score = compute_score(current_pts)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Note: we are maximizing, so delta = new - current < 0
            # Probability = exp(-abs(delta)/T)
            delta = new_score - current_score
            if delta < 0:
                # Avoid division by zero or overflow
                if T < 1e-12:
                    prob = 0.0
                else:
                    prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    current_pts[idx, 0] = old_x
                    current_pts[idx, 1] = old_y
        
        T *= cooling
        
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    rng = np.random.default_rng(42)

    best_points = None
    best_score = -1.0

    candidates = []
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    # 2. 7 outer, 6 inner
    ang7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p7 = np.column_stack([np.cos(ang7), np.sin(ang7)])
    ang6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    p6 = np.column_stack([0.55 * np.cos(ang6), 0.55 * np.sin(ang6)])
    candidates.append(np.vstack([p7, p6]))
    # 3. 7 outer, 6 inner (offset)
    ang6b = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 12
    p6b = np.column_stack([0.6 * np.cos(ang6b), 0.6 * np.sin(ang6b)])
    candidates.append(np.vstack([p7, p6b]))
    # 4. 5/5/3
    ang5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5o = np.column_stack([np.cos(ang5), np.sin(ang5)])
    ang5m = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5m = np.column_stack([0.6 * np.cos(ang5m), 0.6 * np.sin(ang5m)])
    ang3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
    p3i = np.column_stack([0.3 * np.cos(ang3), 0.3 * np.sin(ang3)])
    candidates.append(np.vstack([p5o, p5m, p3i]))
    # 5. 4/5/4
    ang4o = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4o = np.column_stack([np.cos(ang4o), np.sin(ang4o)])
    ang5m2 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5m2 = np.column_stack([0.65 * np.cos(ang5m2), 0.65 * np.sin(ang5m2)])
    ang4i = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i = np.column_stack([0.35 * np.cos(ang4i), 0.35 * np.sin(ang4i)])
    candidates.append(np.vstack([p4o, p5m2, p4i]))
    # 6. 6 outer, 7 inner
    ang6o = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p6o = np.column_stack([np.cos(ang6o), np.sin(ang6o)])
    ang7i = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    p7i = np.column_stack([0.5 * np.cos(ang7i), 0.5 * np.sin(ang7i)])
    candidates.append(np.vstack([p6o, p7i]))
    # 7. Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    fib_angles = 2 * np.pi * np.arange(N) / (golden ** 2)
    fib_radii = np.sqrt(np.arange(N) / N)
    candidates.append(np.column_stack([fib_radii * np.cos(fib_angles),
                                       fib_radii * np.sin(fib_angles)]))
    # 8. 4/5/4
    ang4a = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4a = np.column_stack([np.cos(ang4a), np.sin(ang4a)])
    ang5b = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5b = np.column_stack([0.65 * np.cos(ang5b), 0.65 * np.sin(ang5b)])
    ang4c = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4c = np.column_stack([0.3 * np.cos(ang4c), 0.3 * np.sin(ang4c)])
    candidates.append(np.vstack([p4a, p5b, p4c]))
    # 9. 5/4/4
    ang5o2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 10
    p5o2 = np.column_stack([np.cos(ang5o2), np.sin(ang5o2)])
    ang4m2 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4m2 = np.column_stack([0.6 * np.cos(ang4m2), 0.6 * np.sin(ang4m2)])
    ang4i2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i2 = np.column_stack([0.3 * np.cos(ang4i2), 0.3 * np.sin(ang4i2)])
    candidates.append(np.vstack([p5o2, p4m2, p4i2]))
    # 10. Random
    for _ in range(5):
        candidates.append(rng.random((N, 2)) * 2 - 1)
    # Perturbed structured
    for base_idx in [0, 1, 3, 4]:
        if base_idx < len(candidates):
            candidates.append(candidates[base_idx] + rng.normal(0, 0.03, size=(N, 2)))

    n_cand = len(candidates)
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline - 5:
            break
        
        init_score = compute_score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()

        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        remaining_cands = n_cand - i
        time_per_cand = min(remaining / max(remaining_cands, 1), 60.0)
        cand_deadline = time.time() + time_per_cand

        # Run Simulated Annealing
        if time.time() < cand_deadline:
            sa_pts, sa_score = simulated_annealing(init_pts, cand_deadline, rng)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts.copy()

        # If we still have time, run SA on the best found so far (restart from best)
        if time.time() < deadline - 10 and best_points is not None:
             # Use a fraction of remaining time for this "best" restart
             restart_deadline = time.time() + min((deadline - 10 - time.time()) * 0.5, 30.0)
             if restart_deadline > time.time():
                 sa_pts2, sa_score2 = simulated_annealing(best_points, restart_deadline, rng)
                 if sa_score2 > best_score:
                     best_score = sa_score2
                     best_points = sa_pts2.copy()

    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
# EVOLVE-BLOCK-END
