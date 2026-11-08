import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _refine_powell(points):
        """Powell refinement for the hard objective."""
        def _hard_objective(points_flat):
            pts = points_flat.reshape(-1, 2)
            pts = _normalize_points(pts)
            score = compute_score(pts)
            if score == 0.0:
                return 1e6
            return -score
        
        try:
            res = opt.minimize(
                _hard_objective,
                points.flatten(),
                method='Powell',
                options={'maxiter': 20, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            return candidate, score
        except Exception:
            return points, compute_score(points)

    def _simulated_annealing(initial_points, n_steps=50000):
        """
        Simulated Annealing with worst-triangle targeting.
        """
        pts = _normalize_points(initial_points.copy())
        current_score = compute_score(pts)
        best_score = current_score
        best_pts = pts.copy()
        
        # Temperature schedule
        T_start = 0.05
        T_end = 1e-5
        cooling_rate = (T_end / T_start) ** (1.0 / n_steps)
        T = T_start
        
        rng = np.random.RandomState(42)
        
        for step in range(n_steps):
            if time.time() > start_time + time_limit - 2:
                break
                
            # Generate a move
            # 70% chance: move the vertex of the worst triangle
            # 30% chance: move a random vertex
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            
            worst_idx = np.argmin(areas)
            worst_tri = tri_indices[worst_idx]
            
            if rng.random() < 0.7:
                # Pick one of the 3 vertices of the worst triangle
                v_idx = worst_tri[rng.randint(0, 3)]
            else:
                v_idx = rng.randint(0, N)
            
            # Generate perturbation
            # Scale perturbation with temperature
            scale = 0.1 * T / T_start + 0.005
            dx = rng.normal(0, scale)
            dy = rng.normal(0, scale)
            
            new_pts = pts.copy()
            new_pts[v_idx, 0] += dx
            new_pts[v_idx, 1] += dy
            
            new_pts = _normalize_points(new_pts)
            new_score = compute_score(new_pts)
            
            # Acceptance criterion
            if new_score > current_score or rng.random() < np.exp((new_score - current_score) / T):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            
            T *= cooling_rate
            
        return best_pts, best_score

    # Generate initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. Two rings (7+6)
    for r_inner in [0.4, 0.5, 0.6]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r_inner, offset=np.pi/6)
        initializations.append(np.vstack([outer, inner]))
        
    # 4. Three rings (5+4+4)
    for r_mid, r_inner in [(0.6, 0.3), (0.7, 0.4)]:
        outer = _regular_polygon(5, 1.0)
        middle = _regular_polygon(4, r_mid, offset=np.pi/4)
        inner = _regular_polygon(4, r_inner, offset=np.pi/2)
        initializations.append(np.vstack([outer, middle, inner]))
        
    # 5. Random starts
    rng = np.random.RandomState(123)
    for _ in range(10):
        n_outer = rng.choice([5, 6, 7, 8, 9])
        n_inner = N - n_outer
        outer = _regular_polygon(n_outer, 1.0, offset=rng.uniform(0, 2*np.pi))
        inner = _regular_polygon(n_inner, rng.uniform(0.3, 0.7), offset=rng.uniform(0, 2*np.pi))
        pts = np.vstack([outer, inner])
        pts += rng.randn(*pts.shape) * 0.1
        initializations.append(pts)

    best_score = -float('inf')
    best_points = None

    # Run SA on each initialization
    for init in initializations:
        if time.time() > start_time + time_limit - 10:
            break
            
        # Determine steps based on remaining time
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        # Assume ~1ms per step (conservative), so we can do ~1000 steps per second
        # But we want to use the time effectively. Let's do a fixed chunk and check time.
        # Actually, let's just run a fixed number of steps and check time inside the loop.
        steps = 20000
        
        sa_pts, sa_score = _simulated_annealing(init, n_steps=steps)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
            
        # Polish the best candidate found by SA with Powell
        if time.time() > start_time + time_limit - 5:
            break
            
        pw_pts, pw_score = _refine_powell(sa_pts)
        
        if pw_score > best_score:
            best_score = pw_score
            best_points = pw_pts.copy()

    # Final polish on the absolute best
    if time.time() < start_time + time_limit - 2 and best_points is not None:
        final_pts, final_score = _refine_powell(best_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()

    if best_points is None:
        best_points = _normalize_points(initializations[0])

    return best_points
