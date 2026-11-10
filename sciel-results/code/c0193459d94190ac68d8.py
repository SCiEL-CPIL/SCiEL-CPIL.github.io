# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize (d_min/d_max)^2.
    
    Strategy: 
    1. Use a smooth surrogate objective (SoftMin/SoftMax via log-sum-exp) 
       to enable gradient-based optimization (L-BFGS-B).
    2. Initialize with hexagonal lattice priors and diverse geometric structures.
    3. Perform multi-start optimization with a final high-precision polish 
       on the raw objective.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

    def time_exceeded():
        return time.time() > deadline

    def evaluate(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(params, temperature=100.0):
        """
        Smooth surrogate for -(d_min/d_max)^2.
        Uses log-sum-exp to approximate min and max smoothly.
        We minimize this function.
        """
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        
        # Avoid log(0) and numerical instability
        dists = np.clip(dists, 1e-12, None)
        
        # Soft Min: exp(-1/T * sum(exp(-T * d)))
        # As T -> inf, this approaches min(d)
        # Soft Max: exp(1/T * sum(exp(T * d)))
        # As T -> inf, this approaches max(d)
        
        # For numerical stability in log-sum-exp:
        # log(sum(exp(a_i))) = max(a) + log(sum(exp(a_i - max(a))))
        
        # Soft Min
        a_min = -dists / temperature
        max_a_min = np.max(a_min)
        soft_min = -temperature * (max_a_min + np.log(np.sum(np.exp(a_min - max_a_min))))
        
        # Soft Max
        a_max = dists / temperature
        max_a_max = np.max(a_max)
        soft_max = temperature * (max_a_max + np.log(np.sum(np.exp(a_max - max_a_max))))
        
        if soft_max < 1e-12:
            return 0.0
            
        ratio = soft_min / soft_max
        # We want to maximize ratio^2, so minimize -ratio^2
        return -(ratio ** 2)

    best_points = None
    best_score = -1.0

    def update_best(points):
        nonlocal best_score, best_points
        score = evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate diverse starting configurations
    starts = []

    # 1. Hexagonal lattice: pick 16 closest to center from a larger grid
    def generate_hex_grid(radius, spacing=1.0):
        points = []
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                x = spacing * (q + 0.5 * r)
                y = spacing * (np.sqrt(3) / 2 * r)
                points.append([x, y])
        return np.array(points)

    for radius in [3, 4, 5]:
        hex_pts = generate_hex_grid(radius)
        hex_pts -= np.mean(hex_pts, axis=0)
        # Normalize to unit scale
        norm = np.max(np.linalg.norm(hex_pts, axis=1))
        if norm > 1e-10:
            hex_pts = hex_pts / norm
        dists_center = np.linalg.norm(hex_pts, axis=1)
        sorted_idx = np.argsort(dists_center)
        starts.append(hex_pts[sorted_idx[:16]])

    # 2. Circular arrangement
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(circle)

    # 3. Random starts with fixed seeds for reproducibility
    for seed in [1, 7, 13, 42, 99, 123, 456, 789]:
        np.random.seed(seed)
        pts = np.random.uniform(-1, 1, (n, 2))
        norm = np.max(np.linalg.norm(pts, axis=1))
        if norm > 1e-10:
            pts = pts / norm
        starts.append(pts)

    # 4. Square grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    starts.append(grid)

    # 5. Two concentric circles (8 points each)
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r1, r2 = 0.5, 1.0
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    starts.append(np.vstack([pts1, pts2]))

    # 6. Triangular lattice (equilateral)
    tri_pts = []
    for i in range(4):
        for j in range(4):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            tri_pts.append([x, y])
    tri_pts = np.array(tri_pts)
    tri_pts -= np.mean(tri_pts, axis=0)
    norm = np.max(np.linalg.norm(tri_pts, axis=1))
    if norm > 1e-10:
        tri_pts = tri_pts / norm
    starts.append(tri_pts)

    # Local optimization from each start
    # We use L-BFGS-B on the smooth objective first, then Nelder-Mead on the raw objective for polish
    
    for i, start in enumerate(starts):
        if time_exceeded():
            break

        # Phase 1: Gradient-based optimization on smooth surrogate
        # Try a few temperatures to balance smoothness and accuracy
        for temp in [10.0, 50.0, 100.0]:
            if time_exceeded():
                break
            
            def callback_smooth(xk):
                return time_exceeded()

            try:
                res = minimize(
                    lambda x: smooth_objective(x, temperature=temp),
                    start.ravel(),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    },
                    callback=callback_smooth
                )
                if not time_exceeded():
                    candidate = res.x.reshape((n, 2))
                    update_best(candidate)
            except Exception:
                pass

        # Phase 2: Polish with Nelder-Mead on the RAW objective (non-smooth)
        # Use the best candidate from Phase 1 or the initial start
        # We'll just run NM on the initial start and on the best found so far if it improved
        if not time_exceeded():
            def callback_nm(xk):
                return time_exceeded()

            try:
                # Objective for NM
                def objective_raw(params):
                    points = params.reshape((n, 2))
                    dists = pdist(points, 'euclidean')
                    d_min = np.min(dists)
                    d_max = np.max(dists)
                    if d_max < 1e-12:
                        return 0.0
                    ratio = d_min / d_max
                    return -(ratio ** 2)

                res = minimize(
                    objective_raw,
                    start.ravel(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    },
                    callback=callback_nm
                )
                if not time_exceeded():
                    update_best(res.x.reshape((n, 2)))
            except Exception:
                pass

    # Final polish: very tight optimization on the best
    if not time_exceeded() and best_points is not None:
        def callback_final(xk):
            return time_exceeded()

        try:
            def objective_raw(params):
                points = params.reshape((n, 2))
                dists = pdist(points, 'euclidean')
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max < 1e-12:
                    return 0.0
                ratio = d_min / d_max
                return -(ratio ** 2)

            res = minimize(
                objective_raw,
                best_points.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                },
                callback=callback_final
            )
            if not time_exceeded():
                update_best(res.x.reshape((n, 2)))
        except Exception:
            pass

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)

    return best_points

# EVOLVE-BLOCK-END
