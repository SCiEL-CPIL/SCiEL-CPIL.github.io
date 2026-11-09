import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def score(points):
        """Compute the score: product of pairwise distances / n^n after scaling to diameter 2"""
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D < 1e-12:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    best_score = score(regular_octagon())
    best_points = regular_octagon().copy()

    # STRUCTURAL CHANGE: Parametrize 8 points as vertices of a convex polygon
    # using 8 positive edge lengths that sum to a fixed perimeter.
    # This reduces the search space from 16D to 7D (since sum of edges is fixed)
    # and inherently enforces convexity and distinctness.
    
    # Build points from edge lengths
    def build_points_from_edges(edges):
        """
        Given 8 positive edge lengths, construct a convex polygon with those edges.
        We place vertices sequentially, choosing angles to close the polygon.
        
        Strategy: Use the fact that for a convex polygon with given side lengths,
        we can determine the turning angles. We'll use a parameterization where
        we specify 7 angles (the 8th is determined by closure).
        
        Actually, a simpler approach: represent the polygon by 8 points on a circle
        with non-uniform angular spacing. The edge lengths are then determined by
        the angular gaps. But we want to optimize edge lengths directly.
        
        Better approach: Use the fact that any convex polygon can be represented
        by its vertices in order. We'll parameterize by:
        - 7 turning angles (the 8th is determined by sum = 2*pi for convexity)
        - Scale to fix perimeter
        
        Actually, let's use a cleaner parameterization:
        Represent the polygon by 8 unit vectors (directions) with angles theta_0, ..., theta_7
        where theta_i are in increasing order and the polygon is the intersection of half-planes.
        
        Simpler: Just use 8 points in CCW order, parameterized by 8 angles on a unit circle
        (non-uniform), then scale. The edge lengths are determined by the angular gaps.
        
        Let's use: 8 angles alpha_0 < alpha_1 < ... < alpha_7 < alpha_0 + 2*pi
        Points: (R*cos(alpha_i), R*sin(alpha_i)) for i=0..7
        Edge i has length 2*R*sin((alpha_{i+1} - alpha_i)/2) where indices mod 8
        
        We optimize the 8 angular gaps (sum = 2*pi), so 7 free parameters.
        """
        # edges: 7 free parameters (the 8th is 2*pi - sum of first 7)
        # Ensure all gaps are positive
        gaps = np.abs(edges[:7]) + 0.01  # Ensure positive
        gap8 = 2 * np.pi - np.sum(gaps)
        if gap8 <= 0.01:
            # Renormalize to ensure all positive
            gaps = gaps / np.sum(gaps) * (2 * np.pi - 0.08)
            gap8 = 2 * np.pi - np.sum(gaps)
        
        all_gaps = np.append(gaps, gap8)
        
        # Cumulative angles
        alphas = np.cumsum(np.concatenate([[0], all_gaps[:7]]))
        # alphas has 8 elements: 0, g0, g0+g1, ..., sum(g0..g6)
        
        # Points on unit circle
        pts = np.column_stack([np.cos(alphas), np.sin(alphas)])
        
        # Scale so that diameter = 2
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D < 1e-12:
            D = 1.0
        pts = pts * (2.0 / D)
        
        return pts

    # Objective in edge-length (angular gap) space
    def objective_edge_space(x):
        """
        x: 7-dimensional vector representing 7 of the 8 angular gaps.
        Returns negative log product of pairwise distances (to minimize).
        """
        pts = build_points_from_edges(x)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        unique_dists = dists[mask]
        
        # Penalize very small distances
        min_d = np.min(unique_dists)
        if min_d < 1e-6:
            return 1e10
        
        return -np.sum(np.log(unique_dists))

    # Initial guess: regular octagon has equal gaps of 2*pi/8 = pi/4
    initial_gaps = np.full(7, np.pi / 4)
    
    # Phase 1: Differential Evolution in 7D edge-space
    if time.time() < deadline - 30:
        # Bounds for each gap: [0.05, 2*pi - 7*0.05] approximately
        # But we need sum < 2*pi, so each gap < 2*pi - 7*min_gap
        min_gap = 0.05
        max_gap = 2 * np.pi - 7 * min_gap
        bounds = [(min_gap, max_gap)] * 7
        
        try:
            de_res = differential_evolution(
                objective_edge_space,
                bounds=bounds,
                maxiter=300,
                tol=1e-10,
                seed=42,
                popsize=30,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=False
            )
            pts = build_points_from_edges(de_res.x)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # Phase 2: Local refinement with L-BFGS-B from multiple starting points
    start_points = [np.full(7, np.pi / 4)]  # Regular octagon
    
    # Add DE result if available
    if 'de_res' in locals() and de_res.success:
        start_points.append(de_res.x.copy())
    
    # Add perturbed versions
    rng = np.random.RandomState(42)
    for trial in range(15):
        if time.time() > deadline - 15:
            break
        perturb = rng.normal(0, 0.2, size=7)
        candidate = np.full(7, np.pi / 4) + perturb
        # Ensure positivity
        candidate = np.maximum(candidate, 0.05)
        start_points.append(candidate)
    
    # Also add some structured perturbations
    # Symmetry-breaking patterns
    patterns = [
        [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
        [1.2, 0.9, 1.2, 0.9, 1.2, 0.9, 1.2],
        [1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 0.7],
        [1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1],
        [1.3, 0.7, 1.0, 1.0, 1.3, 0.7, 1.0],
    ]
    for pat in patterns:
        if time.time() > deadline - 15:
            break
        scaled = np.array(pat) / np.sum(pat) * 7 * (np.pi / 4)
        start_points.append(scaled)
    
    for sp in start_points:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                objective_edge_space,
                sp,
                method='L-BFGS-B',
                bounds=[(0.05, 2*np.pi - 7*0.05)] * 7,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            if res.success:
                pts = build_points_from_edges(res.x)
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass

    # Phase 3: Try a different parameterization - direct 16D optimization
    # as a fallback to compare
    def objective_16d(x):
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        unique_dists = dists[mask]
        min_d = np.min(unique_dists)
        if min_d < 1e-4:
            return 1e10
        D = np.max(dists)
        if D > 2.0:
            return -np.sum(np.log(unique_dists)) + 1000.0 * (D - 2.0)**2
        return -np.sum(np.log(unique_dists))

    if time.time() < deadline - 10:
        # Quick DE in 16D from the best points found
        x0 = best_points.flatten()
        bounds_16d = [(-2.0, 2.0)] * (2 * n)
        try:
            de_res_16d = differential_evolution(
                objective_16d,
                bounds=bounds_16d,
                maxiter=100,
                tol=1e-8,
                seed=123,
                popsize=15,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=False
            )
            pts = de_res_16d.x.reshape(n, 2)
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            if np.max(dists) <= 2.0 + 1e-6:
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass

    return best_points.tolist()
