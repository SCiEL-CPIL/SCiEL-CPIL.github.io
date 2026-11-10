# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580

    def exact_score_from_pts(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def exact_objective(x):
        pts = x.reshape(n, d)
        return -exact_score_from_pts(pts)

    best_x = None
    best_score = 0.0

    def update_best(x, score):
        nonlocal best_x, best_score
        if score > best_score:
            best_score = score
            best_x = x.copy()

    # Generate 19-point hexagonal cluster
    def gen_hex_cluster(s=1.0):
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring 1 (radius s)
        for k in range(6):
            angle = k * np.pi / 3
            pts.append([s * np.cos(angle), s * np.sin(angle)])
        # Ring 2 (radius 2s)
        for k in range(12):
            angle = k * np.pi / 6
            # Only keep points that are at distance 2s from center and s from a ring-1 point
            # Actually, the 12 points at radius 2s with angles k*pi/6 form the hex cluster
            # But we only want the 6 that are at distance s from ring-1 points?
            # No, the full hex cluster of 19 has:
            # 1 center, 6 at radius 1, 12 at radius 2? No.
            # Hex cluster: 
            # Layer 0: (0,0)
            # Layer 1: 6 points at distance 1
            # Layer 2: 12 points at distance 2? No.
            # The hexagonal number sequence is 1, 7, 19, 37...
            # 19 = 1 + 6 + 12.
            # So layer 2 has 12 points.
            # These are at distance 2 from center?
            # Let's check: (2, 0) is at dist 2. (1, sqrt(3)) is at dist 2.
            # Yes, all 12 points in layer 2 are at distance 2 from center.
            pts.append([2 * s * np.cos(angle), 2 * s * np.sin(angle)])
        return np.array(pts)

    cluster_19 = gen_hex_cluster(1.0)

    # Define removal patterns: remove 3 points from the 19
    # Indices: 0=center, 1-6=ring1, 7-18=ring2
    # To maximize min distance, we should remove points that are "far apart" or create "holes"
    # But we want to minimize d_max, so we should remove outer points.
    # Removing 3 points from ring 2 (indices 7-18) seems best.
    # Which 3? Symmetric patterns.
    # Pattern A: remove 3 alternating points from ring 2 (every 4th? 12/3 = 4. 0, 4, 8 in ring 2 indices)
    # Ring 2 indices: 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
    # Angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    # Remove 0, 120, 240 -> indices 7, 11, 15
    # Remove 30, 150, 270 -> indices 8, 12, 16
    # Remove 60, 180, 300 -> indices 9, 13, 17
    # Remove 90, 210, 330 -> indices 10, 14, 18

    removal_patterns = [
        [7, 11, 15],   # 0, 120, 240
        [8, 12, 16],   # 30, 150, 270
        [9, 13, 17],   # 60, 180, 300
        [10, 14, 18],  # 90, 210, 330
        [7, 10, 13],   # 0, 90, 180 (less symmetric?)
        [7, 9, 11],    # 0, 60, 120
        [7, 8, 13],    # 0, 30, 180
        [7, 11, 14],   # 0, 120, 210
    ]

    initial_guesses = []
    for rem in removal_patterns:
        mask = np.ones(19, dtype=bool)
        mask[rem] = False
        pts = cluster_19[mask]
        # Center
        pts = pts - np.mean(pts, axis=0)
        initial_guesses.append(pts)

    # Also add a few random perturbations of the first pattern
    base_pts = initial_guesses[0].copy()
    for _ in range(4):
        perturbed = base_pts + np.random.normal(0, 0.05, (16, 2))
        initial_guesses.append(perturbed)

    # Phase 1: DE on free coordinates for each initial guess
    # This is expensive: 12 initial guesses * DE
    # Let's do a smaller DE first, then Nelder-Mead
    # Or just Nelder-Mead on each initial guess, then DE on the best.
    
    # Let's do Nelder-Mead on each initial guess
    for i, init_pts in enumerate(initial_guesses):
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                exact_objective,
                init_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts = res.x.reshape(n, d)
            score = exact_score_from_pts(pts)
            update_best(res.x, score)
        except Exception:
            pass

    # Phase 2: DE on free coordinates, starting from best
    if best_x is not None and time.time() < deadline - 120:
        try:
            # DE needs bounds. Use a reasonable box around the best solution.
            # Or just use a large box and let it converge.
            # Let's use a box based on the spread of the best solution.
            best_pts = best_x.reshape(n, d)
            center = np.mean(best_pts, axis=0)
            spread = np.max(np.abs(best_pts - center)) + 0.5
            bounds = [(center[i] - spread, center[i] + spread) for i in range(n * d)]
            
            res_de = differential_evolution(
                exact_objective,
                bounds,
                maxiter=300,
                popsize=30,
                seed=42,
                tol=1e-10,
                init='latinhypercube',
                polish=False,
                updating='immediate'
            )
            pts = res_de.x.reshape(n, d)
            score = exact_score_from_pts(pts)
            update_best(res_de.x, score)
        except Exception:
            pass

    # Phase 3: Final Nelder-Mead polish
    if best_x is not None and time.time() < deadline - 30:
        try:
            res = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            pts = res.x.reshape(n, d)
            score = exact_score_from_pts(pts)
            update_best(res.x, score)
        except Exception:
            pass

    # Fallback
    if best_x is None:
        best_x = initial_guesses[0].ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
