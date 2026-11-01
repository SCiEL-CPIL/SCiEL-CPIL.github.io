import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time

N = 14

# Precompute triangle indices for vectorized area calculation
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def soft_min_score(points, beta=50.0):
    """Smooth surrogate using log-sum-exp for the min triangle area."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * areas))) / beta approximates min
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    
    # Use soft min for smoothness
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return soft_min / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def neg_soft_score(x, beta=50.0):
    points = x.reshape((N, 2))
    return -soft_min_score(points, beta)

def ring_configuration(params):
    """
    Generate points from ring parameters.
    params: [r1, r2, r3, r4, theta_offset1, theta_offset2, theta_offset3, theta_offset4,
             n1, n2, n3, n4] where n1+n2+n3+n4=14, r_i >= 0, theta offsets in [0, 2pi)
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    # Ensure we have exactly 14 points with at least 1 per ring
    # First clamp each to [1, 14]
    n1 = max(1, min(14, n1))
    n2 = max(1, min(14, n2))
    n3 = max(1, min(14, n3))
    n4 = max(1, min(14, n4))
    
    # Adjust to sum to exactly 14
    total = n1 + n2 + n3 + n4
    while total > 14:
        # Reduce the largest ring
        max_idx = np.argmax([n1, n2, n3, n4])
        if max_idx == 0:
            n1 -= 1
        elif max_idx == 1:
            n2 -= 1
        elif max_idx == 2:
            n3 -= 1
        else:
            n4 -= 1
        total -= 1
    while total < 14:
        # Add to the smallest ring
        min_idx = np.argmin([n1, n2, n3, n4])
        if min_idx == 0:
            n1 += 1
        elif min_idx == 1:
            n2 += 1
        elif min_idx == 2:
            n3 += 1
        else:
            n4 += 1
        total += 1
    
    # Ensure minimum of 1 per ring
    n1 = max(1, n1)
    n2 = max(1, n2)
    n3 = max(1, n3)
    n4 = max(1, n4)
    
    # Final adjustment if still not 14
    total = n1 + n2 + n3 + n4
    if total != 14:
        # Fallback: use 4+4+3+3
        n1, n2, n3, n4 = 4, 4, 3, 3
    
    points = []
    
    # Ring 1 (outermost)
    angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
    points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    
    # Ring 2
    angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
    points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    
    # Ring 3
    angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
    points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    
    # Ring 4 (innermost)
    angles = th4 + np.linspace(0, 2*np.pi, n4, endpoint=False)
    points.append(np.column_stack((r4*np.cos(angles), r4*np.sin(angles))))
    
    result = np.vstack(points)
    
    # Should be exactly N points now
    assert len(result) == N, f"Expected {N} points, got {len(result)}"
    
    return result

def neg_ring_score(params):
    """Score for ring parameterization (negative for minimization)."""
    try:
        points = ring_configuration(params)
        return -compute_score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s
    
    # Phase 1: Differential Evolution on 4-ring parameters
    # Parameters: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    bounds = [
        (0.3, 1.5),  # r1 (outermost)
        (0.1, 1.0),  # r2
        (0.0, 0.7),  # r3
        (0.0, 0.4),  # r4 (innermost)
        (0, 2*np.pi),  # th1
        (0, 2*np.pi),  # th2
        (0, 2*np.pi),  # th3
        (0, 2*np.pi),  # th4
        (1, 14),  # n1
        (1, 14),  # n2
        (1, 14),  # n3
        (1, 14),  # n4
    ]
    
    # Try differential evolution with multiple restarts
    for restart in range(3):
        if time.time() > deadline - 50:
            break
        
        try:
            res = differential_evolution(
                neg_ring_score,
                bounds,
                seed=rng.integers(0, 2**31),
                maxiter=80,
                tol=1e-6,
                popsize=12,
                mutation=(0.3, 1.2),
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            
            if res.success or res.fun < 0:
                best_params = res.x
                points = ring_configuration(best_params)
                try_update(points)
        except Exception:
            pass
    
    # Phase 2: Multi-start Nelder-Mead on full 28D space from good ring configs
    if best_points is not None and time.time() < deadline - 30:
        # Generate diverse starting points
        starts = [best_points]
        
        # Add jittered versions with varying magnitudes
        for sigma in [0.05, 0.1, 0.15]:
            jittered = best_points + rng.normal(0, sigma, (N, 2))
            starts.append(jittered)
        
        # Add some structured starts
        # Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(np.column_stack((np.cos(angles), np.sin(angles))))
        
        # Two rings 7+7
        a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a_in = a_out + np.pi/7
        starts.append(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
        )))
        
        # Three rings 5+5+4
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = a1 + np.pi/5
        a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6*np.cos(a2), 0.6*np.sin(a2))),
            np.column_stack((0.25*np.cos(a3), 0.25*np.sin(a3)))
        )))
        
        # Four rings 4+4+3+3
        a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        a2 = a1 + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
        a4 = a3 + np.pi/3
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.65*np.cos(a2), 0.65*np.sin(a2))),
            np.column_stack((0.35*np.cos(a3), 0.35*np.sin(a3))),
            np.column_stack((0.1*np.cos(a4), 0.1*np.sin(a4)))
        )))
        
        # Five rings 3+3+3+3+2
        a1 = np.linspace(0, 2*np.pi, 3, endpoint=False)
        a2 = a1 + np.pi/3
        a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/6
        a4 = a3 + np.pi/3
        a5 = np.linspace(0, 2*np.pi, 2, endpoint=False) + np.pi/4
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.75*np.cos(a2), 0.75*np.sin(a2))),
            np.column_stack((0.5*np.cos(a3), 0.5*np.sin(a3))),
            np.column_stack((0.3*np.cos(a4), 0.3*np.sin(a4))),
            np.column_stack((0.1*np.cos(a5), 0.1*np.sin(a5)))
        )))
        
        for start in starts:
            if time.time() > deadline - 10:
                break
            
            x0 = start.flatten()
            
            try:
                # Use Nelder-Mead for robustness
                res = minimize(
                    neg_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7, 'disp': False}
                )
                
                if res.fun < -best_score:
                    pts = res.x.reshape((N, 2))
                    try_update(pts)
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish with soft min surrogate
    if best_points is not None and time.time() < deadline - 5:
        try:
            x0 = best_points.flatten()
            # Use soft min for smoother landscape in final polish
            res = minimize(
                lambda x: neg_soft_score(x, beta=100.0),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'disp': False}
            )
            
            if res.fun < -best_score:
                pts = res.x.reshape((N, 2))
                try_update(pts)
        except Exception:
            pass
    
    # Phase 4: Ultra-precise final polish on exact objective
    if best_points is not None and time.time() < deadline - 2:
        try:
            x0 = best_points.flatten()
            res = minimize(
                neg_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
            )
            
            if res.fun < -best_score:
                pts = res.x.reshape((N, 2))
                try_update(pts)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
