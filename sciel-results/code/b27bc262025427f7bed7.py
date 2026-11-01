import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices once
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

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def soft_min_score(x, tau=0.01):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    """
    points = x.reshape((N, 2))
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
    
    areas = np.maximum(areas, 1e-12)
        
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def ring_configuration(params):
    """
    Generate points from ring parameters.
    params: [r1, r2, r3, r4, theta1, theta2, theta3, theta4, n1, n2, n3, n4]
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    
    # Round n's to integers, ensuring they sum to 14
    n1_i = int(round(n1))
    n2_i = int(round(n2))
    n3_i = int(round(n3))
    n4_i = int(round(n4))
    
    # Adjust to sum to 14
    total = n1_i + n2_i + n3_i + n4_i
    diff = 14 - total
    
    # Simple adjustment: add/subtract from the largest/smallest or just n4
    # To keep it simple and robust, we adjust n4 primarily, or distribute
    if diff != 0:
        # Distribute diff evenly or to n4
        # Let's just adjust n4 to keep the logic simple, but ensure n4 >= 1
        n4_i += diff
        if n4_i < 1:
            # If n4 becomes invalid, shift to n3
            n3_i += (1 - n4_i)
            n4_i = 1
            # Re-check sum? This is getting messy. 
            # Better approach: Use a proper integer partition or just fix n1, n2, n3 and let n4 = 14 - n1 - n2 - n3.
            # But DE searches all 4. Let's stick to a simpler constraint handling.
            # We will normalize the counts.
            pass

    # Robust count assignment:
    # Take the raw floats, round them, then fix the sum.
    counts = [n1_i, n2_i, n3_i, n4_i]
    # Ensure no negatives
    for i in range(4):
        if counts[i] < 1:
            counts[i] = 1
    
    # Adjust sum to 14
    while sum(counts) < 14:
        # Add to the one with the largest fractional part originally? Or just round-robin
        # For simplicity, add to n4, then n3, etc.
        idx = 3
        while idx >= 0 and counts[idx] >= 10: # Avoid making one ring too big? No, just add
            idx -= 1
        if idx < 0:
            counts[0] += 1
        else:
            counts[idx] += 1
            
    while sum(counts) > 14:
        idx = 3
        while idx >= 0 and counts[idx] <= 1:
            idx -= 1
        if idx < 0:
            counts[0] -= 1
        else:
            counts[idx] -= 1
            
    n1, n2, n3, n4 = counts

    points = []
    
    # Ring 1 (outermost)
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    
    # Ring 2
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    
    # Ring 3
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    
    # Ring 4 (innermost)
    if n4 > 0:
        angles = th4 + np.linspace(0, 2*np.pi, n4, endpoint=False)
        points.append(np.column_stack((r4*np.cos(angles), r4*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    # Pad or trim to exactly N points if needed (should be exactly N)
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    
    return result

def neg_soft_min_ring_score(params):
    try:
        points = ring_configuration(params)
        return soft_min_score(points.flatten(), tau=0.05)
    except Exception:
        return 1.0 # Large penalty

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(12345)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # --- Phase 1: Global Search on Flexible Ring Parameters ---
    
    # Parameters: r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4
    bounds = [
        (0.3, 1.5),  # r1
        (0.1, 1.0),  # r2
        (0.0, 0.7),  # r3
        (0.0, 0.4),  # r4
        (0, 2*np.pi), # th1
        (0, 2*np.pi), # th2
        (0, 2*np.pi), # th3
        (0, 2*np.pi), # th4
        (1, 13),  # n1
        (1, 13),  # n2
        (1, 13),  # n3
        (1, 13),  # n4
    ]
    
    try:
        res_de = differential_evolution(
            neg_soft_min_ring_score,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            popsize=15,
            mutation=(0.3, 1.2),
            recombination=0.9,
            polish=True,
            init='sobol'
        )
        
        if res_de.fun < 0: # Valid score found
            best_params = res_de.x
            points = ring_configuration(best_params)
            try_update(points)
    except Exception:
        pass

    # If DE didn't find a good result, try a few random ring configs
    if best_score < 40.0:
        for _ in range(10):
            if time.time() - start_time > deadline - 20:
                break
            params = [
                rng.uniform(0.8, 1.2), # r1
                rng.uniform(0.4, 0.8), # r2
                rng.uniform(0.1, 0.5), # r3
                rng.uniform(0.0, 0.2), # r4
                rng.uniform(0, 2*np.pi), # th1
                rng.uniform(0, 2*np.pi), # th2
                rng.uniform(0, 2*np.pi), # th3
                rng.uniform(0, 2*np.pi), # th4
                rng.integers(3, 8), # n1
                rng.integers(3, 8), # n2
                rng.integers(2, 6), # n3
                0 # placeholder, will be fixed in ring_config
            ]
            # Force sum to 14 in a simple way for random init
            n1 = params[8]
            n2 = params[9]
            n3 = params[10]
            n4 = 14 - n1 - n2 - n3
            if n4 < 1:
                n1 = 5; n2 = 5; n3 = 3; n4 = 1
            params[8], params[9], params[10], params[11] = n1, n2, n3, n4
            points = ring_configuration(params)
            try_update(points)

    # --- Phase 2: Local Refinement ---
    
    if best_points is not None and time.time() - start_time < deadline - 10:
        
        # 2a. L-BFGS-B on soft min (smooth)
        x0 = best_points.flatten()
        try:
            res_lbfgs = minimize(
                lambda x: soft_min_score(x, tau=0.01), 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            try_update(res_lbfgs.x.reshape((N, 2)))
        except Exception:
            pass
        
        # 2b. Nelder-Mead on hard min (precise)
        if time.time() - start_time < deadline - 5:
            x0 = best_points.flatten()
            try:
                res_nm = minimize(
                    neg_score, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                try_update(res_nm.x.reshape((N, 2)))
            except Exception:
                pass
        
        # 2c. Powell on hard min (final polish)
        if time.time() - start_time < deadline - 2:
            x0 = best_points.flatten()
            try:
                res_powell = minimize(
                    neg_score, 
                    x0, 
                    method='Powell',
                    options={'maxiter': 300, 'xtol': 1e-9, 'ftol': 1e-11}
                )
                try_update(res_powell.x.reshape((N, 2)))
            except Exception:
                pass

    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
