import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# Precompute triangle indices for vectorized area calculation
def _precompute_triangles(n):
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]

TRI_I, TRI_J, TRI_K = _precompute_triangles(N)
NUM_TRIS = len(TRI_I)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0
    
    # Check for degenerate hull
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    a = points[TRI_I]
    b = points[TRI_J]
    c = points[TRI_K]
    
    # Vectorized area computation: 0.5 * |cross(b-a, c-a)|
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(points):
    return -get_score(points)

def _generate_structured_configs():
    """
    Generate a list of highly structured, symmetric initial configurations.
    These are much better starting points than random for this problem.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric Rings (Outer k, Inner N-k)
    # Try various splits and radii/offsets
    
    # 7 Outer, 6 Inner
    for r_inner in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for offset in [0, np.pi/6, np.pi/7]:
            angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
            pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
            angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False) + offset
            pts_in = np.column_stack([r_inner*np.cos(angles_in), r_inner*np.sin(angles_in)])
            configs.append(np.vstack([pts_out, pts_in]))
            
    # 6 Outer, 6 Inner, 1 Center
    for r_inner in [0.5, 0.6, 0.7]:
        for offset in [0, np.pi/6]:
            angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False)
            pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
            angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False) + offset
            pts_in = np.column_stack([r_inner*np.cos(angles_in), r_inner*np.sin(angles_in)])
            center = np.array([[0.0, 0.0]])
            configs.append(np.vstack([pts_out, pts_in, center]))

    # 5 Outer, 5 Inner, 3 Innermost
    for r1, r2 in [(0.7, 0.4), (0.8, 0.5), (0.6, 0.3)]:
        for off1, off2 in [(0, 0), (np.pi/5, np.pi/3)]:
            angles_out = np.linspace(0, 2*np.pi, 5, endpoint=False)
            pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
            angles_mid = np.linspace(0, 2*np.pi, 5, endpoint=False) + off1
            pts_mid = np.column_stack([r1*np.cos(angles_mid), r1*np.sin(angles_mid)])
            angles_in = np.linspace(0, 2*np.pi, 3, endpoint=False) + off2
            pts_in = np.column_stack([r2*np.cos(angles_in), r2*np.sin(angles_in)])
            configs.append(np.vstack([pts_out, pts_mid, pts_in]))

    # 8 Outer, 5 Inner
    for r_inner in [0.5, 0.6, 0.7]:
        for offset in [0, np.pi/8]:
            angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False)
            pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
            angles_in = np.linspace(0, 2*np.pi, 5, endpoint=False) + offset
            pts_in = np.column_stack([r_inner*np.cos(angles_in), r_inner*np.sin(angles_in)])
            configs.append(np.vstack([pts_out, pts_in]))

    # 4 Outer, 9 Inner (Square + nonagon)
    for r_inner in [0.6, 0.7, 0.8]:
        angles_out = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
        angles_in = np.linspace(0, 2*np.pi, 9, endpoint=False)
        pts_in = np.column_stack([r_inner*np.cos(angles_in), r_inner*np.sin(angles_in)])
        configs.append(np.vstack([pts_out, pts_in]))

    # 3 Outer, 10 Inner (Triangle + decagon)
    for r_inner in [0.6, 0.7, 0.8]:
        angles_out = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/6
        pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
        angles_in = np.linspace(0, 2*np.pi, 10, endpoint=False)
        pts_in = np.column_stack([r_inner*np.cos(angles_in), r_inner*np.sin(angles_in)])
        configs.append(np.vstack([pts_out, pts_in]))

    # 12 Outer, 1 Center
    angles_out = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    center = np.array([[0.0, 0.0]])
    configs.append(np.vstack([pts_out, center]))

    # 11 Outer, 2 Inner (opposite)
    for r_inner in [0.4, 0.5, 0.6]:
        angles_out = np.linspace(0, 2*np.pi, 11, endpoint=False)
        pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
        # Two inner points opposite each other
        p1 = np.array([[r_inner, 0.0]])
        p2 = np.array([[-r_inner, 0.0]])
        configs.append(np.vstack([pts_out, p1, p2]))
        
    # 10 Outer, 3 Inner (triangle)
    for r_inner in [0.4, 0.5, 0.6]:
        angles_out = np.linspace(0, 2*np.pi, 10, endpoint=False)
        pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
        angles_in = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
        pts_in = np.column_stack([r_inner*np.cos(angles_in), r_inner*np.sin(angles_in)])
        configs.append(np.vstack([pts_out, pts_in]))

    return configs

def _local_polish(points, deadline, max_iters=500, step_size=0.05):
    """
    Perform a local search polish using random perturbations.
    """
    best_points = points.copy()
    best_score = get_score(best_points)
    
    rng = np.random.default_rng(42)
    current_step = step_size
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        # Generate a random perturbation
        noise = rng.normal(0, current_step, points.shape)
        new_points = points + noise
        
        new_score = get_score(new_points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            # Slightly increase step if we are improving rapidly? Or keep constant?
            # Let's decay step size slowly to converge
            current_step = max(current_step * 0.99, 0.001)
        else:
            # If no improvement, reduce step size
            current_step = max(current_step * 0.95, 0.001)
            
        points = new_points
        
    return best_points, best_score

def _optimize_config(init_points, deadline, max_iter=500):
    """
    Run Nelder-Mead optimization on a single configuration.
    """
    start_time = time.time()
    
    # Nelder-Mead options
    options = {
        'maxiter': max_iter,
        'xatol': 1e-5,
        'fatol': 1e-6,
        'adaptive': True
    }
    
    try:
        result = minimize(
            _neg_score, 
            init_points, 
            method='Nelder-Mead', 
            options=options
        )
    except Exception:
        return init_points, get_score(init_points)
    
    best_points = result.x
    best_score = -result.fun
    
    # If Nelder-Mead finished early or with room to spare, do local polish
    if time.time() < deadline - 2.0:
        polished_points, polished_score = _local_polish(best_points, deadline, max_iters=200, step_size=0.02)
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score
            
    return best_points, best_score

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds
    deadline = start_time + 110.0
    
    configs = _generate_structured_configs()
    
    best_score = -np.inf
    best_points = None
    
    # Sort configs by initial score to prioritize good starts
    # This is cheap and helps focus budget on promising areas
    scored_configs = []
    for c in configs:
        s = get_score(c)
        scored_configs.append((s, c))
    
    scored_configs.sort(key=lambda x: x[0], reverse=True)
    
    # Allocate budget
    # We have ~110 seconds.
    # Nelder-Mead can take a few seconds per run.
    # Local polish takes ~1-2 seconds.
    # Let's try to optimize the top 10-15 configurations.
    
    num_to_optimize = min(len(scored_configs), 15)
    
    for i in range(num_to_optimize):
        if time.time() > deadline - 5.0:
            break
            
        init_score, init_points = scored_configs[i]
        
        # If the initial score is already very high, maybe spend less time?
        # Or just standard budget.
        
        # Allocate remaining time to this optimization
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
            
        # Reduce max_iter if time is tight
        max_iter = 200 if time_left < 20 else 400
        
        try:
            optimized_points, score = _optimize_config(init_points, deadline - 2.0, max_iter=max_iter)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points
        except Exception:
            # Fallback to initial if optimization fails
            if init_score > best_score:
                best_score = init_score
                best_points = init_points

    # If we have significant time left, do a final polish on the global best
    if best_points is not None and time.time() < deadline - 10.0:
        final_polished, final_score = _local_polish(best_points, deadline - 2.0, max_iters=1000, step_size=0.01)
        if final_score > best_score:
            best_points = final_polished
            best_score = final_score

    # Fallback if nothing worked (should not happen)
    if best_points is None:
        best_points = configs[0]
        
    return best_points

# For debugging/testing
if __name__ == "__main__":
    pts = construct()
    print("Score:", get_score(pts))
