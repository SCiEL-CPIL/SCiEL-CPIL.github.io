import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _normalize_batch(pts):
    """Normalize batch of vectors."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_hull(config):
    """
    Compute covering radius in degrees using convex hull facets.
    Returns (radius_deg, circumcenters_array, facet_indices_array)
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([], dtype=int)
    
    simplices = hull.simplices
    n_facets = len(simplices)
    
    # Vectorized circumcenter computation
    p0 = config[simplices[:, 0]]  # (F, 3)
    p1 = config[simplices[:, 1]]
    p2 = config[simplices[:, 2]]
    
    v1 = p1 - p0  # (F, 3)
    v2 = p2 - p0  # (F, 3)
    
    # Cross product for each facet
    c = np.cross(v1, v2)  # (F, 3)
    c_norms = np.linalg.norm(c, axis=1)
    
    # Filter out degenerate facets
    valid_mask = c_norms > 1e-12
    if not np.any(valid_mask):
        return 0.0, np.array([]), np.array([], dtype=int)
    
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    valid_facets = simplices[valid_mask]
    
    # Check if circumcenter is equidistant from all 3 vertices
    dots0 = np.einsum('ij,ij->i', c, p0[valid_mask])
    dots1 = np.einsum('ij,ij->i', c, p1[valid_mask])
    dots2 = np.einsum('ij,ij->i', c, p2[valid_mask])
    
    dots = np.column_stack([dots0, dots1, dots2])  # (F, 3)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (F, 3)
    
    # Valid circumcenters: all three angles should be approximately equal
    angle_spread = np.max(angles, axis=1) - np.min(angles, axis=1)
    valid_circ = angle_spread < 1e-6
    
    if not np.any(valid_circ):
        grid = _fib_sample(10000)
        dots_grid = grid @ config.T
        min_dots = np.min(dots_grid, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([], dtype=int)
    
    c = c[valid_circ]
    valid_facets = valid_facets[valid_circ]
    
    # For each valid circumcenter, find nearest config point
    dots_all = c @ config.T  # (F, N)
    dots_all = np.clip(dots_all, -1.0, 1.0)
    max_dots = np.max(dots_all, axis=1)  # (F,)
    min_angles = np.arccos(max_dots)  # (F,)
    
    max_angle = np.max(min_angles)
    radius_deg = np.degrees(max_angle)
    
    return radius_deg, c, valid_facets

def _compute_forces(config, beta=50.0):
    """
    Compute smooth force on each point:
    - Repulsion from other points (inverse square of angular distance)
    - Attraction toward worst circumcenters (smooth max via log-sum-exp)
    """
    F = np.zeros_like(config)
    
    # Pairwise repulsion
    for i in range(N):
        dots = np.clip(config[i] @ config.T, -1.0, 1.0)
        angles = np.arccos(dots)
        # Avoid division by zero for self
        angles[i] = np.pi  # large angle for self
        # Repulsive force: direction away from other points, magnitude ~ 1/angle^2
        for j in range(N):
            if i == j:
                continue
            ang = angles[j]
            if ang < 1e-6:
                continue
            # Force magnitude: stronger when closer
            mag = 1.0 / (ang * ang)
            # Direction: away from point j (tangent component)
            v = config[i] - config[j]
            v_proj = v - np.dot(v, config[i]) * config[i]
            vn = np.linalg.norm(v_proj)
            if vn > 1e-10:
                F[i] += mag * v_proj / vn
    
    # Attraction toward worst circumcenters
    score, circumcenters, _ = _covering_radius_hull(config)
    if len(circumcenters) == 0:
        return F
    
    dots_all = circumcenters @ config.T  # (F, N)
    dots_all = np.clip(dots_all, -1.0, 1.0)
    angles_all = np.arccos(dots_all)  # (F, N)
    
    # Smooth max over facets for each point: -log-sum-exp(-beta * angles)
    # We want to attract points toward circumcenters with large angles
    # Force toward circumcenter c: tangent component of (c - config[i])
    
    for i in range(N):
        # Weighted sum of circumcenters based on how "worst" they are for this point
        # Use softmax over angles
        logits = -beta * angles_all[:, i]  # (F,)
        logits -= np.max(logits)  # for numerical stability
        weights = np.exp(logits)
        weights /= np.sum(weights)
        
        # Weighted circumcenter
        c_weighted = weights @ circumcenters  # (3,)
        
        # Tangent direction from config[i] toward c_weighted
        v = c_weighted - config[i]
        v_proj = v - np.dot(v, config[i]) * config[i]
        vn = np.linalg.norm(v_proj)
        if vn > 1e-10:
            F[i] += 0.5 * v_proj / vn  # attraction strength
    
    return F

def _force_based_optimize(config, deadline, min_time_remaining=10.0):
    """
    Optimize using force-based method with adaptive step size.
    """
    config = config.copy()
    current_score, _, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.02
    beta = 50.0
    stagnation = 0
    max_stagnation = 50
    
    while time.time() < deadline - min_time_remaining:
        score, _, _ = _covering_radius_hull(config)
        current_score = score
        
        if current_score < best_score - 1e-10:
            best_score = current_score
            best_config = config.copy()
            stagnation = 0
        else:
            stagnation += 1
        
        if stagnation > max_stagnation:
            break
        
        # Compute forces
        F = _compute_forces(config, beta)
        
        # Normalize forces
        F_norms = np.linalg.norm(F, axis=1, keepdims=True)
        F_norms = np.where(F_norms < 1e-12, 1.0, F_norms)
        F = F / F_norms
        
        # Apply forces with step size
        new_config = config + step_size * F
        new_config = _normalize_batch(new_config)
        
        # Check if improved
        new_score, _, _ = _covering_radius_hull(new_config)
        
        if new_score < current_score - 1e-9:
            config = new_config
            # Adjust step size
            if new_score < current_score - 0.01:
                step_size = min(step_size * 1.1, 0.05)
            else:
                step_size *= 0.95
        else:
            # Try smaller step
            for ss_factor in [0.5, 0.25, 0.1]:
                new_config = config + step_size * ss_factor * F
                new_config = _normalize_batch(new_config)
                new_score, _, _ = _covering_radius_hull(new_config)
                if new_score < current_score - 1e-9:
                    config = new_config
                    step_size *= 0.8
                    break
            else:
                step_size *= 0.7
                if step_size < 0.001:
                    break
    
    return best_config

def _local_optimize(config, deadline, min_time_remaining=10.0):
    """
    Local optimization using circumcenter-based critical points.
    """
    config = config.copy()
    current_score, _, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.04
    stagnation = 0
    max_stagnation = 30
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters, facets = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        # Find the worst circumcenter
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        worst_idx = np.argmax(min_angles)
        worst_point = circumcenters[worst_idx]
        worst_angle = min_angles[worst_idx]
        
        # Find candidate points to move: the 3 nearest config points to worst_point
        dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
        sorted_idx = np.argsort(-dots_to_worst)
        
        improved = False
        
        for trial in range(3):
            if trial >= len(sorted_idx):
                break
            idx = sorted_idx[trial]
            
            # Compute tangent direction toward worst_point
            v = worst_point - config[idx]
            v_proj = v - np.dot(v, config[idx]) * config[idx]
            v_norm = np.linalg.norm(v_proj)
            
            if v_norm < 1e-10:
                continue
            
            v_dir = v_proj / v_norm
            
            # Try a few step sizes
            for ss in [step_size, step_size * 0.5, step_size * 2.0]:
                new_pt = config[idx] + ss * v_dir
                new_pt = _normalize(new_pt)
                
                old_pt = config[idx].copy()
                config[idx] = new_pt
                
                new_score, _, _ = _covering_radius_hull(config)
                
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    stagnation = 0
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                    if new_score < current_score - 0.01:
                        step_size = min(step_size * 1.2, 0.1)
                    break
                else:
                    config[idx] = old_pt
            
            if improved:
                break
        
        if not improved:
            stagnation += 1
            step_size *= 0.85
            if step_size < 0.001 or stagnation > max_stagnation:
                break
    
    return best_config

def _polish(config, deadline, min_time_remaining=5.0):
    """
    High-precision polish pass with very small steps.
    """
    config = config.copy()
    current_score, _, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.005
    stagnation = 0
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters, facets = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        # Find all circumcenters with angles close to the maximum
        threshold = np.max(min_angles) - 0.001
        worst_mask = min_angles > threshold
        worst_indices = np.where(worst_mask)[0]
        
        if len(worst_indices) == 0:
            break
        
        improved = False
        
        for wi in worst_indices[:5]:
            worst_point = circumcenters[wi]
            
            dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
            sorted_idx = np.argsort(-dots_to_worst)
            
            for trial in range(2):
                if trial >= len(sorted_idx):
                    break
                idx = sorted_idx[trial]
                
                v = worst_point - config[idx]
                v_proj = v - np.dot(v, config[idx]) * config[idx]
                v_norm = np.linalg.norm(v_proj)
                
                if v_norm < 1e-10:
                    continue
                
                v_dir = v_proj / v_norm
                
                new_pt = config[idx] + step_size * v_dir
                new_pt = _normalize(new_pt)
                
                old_pt = config[idx].copy()
                config[idx] = new_pt
                
                new_score, _, _ = _covering_radius_hull(config)
                
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    stagnation = 0
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                    break
                else:
                    config[idx] = old_pt
            
            if improved:
                break
        
        if not improved:
            stagnation += 1
            step_size *= 0.9
            if step_size < 0.0005 or stagnation > 50:
                break
    
    return best_config

def _optimize(config, deadline):
    """
    Full optimization pipeline: Lloyd's + force-based + local optimization + polish.
    """
    config = config.copy()
    
    # Phase 1: Lloyd's algorithm for good initial configuration
    M = 10000
    grid = _fib_sample(M)
    
    for iteration in range(150):
        if time.time() > deadline - 50:
            break
        dots = grid @ config.T
        max_idx = np.argmax(dots, axis=1)
        
        new_config = np.zeros_like(config)
        for i in range(N):
            mask = (max_idx == i)
            count = np.sum(mask)
            if count > 0:
                center = np.mean(grid[mask], axis=0)
                new_config[i] = _normalize(center)
            else:
                new_config[i] = _normalize(np.random.randn(3))
        
        if np.allclose(config, new_config, atol=1e-8):
            break
        config = new_config
    
    # Phase 2: Force-based optimization
    if time.time() < deadline - 30:
        config = _force_based_optimize(config, deadline, min_time_remaining=20.0)
    
    # Phase 3: Local optimization
    if time.time() < deadline - 15:
        config = _local_optimize(config, deadline, min_time_remaining=10.0)
    
    # Phase 4: Polish
    if time.time() < deadline - 5:
        config = _polish(config, deadline, min_time_remaining=3.0)
    
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Run optimization
    config = _optimize(config, deadline)
    
    # Final normalization and validation
    config = _normalize_batch(config)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
    
    return config
