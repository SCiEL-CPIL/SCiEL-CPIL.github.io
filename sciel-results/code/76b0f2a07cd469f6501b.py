import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
MIN_DIST = math.sqrt(3) # Distance between centers of touching hexagons
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def axial_to_cartesian(axial):
    x = math.sqrt(3) * (axial[0] + axial[1] / 2.0)
    y = 1.5 * axial[1]
    return np.array([x, y])

def get_initial_configs():
    configs = []
    
    # Config 1: 3x3 block + 2 on right
    # Axial: (0,0) to (2,2) is 9. Add (3,1), (3,2)?
    # Let's use a helper to generate blocks
    def make_block(w, h):
        pts = []
        for q in range(w):
            for r in range(h):
                pts.append((q, r))
        return pts

    # 3x3 + 2
    c1 = make_block(3, 3) + [(3, 1), (3, 2)]
    configs.append(c1)
    
    # 4x3 - 1 (remove corner)
    c2 = make_block(4, 3)
    del c2[0] # Remove (0,0)
    configs.append(c2)
    
    # 2x5 + 1
    c3 = make_block(2, 5) + [(1, 5)]
    configs.append(c3)
    
    # "Cross" shape?
    # Center (0,0), 6 neighbors, + 4 more
    c4 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
          (2,0), (-2,0), (0,2), (0,-2)]
    configs.append(c4)
    
    # 5x3 - 4
    c5 = make_block(5, 3)
    del c5[0]
    del c5[14] # roughly corners
    # Ensure length 11
    while len(c5) > 11: c5.pop()
    while len(c5) < 11: c5.append((0, -1)) # Fill
    configs.append(c5)
    
    # Convert to Cartesian
    cart_configs = []
    for c in configs:
        pts = [axial_to_cartesian(ax) for ax in c]
        cart_configs.append(np.array(pts))
        
    return cart_configs

def fast_min_enc_hex_side(pts):
    """
    Fast approximate min enclosing hexagon side length.
    Uses centroid as center and coarse angle sampling.
    """
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    best_S = float('inf')
    # Sample 12 angles in [0, pi/6]
    for i in range(13):
        phi = i * (math.pi / 6.0) / 12.0
        # Normals for outer hexagon
        angles = phi + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Projections
        projs = np.abs(centered @ normals.T) # (N, 6)
        max_proj = np.max(projs)
        S = (max_proj + APOTHEM) / APOTHEM
        if S < best_S:
            best_S = S
    return best_S

def check_overlap(pts):
    """
    Check if any two hexagons overlap.
    Returns True if valid (no overlap).
    """
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist < MIN_DIST - 1e-6:
                return False
    return True

def sa_optimize(initial_pts, n_steps=20000, t_start=0.5, t_end=0.01):
    """
    Simulated Annealing to optimize positions.
    """
    current_pts = initial_pts.copy()
    current_cost = fast_min_enc_hex_side(current_pts)
    if not check_overlap(current_pts):
        # Try to fix overlaps by spreading out
        # Simple heuristic: move apart
        # For now, return inf cost if invalid
        return None, float('inf')
        
    best_pts = current_pts.copy()
    best_cost = current_cost
    
    for step in range(n_steps):
        # Cooling schedule
        t = t_start * (t_end / t_start) ** (step / n_steps)
        
        # Pick a random hexagon to move
        idx = np.random.randint(0, N)
        
        # Random move
        move_angle = np.random.uniform(0, 2 * math.pi)
        move_dist = np.random.uniform(0, 0.2)
        dx = move_dist * math.cos(move_angle)
        dy = move_dist * math.sin(move_angle)
        
        new_pts = current_pts.copy()
        new_pts[idx] += np.array([dx, dy])
        
        # Check overlap
        if not check_overlap(new_pts):
            continue
            
        new_cost = fast_min_enc_hex_side(new_pts)
        
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < math.exp(-delta / t):
            current_pts = new_pts
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_pts = current_pts.copy()
                
    return best_pts, best_cost

def exact_min_enc_hex_side(pts):
    """
    More precise min enclosing hexagon side length.
    """
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    best_S = float('inf')
    # Sample 100 angles
    for i in range(101):
        phi = i * (math.pi / 6.0) / 100.0
        angles = phi + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = np.abs(centered @ normals.T)
        max_proj = np.max(projs)
        S = (max_proj + APOTHEM) / APOTHEM
        if S < best_S:
            best_S = S
    return best_S

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    np.random.seed(42)
    configs = get_initial_configs()
    
    best_global_pts = None
    best_global_cost = float('inf')
    
    # Run SA for each initial config
    for i, init_pts in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Check if init is valid
        if not check_overlap(init_pts):
            continue
            
        # Run SA
        # Time budget per config: 10s
        t_start_time = time.time()
        pts, cost = sa_optimize(init_pts, n_steps=20000)
        
        if pts is not None and cost < best_global_cost:
            best_global_cost = cost
            best_global_pts = pts
            
    if best_global_pts is None:
        # Fallback
        best_global_pts = configs[0]
        best_global_cost = fast_min_enc_hex_side(best_global_pts)

    # Final Polish with Nelder-Mead
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        def objective(flat_pts):
            pts = flat_pts.reshape(-1, 2)
            if not check_overlap(pts):
                return 100.0
            return exact_min_enc_hex_side(pts)
            
        init_flat = best_global_pts.flatten()
        res = minimize(objective, init_flat, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_global_cost:
            best_global_pts = res.x.reshape(-1, 2)
            best_global_cost = res.fun

    # Compute final result
    pts = best_global_pts
    # Find optimal rotation and center
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    best_S = float('inf')
    best_phi = 0.0
    for i in range(1001):
        phi = i * (math.pi / 6.0) / 1000.0
        angles = phi + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = np.abs(centered @ normals.T)
        max_proj = np.max(projs)
        S = (max_proj + APOTHEM) / APOTHEM
        if S < best_S:
            best_S = S
            best_phi = phi
            
    # Scale down slightly to ensure validity
    SCALE = 1.0 - 1e-6
    pts_scaled = centroid + (pts - centroid) * SCALE
    centroid_scaled = np.mean(pts_scaled, axis=0)
    centered_scaled = pts_scaled - centroid_scaled
    
    # Recalculate S for scaled points
    best_S_scaled = float('inf')
    best_phi_scaled = 0.0
    for i in range(1001):
        phi = i * (math.pi / 6.0) / 1000.0
        angles = phi + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = np.abs(centered_scaled @ normals.T)
        max_proj = np.max(projs)
        S = (max_proj + APOTHEM) / APOTHEM
        if S < best_S_scaled:
            best_S_scaled = S
            best_phi_scaled = phi

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), 0.0]) # Inner hexagons aligned (0 rotation relative to each other)
        
    center = [float(centroid_scaled[0]), float(centroid_scaled[1])]
    side = float(best_S_scaled)
    rotation = float(math.degrees(best_phi_scaled))
    
    return inner, center, side, rotation
