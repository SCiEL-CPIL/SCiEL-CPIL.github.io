"""Packing 11 unit hexagons in minimum enclosing hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def hexagon_support(px, py, angle_deg):
    """Compute max |p·d_k| over 6 hexagon normals for flat-top hexagon rotated by angle_deg."""
    phi = np.radians(angle_deg)
    max_val = 0.0
    for k in range(6):
        d = np.array([np.cos(k * np.pi / 3 + phi), np.sin(k * np.pi / 3 + phi)])
        val = abs(px * d[0] + py * d[1])
        if val > max_val:
            max_val = val
    return max_val

def enclosing_hexagon_side(positions, center, angle_deg):
    """Compute the side length of the minimum enclosing hexagon."""
    # Translate positions relative to center
    rel = np.array(positions) - np.array(center)
    max_val = 0.0
    phi = np.radians(angle_deg)
    for k in range(6):
        d = np.array([np.cos(k * np.pi / 3 + phi), np.sin(k * np.pi / 3 + phi)])
        vals = np.abs(rel[:, 0] * d[0] + rel[:, 1] * d[1])
        mv = np.max(vals)
        if mv > max_val:
            max_val = mv
    return max_val / (np.sqrt(3) / 2)

def min_pairwise_distance(positions):
    """Compute minimum pairwise distance between hexagon centers."""
    n = len(positions)
    min_d = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_d:
                min_d = d
    return min_d

def objective(params):
    """Objective function: minimize enclosing hexagon side + penalty for overlaps."""
    # params: 22 values (11 x 2D positions)
    positions = params.reshape(N, 2)
    
    # Compute enclosing hexagon side
    side = enclosing_hexagon_side(positions, [0.0, 0.0], 0.0)
    
    # Penalty for overlaps (distance < 2)
    min_d = min_pairwise_distance(positions)
    if min_d < 2.0:
        penalty = 100.0 * (2.0 - min_d) ** 2
    else:
        penalty = 0.0
    
    return side + penalty

def run_optimization(x0, method='L-BFGS-B', maxiter=500):
    """Run a single optimization."""
    result = minimize(objective, x0, method=method, maxiter=maxiter, 
                      options={'ftol': 1e-8, 'maxls': 50})
    return result

def construct():
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try multiple initial configurations
    np.random.seed(42)
    
    initial_configs = []
    
    # Config 1: Hexagonal lattice arrangement
    # Place in a hexagonal pattern
    for r in range(4):
        for c in range(4):
            x = c * 2.1 + (r % 2) * 1.05
            y = r * 1.8
            initial_configs.append([x, y])
    # Take first 11
    initial_configs = initial_configs[:11]
    x0_1 = np.array(initial_configs).flatten()
    
    # Config 2: Random
    x0_2 = np.random.randn(N, 2) * 5.0
    x0_2 = x0_2.flatten()
    
    # Config 3: Circular arrangement
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0_3 = np.column_stack([3.0 * np.cos(angles), 3.0 * np.sin(angles)]).flatten()
    
    # Config 4: Grid
    x0_4 = []
    for i in range(11):
        x0_4.append([float(i % 4) * 2.1, float(i // 4) * 1.8])
    x0_4 = np.array(x0_4).flatten()
    
    # Config 5: Dense cluster
    x0_5 = np.random.randn(N, 2) * 2.0
    x0_5 = x0_5.flatten()
    
    all_configs = [x0_1, x0_2, x0_3, x0_4, x0_5]
    
    for x0 in all_configs:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Try L-BFGS-B
        try:
            result = run_optimization(x0, method='L-BFGS-B', maxiter=300)
            if result.success or result.fun < best_side:
                positions = result.x.reshape(N, 2)
                min_d = min_pairwise_distance(positions)
                if min_d >= 2.0 - 1e-6:
                    side = enclosing_hexagon_side(positions, [0.0, 0.0], 0.0)
                    if side < best_side:
                        best_side = side
                        best_positions = positions.copy()
        except Exception:
            pass
        
        if time.time() - START_TIME > TIME_LIMIT:
            break
    
    # If we found a valid solution, use it
    if best_positions is not None:
        # Compute optimal center (centroid)
        center = [float(np.mean(best_positions[:, 0])), float(np.mean(best_positions[:, 1]))]
        # Recompute side with optimal center
        # Actually, for minimum enclosing hexagon, the center should be chosen optimally
        # Let's try a few center positions
        best_local_side = best_side
        best_local_center = center
        
        # Try center at centroid
        for cx in [center[0] - 0.5, center[0], center[0] + 0.5]:
            for cy in [center[1] - 0.5, center[1], center[1] + 0.5]:
                s = enclosing_hexagon_side(best_positions, [cx, cy], 0.0)
                if s < best_local_side:
                    best_local_side = s
                    best_local_center = [cx, cy]
        
        # Convert to inner format
        inner = []
        for pos in best_positions:
            inner.append([float(pos[0] - best_local_center[0]), float(pos[1] - best_local_center[1]), 0.0])
        
        return inner, (best_local_center[0], best_local_center[1]), float(best_local_side), 0.0
    else:
        # Fallback: return the seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
