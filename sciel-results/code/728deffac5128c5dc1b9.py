import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _min_enclosing_hex_apothem(points, outer_angle_deg=0.0):
    """
    Find the minimum apothem of a regular hexagon (with given orientation)
    that contains all given points.
    """
    angle = np.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        a = angle + np.pi / 6 + k * np.pi / 3
        normals.append(np.array([np.cos(a), np.sin(a)]))
    
    max_apothem = 0.0
    for n in normals:
        projections = points @ n
        apothem_needed = (projections.max() - projections.min()) / 2.0
        max_apothem = max(max_apothem, apothem_needed)
    return max_apothem

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_score = float('inf')
    best_result = None
    
    # Define lattice topologies for 12 hexagons
    topologies = []
    
    # 4 rows x 3 cols
    for r in range(4):
        for c in range(3):
            topologies.append((c, r))
    
    # Also try 3 rows x 4 cols as a separate topology
    topologies_3x4 = []
    for r in range(3):
        for c in range(4):
            topologies_3x4.append((c, r))
    
    all_topologies = [topologies, topologies_3x4]
    
    for topology in all_topologies:
        if time.time() > deadline - 10:
            break
        
        def objective(params):
            if time.time() > deadline - 5:
                return 1e10
            scale, outer_rot, row_shift = params
            if scale < 0.85:
                return 1e10
            
            dx = np.sqrt(3) * scale
            dy = 1.5 * scale
            
            points = []
            for (c, r) in topology:
                x = c * dx + (r % 2) * dx / 2 + row_shift
                y = r * dy
                points.append([x, y])
            points = np.array(points)
            
            center = points.mean(axis=0)
            points_centered = points - center
            
            apothem = _min_enclosing_hex_apothem(points_centered, np.degrees(outer_rot))
            total_apothem = apothem + np.sqrt(3) / 2
            side_length = total_apothem * 2 / np.sqrt(3)
            
            return side_length
        
        x0 = np.array([1.0, 0.0, 0.0])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            
            if result.fun < best_score:
                best_score = result.fun
                scale, outer_rot, row_shift = result.x
                
                dx = np.sqrt(3) * scale
                dy = 1.5 * scale
                
                inner_points = []
                for (c, r) in topology:
                    x = c * dx + (r % 2) * dx / 2 + row_shift
                    y = r * dy
                    inner_points.append([x, y])
                inner_points = np.array(inner_points)
                
                center = inner_points.mean(axis=0)
                inner_points_centered = inner_points - center
                
                apothem = _min_enclosing_hex_apothem(inner_points_centered, np.degrees(outer_rot))
                total_apothem = apothem + np.sqrt(3) / 2
                side_length = total_apothem * 2 / np.sqrt(3)
                
                best_result = {
                    'inner': inner_points_centered.tolist(),
                    'center': [0.0, 0.0],
                    'side': side_length,
                    'angle': np.degrees(outer_rot)
                }
        except Exception:
            continue
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner_formatted = []
    for pt in best_result['inner']:
        inner_formatted.append([pt[0], pt[1], 0.0])
    
    return inner_formatted, (best_result['center'][0], best_result['center'][1]), best_result['side'], best_result['angle']

