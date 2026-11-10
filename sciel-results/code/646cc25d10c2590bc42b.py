# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _all_inner_vertices(positions):
    """Return (72, 2) array of all vertices of 12 unit hexagons at given positions (angle=0)."""
    n = len(positions)
    all_verts = np.empty((n * 6, 2))
    for i in range(n):
        cx, cy = positions[i]
        all_verts[i*6:(i+1)*6] = _hex_vertices(cx, cy, 1.0, 0)
    return all_verts

def _separating_axis_check(positions, tol=1e-6):
    """Check that no two inner hexagons overlap using separating axis test."""
    n = len(positions)
    # For unit hexagons at angle 0, the 6 edge normals are at angles 0, 60, 120, 180, 240, 300
    # Edge normals (perpendicular to edges): for a flat-top hex, edges have normals at 30, 90, 150, 210, 270, 330
    # Actually for pointy-top (angle 0 means vertex at 0 deg), let's just use the 6 directions
    # The separating axes for two unit hexagons are the 6 edge normals of each
    axes = []
    for i in range(6):
        ang = np.radians(60 * i)
        axes.append(np.array([np.cos(ang), np.sin(ang)]))
    # Also the perpendiculars
    for i in range(6):
        ang = np.radians(60 * i + 30)
        axes.append(np.array([np.cos(ang), np.sin(ang)]))
    
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0))
    
    for i in range(n):
        for j in range(i+1, n):
            vi = verts_list[i]
            vj = verts_list[j]
            for axis in axes:
                proj_i = np.dot(vi, axis)
                proj_j = np.dot(vj, axis)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                # Check if intervals are separated
                if max_i + tol < min_j or max_j + tol < min_i:
                    break  # This axis separates them, no overlap on this axis
            else:
                # No axis separates them -> overlap
                return False
    return True

def _min_containing_hex_side(positions, ox, oy, angle_deg):
    """
    Compute minimum side s of outer hexagon centered at (ox,oy) with given angle
    that contains all inner hexagons.
    """
    all_verts = _all_inner_vertices(positions)
    
    def _is_inside(s):
        outer_verts = _hex_vertices(ox, oy, s, angle_deg)
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1) % 6]
            edge = p2 - p1
            diffs = all_verts - p1
            crosses = edge[0] * diffs[:, 1] - edge[1] * diffs[:, 0]
            if np.any(crosses < -1e-6):
                return False
        return True
    
    lo = np.sqrt(12)
    hi = 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _is_inside(mid):
            hi = mid
        else:
            lo = mid
    return hi

def _objective(params, positions):
    """
    params: [ox, oy, angle_deg]
    Returns the minimal side length s.
    """
    ox, oy, angle_deg = params
    return _min_containing_hex_side(positions, ox, oy, angle_deg)

def _generate_cluster_configurations():
    """
    Generate multiple 12-hex cluster configurations on hexagonal lattice.
    Hex lattice: centers at (2.5*i + 1.25*j, sqrt(3)*1.25*j)
    """
    sqrt3 = np.sqrt(3)
    configs = []
    
    # Configuration 1: Center + Ring 1 (6) + 5 from Ring 2
    # Ring 2 has 12 positions; pick 5 that give good symmetry
    center = np.array([0.0, 0.0])
    ring1 = []
    for i in range(6):
        ang = np.radians(60 * i)
        ring1.append(np.array([2.5 * np.cos(ang), 2.5 * np.sin(ang)]))
    
    ring2 = []
    for i in range(6):
        ang = np.radians(60 * i)
        ring2.append(np.array([5.0 * np.cos(ang), 5.0 * np.sin(ang)]))
    for i in range(6):
        ang = np.radians(60 * i + 30)
        ring2.append(np.array([5.0 * np.cos(ang), 5.0 * np.sin(ang)]))
    
    # Pick 5 from ring2: try different subsets
    # Option A: 5 consecutive from ring2
    for start in range(6):
        subset = []
        for k in range(5):
            idx = (start + k) % 12
            subset.append(ring2[idx])
        positions = [center] + ring1 + subset
        configs.append(np.array(positions))
    
    # Configuration 2: Use a different lattice arrangement
    # Hex numbers: 1, 7, 19, 37... 
    # 12 = 7 + 5, so center+ring1+5 from ring2 is the natural decomposition
    
    # Configuration 3: Offset the center
    for dx in [-0.5, 0.0, 0.5]:
        for dy in [-0.5, 0.0, 0.5]:
            center_off = np.array([dx, dy])
            positions = [center_off + p for p in ([center] + ring1 + ring2[:5])]
            configs.append(np.array(positions))
    
    return configs

def hexagon_packing_12():
    """
    Pack 12 unit hexagons inside a minimal regular hexagon.
    Uses hexagonal lattice configurations and optimizes the containing hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    configs = _generate_cluster_configurations()
    
    best_s = np.inf
    best_positions = None
    best_ox, best_oy, best_angle = 0.0, 0.0, 0.0
    
    for cfg_idx, positions in enumerate(configs):
        if time.time() > deadline - 60:
            break
        
        # Center the configuration
        centroid = positions.mean(axis=0)
        pos_centered = positions - centroid
        
        # Verify separation
        if not _separating_axis_check(pos_centered):
            continue
        
        # Optimize outer hexagon parameters: (ox, oy, angle)
        # Start from several initial points
        best_local_s = np.inf
        best_local_params = None
        
        init_points = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 30.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],
            [0.3, 0.3, 15.0],
            [-0.3, 0.3, 45.0],
        ]
        
        for init in init_points:
            if time.time() > deadline - 30:
                break
            
            # Use Nelder-Mead
            try:
                result = minimize(
                    _objective,
                    x0=init,
                    args=(pos_centered,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                s_val = result.fun
                if s_val < best_local_s:
                    best_local_s = s_val
                    best_local_params = result.x
            except Exception:
                pass
        
        # Also try grid search refinement
        if best_local_params is not None:
            ox, oy, angle = best_local_params
            step = 0.1
            for _ in range(50):
                if time.time() > deadline - 30:
                    break
                improved = False
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        for da in [-step, 0, step]:
                            if dx == 0 and dy == 0 and da == 0:
                                continue
                            new_s = _min_containing_hex_side(pos_centered, ox + dx, oy + dy, angle + da)
                            if new_s < best_local_s - 1e-10:
                                best_local_s = new_s
                                ox, oy, angle = ox + dx, oy + dy, angle + da
                                improved = True
                if not improved:
                    step *= 0.5
                    if step < 0.001:
                        break
        
        if best_local_s < best_s:
            best_s = best_local_s
            best_positions = pos_centered
            if best_local_params is not None:
                best_ox, best_oy, best_angle = best_local_params
    
    # Final high-precision polish
    if best_positions is not None and time.time() < deadline - 10:
        ox, oy, angle = best_ox, best_oy, best_angle
        step = 0.01
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    for da in [-step, 0, step]:
                        if dx == 0 and dy == 0 and da == 0:
                            continue
                        new_s = _min_containing_hex_side(best_positions, ox + dx, oy + dy, angle + da)
                        if new_s < best_s - 1e-12:
                            best_s = new_s
                            ox, oy, angle = ox + dx, oy + dy, angle + da
                            improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
        best_ox, best_oy, best_angle = ox, oy, angle
    
    # Build output: shift inner positions by (-best_ox, -best_oy)
    inner_positions = best_positions - np.array([best_ox, best_oy])
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = inner_positions[:, 0]
    inner_hex_data[:, 1] = inner_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
