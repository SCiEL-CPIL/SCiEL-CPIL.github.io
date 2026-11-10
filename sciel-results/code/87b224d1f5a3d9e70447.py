# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack((xs, ys))

def _min_outer_side(vertices, outer_angle_deg):
    """
    Calculate the minimum side length of a regular hexagon with the given angle
    that contains all given vertices.
    vertices: (N, 2) array of points.
    outer_angle_deg: angle of the outer hexagon.
    Returns: side length.
    """
    a = np.deg2rad(outer_angle_deg)
    # Normals to the edges of the outer hexagon
    # Edge normals are at a + 30 + 60*i
    normal_angles = a + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Project all vertices onto each normal
    # projections: (6, N)
    projections = vertices @ normals.T
    
    # Max projection for each normal
    max_projections = np.max(projections, axis=1)
    
    # The apothem (distance to side) must be >= max projection
    # Apothem = side * cos(30) = side * sqrt(3)/2
    # So side >= max_proj / cos(30)
    min_apothem = np.max(max_projections)
    side = min_apothem / np.cos(np.deg2rad(30))
    return side

def _check_overlap_penalty(centers, angles_deg, s=1.0, eps=1e-6):
    """
    Calculate penalty for overlapping inner hexagons.
    centers: (11, 2)
    angles_deg: (11,)
    Returns: sum of squared overlaps.
    """
    n = len(centers)
    penalty = 0.0
    vertices_list = []
    for i in range(n):
        v = _hex_vertices(centers[i][0], centers[i][1], s, angles_deg[i])
        vertices_list.append(v)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            
            # Get edge normals for hex 1 and hex 2
            a1 = np.deg2rad(angles_deg[i])
            a2 = np.deg2rad(angles_deg[j])
            
            n1_angles = a1 + np.deg2rad(60 * np.arange(6)) + np.deg2rad(30)
            n2_angles = a2 + np.deg2rad(60 * np.arange(6)) + np.deg2rad(30)
            
            normals1 = np.column_stack((np.cos(n1_angles), np.sin(n1_angles)))
            normals2 = np.column_stack((np.cos(n2_angles), np.sin(n2_angles)))
            
            all_normals = np.vstack([normals1, normals2])
            
            min_gap = np.inf
            for n_vec in all_normals:
                proj1 = v1 @ n_vec
                proj2 = v2 @ n_vec
                
                max1 = np.max(proj1)
                min1 = np.min(proj1)
                max2 = np.max(proj2)
                min2 = np.min(proj2)
                
                # Gap is max(min1 - max2, min2 - max1)
                gap = max(min1 - max2, min2 - max1)
                if gap < min_gap:
                    min_gap = gap
            
            if min_gap < eps:
                # Overlapping or touching
                # Penalty proportional to overlap amount
                overlap = max(0, eps - min_gap)
                penalty += overlap ** 2
    
    return penalty

def _objective(params):
    """
    params: [x0..x10, y0..y10, cluster_angle, outer_angle]
    """
    centers = np.array(params[:22]).reshape(11, 2)
    cluster_angle = params[22]
    outer_angle = params[23]
    
    # All inner hexagons have the same angle
    angles = np.full(11, cluster_angle)
    
    # Check overlap
    penalty = _check_overlap_penalty(centers, angles)
    
    if penalty > 1e-4:
        return 1e6 + penalty
    
    # Calculate min outer side
    # Get all vertices
    all_vertices = []
    for i in range(11):
        v = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        all_vertices.append(v)
    
    all_vertices = np.vstack(all_vertices)
    
    # Find best outer angle? 
    # The objective function receives outer_angle as a variable.
    # But we can also optimize it locally to be faster/more robust.
    # Let's just use the provided outer_angle for now, as DE will optimize it.
    side = _min_outer_side(all_vertices, outer_angle)
    
    return side

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # 580s limit
    
    # Initial guess: Grid
    # Center at 0,0
    # Hex side 1. Distance between centers for touching hexes is 2*apothem = sqrt(3) ~ 1.732?
    # No, distance between centers of touching hexes is 2 * s * sin(60) = s * sqrt(3).
    # Wait, hexagon geometry:
    # Vertices at distance s from center.
    # Distance between parallel sides is s * sqrt(3).
    # Distance between centers of two touching hexes (sharing a side) is s * sqrt(3).
    # So if s=1, distance is sqrt(3) ~ 1.732.
    
    # The seed used 2.5 and 1.25.
    # 2.5 is > 1.732, so they don't touch.
    # Let's use a tighter grid.
    # Row 0: 3 hexes?
    # Let's just use the seed positions but scaled down?
    # Seed:
    # [0, 0]
    # [-2.5, 0], [2.5, 0]
    # [-1.25, 2.17], [1.25, 2.17]
    # [-1.25, -2.17], [1.25, -2.17]
    # [-3.75, 2.17], [3.75, 2.17]
    # [-3.75, -2.17], [3.75, -2.17]
    
    # This is a 1-2-2-2-1 pattern? No.
    # Let's construct a valid tight starting point.
    # A "flower" of 6 around 1 center is 7 hexes.
    # Add 4 more.
    
    # Let's just use the seed positions as initial bounds center.
    init_centers = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17]
    ])
    
    # Flatten
    init_params = np.zeros(24)
    init_params[:22] = init_centers.flatten()
    init_params[22] = 0.0  # cluster angle
    init_params[23] = 0.0  # outer angle
    
    # Bounds
    # Centers can be roughly in [-5, 5]
    bounds = [(-5, 5)] * 22
    bounds.append((0, 180))  # cluster angle
    bounds.append((0, 60))    # outer angle (symmetry 60 deg)
    
    # Run DE
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            # x0=init_params # DE doesn't take x0 directly in this way, but we can set bounds tight or use init='differential'
        )
        
        if time.time() > deadline:
            # Fallback to seed
            return _get_seed_result()
            
        best_params = result.x
        best_side = result.fun
        
        # Verify validity
        centers = best_params[:22].reshape(11, 2)
        cluster_angle = best_params[22]
        outer_angle = best_params[23]
        angles = np.full(11, cluster_angle)
        
        penalty = _check_overlap_penalty(centers, angles)
        if penalty > 1e-6:
             # Try to repair or fallback
             return _get_seed_result()
             
        # Construct output
        inner_hex_data = np.zeros((11, 3))
        for i in range(11):
            inner_hex_data[i] = [centers[i][0], centers[i][1], cluster_angle]
            
        outer_hex_data = np.array([0.0, 0.0, outer_angle])
        
        # Recalculate side to be precise
        all_vertices = []
        for i in range(11):
            v = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        side = _min_outer_side(all_vertices, outer_angle)
        
        # Check side >= sqrt(11)
        if side < np.sqrt(11):
            side = np.sqrt(11)
            
        return inner_hex_data, outer_hex_data, side

    except Exception:
        return _get_seed_result()

def _get_seed_result():
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
