import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hexagon_vertices(center, angle_deg, side=1.0):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    angle_deg: rotation in degrees
    side: side length
    """
    cx, cy = center
    rad = np.deg2rad(angle_deg)
    # Vertices of unit hexagon (flat top? pointy top?)
    # Let's use pointy top for consistency with standard math angles starting at 0.
    # Vertices at angles 0, 60, 120, 180, 240, 300 relative to center?
    # If angle=0, vertex at (1, 0).
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    # Wait, if I rotate the hex, the vertices rotate.
    # Standard hexagon vertices (side 1, center 0,0):
    # (1, 0), (0.5, sqrt(3)/2), ...
    # This is a "pointy-top" if we consider y-axis up? No, (1,0) is right.
    # Let's generate vertices relative to center.
    vx = side * np.cos(angles)
    vy = side * np.sin(angles)
    return np.column_stack((vx + cx, vy + cy))

def polygons_overlap(poly1, poly2):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    poly1, poly2: (N, 2) arrays of vertices.
    """
    # Get normals of edges for both polygons
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            # Normal is perpendicular to edge
            normal = np.array([-edge[1], edge[0]])
            normal /= np.linalg.norm(normal)
            axes.append(normal)
        return axes

    axes = get_axes(poly1) + get_axes(poly2)
    
    for axis in axes:
        # Project both polygons onto the axis
        proj1 = np.dot(poly1, axis)
        proj2 = np.dot(poly2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def calculate_side_length(positions, angles):
    """
    Calculate the minimum side length of the outer hexagon (fixed orientation 0)
    required to contain all inner hexagons.
    Outer hexagon is centered at (0,0).
    """
    # Outer hexagon normals (outward) for a flat-top hexagon?
    # Let's assume the outer hexagon is also "pointy-top" or "flat-top"?
    # The task says "regular hexagon". Orientation is a variable in the return?
    # "outer hexagon center (x, y), side length and rotation"
    # So we can optimize the outer rotation too?
    # The seed returns rotation 0.0.
    # If we fix outer rotation to 0, we lose generality.
    # However, due to symmetry, we can fix the outer rotation to 0 WLOG?
    # No, the inner packings might not align with the outer symmetry.
    # But we can just include outer rotation in the optimization variables?
    # Or simpler: fix outer rotation to 0. The inner hexagons can rotate freely.
    # Is there a benefit to rotating the outer hexagon?
    # If the inner packing has 6-fold symmetry, rotating outer hexagon by 30 deg might help?
    # Let's stick to outer rotation = 0 for simplicity, as inner rotations are free.
    # If we find a good packing, we can check if rotating outer hexagon helps.
    # Actually, the "side length" is the metric.
    # Let's assume outer hexagon is aligned with axes (flat top/bottom or pointy top/bottom).
    # Let's use the normals for a hexagon with flat top (horizontal top edge).
    # Vertices at 30, 90, 150, 210, 270, 330? No.
    # Flat top: Top edge is horizontal.
    # Vertices: (S/2, S*sqrt(3)/2), (S, 0)... no.
    # Let's use the projection method which is independent of vertex definition.
    # The 6 supporting lines of a regular hexagon (side S, center 0) have normals
    # at angles 0, 60, 120, 180, 240, 300?
    # If normals are at 0, 60..., the hexagon has vertices at 30, 90...
    # This is a "pointy-top" hexagon (vertex at 90).
    # Let's use normals at 30, 90, 150, 210, 270, 330.
    # This corresponds to a "flat-top" hexagon (edge perpendicular to 90 deg normal).
    
    normals = []
    for k in range(6):
        angle = np.deg2rad(30 + k * 60)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(N):
        cx, cy = positions[i]
        ang = angles[i]
        verts = hexagon_vertices((cx, cy), ang)
        # Project vertices onto normals
        projs = np.dot(verts, normals.T)
        # The max projection of any vertex on any normal
        # We need the max over all vertices and all normals
        # But the hexagon is convex, so max projection on a normal is at a vertex.
        # We need the max over all 6 normals.
        # For a given normal, the max projection is max(dot(verts, normal)).
        # We want the max over all normals.
        max_for_normal = np.max(projs, axis=0) # Shape (6,)
        max_proj = max(max_proj, np.max(max_for_normal))
        
    # Apothem A = max_proj
    # Side S = 2 * A / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3)

def objective(params):
    """
    Objective function for optimizer.
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    """
    positions = np.array(params[0::3])
    angles = np.array(params[1::3])
    
    # Calculate side length
    side = calculate_side_length(positions, angles)
    
    # Calculate overlap penalty
    overlap_pen = 0.0
    poly_cache = []
    for i in range(N):
        verts = hexagon_vertices((positions[i], positions[i+1]), angles[i]) # Wait, positions is flat
        # Correct indexing
        cx = positions[i*2]
        cy = positions[i*2+1]
        ang = angles[i]
        verts = hexagon_vertices((cx, cy), ang)
        poly_cache.append(verts)
        
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap(poly_cache[i], poly_cache[j]):
                # Estimate penetration depth or just count?
                # Counting is binary, bad for gradient.
                # Use distance between centers as proxy?
                # Better: Use the minimum separation distance from SAT?
                # For now, let's use a large penalty if overlap.
                # To make it smooth, we can approximate the overlap area or depth.
                # Simple: Penalty = 10.0
                overlap_pen += 10.0
                
    # Calculate containment penalty
    # Check if all vertices are inside the hexagon of side 'side'
    # If side is calculated from max projection, they are by definition inside
    # the hexagon of that side length.
    # So containment is automatically satisfied by the definition of 'side'.
    # We just minimize 'side'.
    
    return side + overlap_pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: Flower + 4
    # 1 center, 6 ring, 4 outer
    init_params = []
    
    # Center
    init_params.extend([0.0, 0.0, 0.0])
    
    # Ring 1
    for k in range(6):
        angle = k * 60.0
        rad = np.deg2rad(angle)
        x = np.sqrt(3) * np.cos(rad)
        y = np.sqrt(3) * np.sin(rad)
        # Rotate hex to align with radial direction?
        # If we rotate by angle, the "pointy" part points out.
        init_params.extend([x, y, angle])
        
    # Outer 4
    # Place them at distance 2*sqrt(3) approx, angles 0, 90, 180, 270?
    # Or 0, 60, 120, 180?
    # Let's try 0, 90, 180, 270 to spread them out.
    for k in range(4):
        angle = k * 90.0
        rad = np.deg2rad(angle)
        x = 3.5 * np.cos(rad)
        y = 3.5 * np.sin(rad)
        init_params.extend([x, y, angle])
        
    best_side = float('inf')
    best_params = None
    
    # Multiple restarts
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Perturb initial guess
        if restart == 0:
            params = np.array(init_params)
        else:
            params = np.array(init_params)
            # Add noise
            noise = np.random.normal(0, 0.5, size=len(params))
            params += noise
            
        # Optimize
        try:
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.success or res.fun < best_side:
                # Verify validity
                positions = res.x[0::3]
                angles = res.x[1::3]
                
                # Check overlaps
                valid = True
                poly_cache = []
                for i in range(N):
                    verts = hexagon_vertices((positions[i*2], positions[i*2+1]), angles[i])
                    poly_cache.append(verts)
                    
                for i in range(N):
                    for j in range(i+1, N):
                        if polygons_overlap(poly_cache[i], poly_cache[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    side = calculate_side_length(positions, angles)
                    if side < best_side:
                        best_side = side
                        best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    positions = best_params[0::3]
    angles = best_params[1::3]
    
    inner = []
    for i in range(N):
        inner.append([positions[i*2], positions[i*2+1], angles[i]])
        
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
