"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def get_all_vertices(params):
    """Extract vertices of all inner hexagons from parameters."""
    # params: [x0, y0, a0, ..., x11, y11, a11]
    all_verts = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        verts = hex_vertices(x, y, 1.0, a)
        all_verts.append(verts)
    return np.vstack(all_verts)

def objective(params):
    """
    Objective function to minimize.
    params: [x0, y0, a0, ..., x11, y11, a11, cx, cy, theta]
    Returns the minimal side length of the enclosing hexagon.
    """
    inner_params = params[:3*N]
    cx, cy, theta = params[3*N], params[3*N+1], params[3*N+2]
    
    # Get all vertices of inner hexagons
    verts = get_all_vertices(inner_params)
    
    # Compute the minimal enclosing regular hexagon side length
    # The enclosing hexagon has center (cx, cy), angle theta, side s.
    # Normals of the enclosing hexagon are at angles theta + k*pi/3
    # The distance from center to each face is s * sqrt(3)/2
    # Condition: | (p - c) dot n_k | <= s * sqrt(3)/2
    # So s >= 2/sqrt(3) * max | (p - c) dot n_k |
    
    theta_rad = np.radians(theta)
    normals = []
    for k in range(6):
        ang = theta_rad + k * np.pi / 3
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    # Relative positions
    rel = verts - np.array([cx, cy])
    
    # Projections
    projections = rel @ normals.T  # Shape (num_verts, 6)
    max_proj = np.max(np.abs(projections))
    
    s = (2.0 / np.sqrt(3.0)) * max_proj
    
    # Add a penalty if inner hexagons overlap? 
    # The problem states "disjoint". 
    # We should check for overlaps. 
    # A simple penalty: sum of overlaps. 
    # Or just rely on the optimizer to find a non-overlapping solution if initialized well.
    # Given the geometry, a hexagonal lattice initialization is non-overlapping.
    # We can add a small penalty for any pair of inner hexagons being too close.
    
    # Check overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
            xj, yj, aj = inner_params[3*j], inner_params[3*j+1], inner_params[3*j+2]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            # Minimum distance between centers of disjoint unit hexagons is sqrt(3)
            # If dist < sqrt(3) - epsilon, they overlap or are too close
            min_dist = np.sqrt(3.0)
            if dist < min_dist:
                # Quadratic penalty for violation
                penalty += (min_dist - dist)**2
    
    return s + 10.0 * penalty

def construct():
    """Construct 12 unit hexagons inside a minimal outer hexagon."""
    # Initialize with a hexagonal lattice pattern
    # Center at (0,0)
    # Ring 0: (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: 12 neighbors, take 5 of them to get 12 total
    
    sqrt3 = np.sqrt(3.0)
    
    # Axial to Cartesian: x = 1.5*q, y = sqrt(3)/2*q + sqrt(3)*r
    # Or simpler: 
    # q, r -> x = q * sqrt(3), y = r * sqrt(3) + q * sqrt(3)/2 ? 
    # Let's use standard hex grid coordinates.
    # Distance between centers is sqrt(3).
    
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    
    # Ring 1 (6 hexagons)
    for i in range(6):
        angle = i * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        positions.append((x, y, 0.0))
        
    # Ring 2 (12 hexagons, we need 5 more)
    # The ring 2 positions are at distance 2*sqrt(3) from center? No.
    # In a hex grid, the next ring is further out.
    # Let's just list 12 distinct positions in a compact cluster.
    # A 3x4 block? No, hex grid.
    # Let's use the first 12 points of a spiral or just pick 5 from ring 2.
    
    ring2_positions = []
    for i in range(12):
        # Approximate angles for ring 2
        # Actually, let's just generate a grid and pick the closest 12 to origin
        pass
        
    # Better initialization: 
    # Generate a grid of potential centers and pick the 12 closest to origin that are mutually non-overlapping.
    # Or just use the known "flower" pattern: 1 center, 6 around, 5 around those.
    
    # Let's refine the 5 positions for ring 2.
    # The 6 ring 1 positions are at angles 0, 60, ..., 300.
    # The ring 2 positions adjacent to ring 1 are at angles 30, 90, 150, ... and further out.
    # Let's just use a simple heuristic: 
    # Take the 6 ring 1 positions.
    # For each, add 1 neighbor in the "outward" direction.
    # This gives 6 more, total 13. We drop one.
    
    # Let's try a different simple pattern:
    # Two rows of 3, two rows of 3, etc?
    
    # Let's stick to the lattice:
    # q, r in hex coordinates.
    # x = 1.5 * q * side_hex_dist? 
    # Distance between centers d = sqrt(3).
    # x = d * (q + r/2) ? No.
    # x = d * q
    # y = d * (r + q/2) * sqrt(3)? 
    # Let's use:
    # x = q * sqrt(3)
    # y = r * 1.5 + q * 0.5 * sqrt(3)? 
    # Standard axial:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # Distance between (0,0) and (1,0) is sqrt(3). Correct.
    # Distance between (0,0) and (0,1) is 1.5? No.
    # Let's check: (0,0) -> (0,0). (0,1) -> (sqrt(3)/2, 1.5). Dist = sqrt(3/4 + 2.25) = sqrt(3). Correct.
    
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            candidates.append((x, y))
            
    # Sort by distance to origin
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Greedy selection: pick closest, if no overlap with already picked
    selected = []
    for x, y in candidates:
        # Check distance to all selected
        ok = True
        for sx, sy in selected:
            dist = np.sqrt((x-sx)**2 + (y-sy)**2)
            if dist < sqrt3 - 1e-6:
                ok = False
                break
        if ok:
            selected.append((x, y))
            if len(selected) == N:
                break
                
    # Initialize angles to 0
    init_params = []
    for x, y in selected:
        init_params.extend([x, y, 0.0])
        
    # Outer hexagon initial guess
    cx = np.mean([p[0] for p in selected])
    cy = np.mean([p[1] for p in selected])
    theta = 0.0
    
    init_params.extend([cx, cy, theta])
    
    # Optimize
    res = minimize(objective, init_params, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_params = res.x
    inner_params = best_params[:3*N]
    cx, cy, theta = best_params[3*N], best_params[3*N+1], best_params[3*N+2]
    
    # Construct inner list
    inner = []
    for i in range(N):
        x, y, a = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        inner.append([x, y, a])
        
    # Calculate final side length
    verts = get_all_vertices(inner_params)
    theta_rad = np.radians(theta)
    normals = []
    for k in range(6):
        ang = theta_rad + k * np.pi / 3
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(np.abs(projections))
    side = (2.0 / np.sqrt(3.0)) * max_proj
    
    return inner, (cx, cy), side, theta

# EVOLVE-BLOCK-END
