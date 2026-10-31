"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
TOL = 1e-9

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def hex_overlap_depth(c1, a1, c2, a2):
    """Compute overlap depth between two unit hexagons (positive if overlapping, 0 if touching/separated).
    Uses SAT: returns the maximum separation across all axes. If max separation < 0, they overlap.
    Returns -max_separation (so positive means overlap)."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    center_dist_sq = dx*dx + dy*dy
    
    # Maximum possible center distance for any overlap: 2 * circumradius = 2.0
    if center_dist_sq > 4.0 + 1e-12:
        return 0.0  # No overlap possible
    
    v1 = hexagon_vertices(c1[0], c1[1], 1.0, a1)
    v2 = hexagon_vertices(c2[0], c2[1], 1.0, a2)
    
    # Collect all edge normals from both hexagons
    axes = []
    for v in (v1, v2):
        for i in range(6):
            edge = v[(i+1) % 6] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    # Also add axis between centers
    if center_dist_sq > 1e-12:
        center_axis = np.array([dx, dy]) / np.sqrt(center_dist_sq)
        axes.append(center_axis)
    
    # For each axis, compute the separation
    max_sep = -float('inf')
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Separation: how much one needs to move to separate
        sep = min(min2 - max1, min1 - max2)
        if sep > max_sep:
            max_sep = sep
    
    # If max_sep >= 0, they are separated or touching (no overlap)
    # If max_sep < 0, they overlap by -max_sep
    if max_sep >= 0:
        return 0.0
    else:
        return -max_sep

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    # For a regular hexagon with side S, the distance from center to each edge is S*sqrt(3)/2
    # The outward normals from center to each edge midpoint
    normals = []
    for k in range(6):
        theta = oa + np.radians(60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for n in normals:
                d = np.dot(v - np.array([ox, oy]), n)
                if d > 0:
                    # For containment: d <= S*sqrt(3)/2 => S >= 2*d/sqrt(3)
                    S_needed = 2.0 * d / np.sqrt(3.0)
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center
    
    # Check no overlaps between inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            overlap = hex_overlap_depth((cx1, cy1), a1, (cx2, cy2), a2)
            if overlap > TOL:  # Overlap detected
                return False
    
    # Check all inner hexagons are inside outer hexagon
    oa_rad = np.radians(outer_angle)
    edge_dist = outer_side * np.sqrt(3.0) / 2.0
    
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for k in range(6):
                theta = oa_rad + np.radians(60 * k)
                n = np.array([np.cos(theta), np.sin(theta)])
                d = np.dot(v - np.array([ox, oy]), n)
                if d > edge_dist + TOL:
                    return False
    
    return True

def objective(params):
    """Objective function: outer side length + penalty for overlaps."""
    n = N
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = params[3*n+2]  # Keep in radians for consistency
    
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]  # Keep in radians
        inner.append((cx, cy, ang))
    
    S = compute_outer_side_length(inner, (ox, oy), np.degrees(outer_angle))
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            overlap = hex_overlap_depth((cx1, cy1), a1, (cx2, cy2), a2)
            if overlap > 0.0:
                # Strong quadratic penalty for overlaps
                penalty += overlap ** 2 * 1000 + overlap * 100
    
    return S + penalty

def generate_initial_config(scale=1.5):
    """Generate a valid initial configuration using a scaled hexagonal lattice."""
    # Hexagonal lattice positions for 11 hexagons
    # Center
    positions = [(0.0, 0.0)]
    
    # First ring (6 neighbors)
    # For pointy-top hexagons (standard grid):
    # Neighbors are at distance sqrt(3)
    # Angles: 0, 60, 120, 180, 240, 300
    for k in range(6):
        theta = np.radians(60 * k)
        positions.append((np.sqrt(3) * np.cos(theta), np.sqrt(3) * np.sin(theta)))
    
    # Second ring (need 4 more to reach 11)
    # Pick 4 symmetric positions from the second ring
    # Second ring distance is 2*sqrt(3)
    # Positions at angles 0, 30, 60... but let's just pick 4 distinct ones
    # e.g., 0, 90, 180, 270 degrees in the second ring?
    # Actually, let's just add 4 more from the second ring at 0, 60, 120, 180
    for k in [0, 1, 2, 3]:
        theta = np.radians(60 * k)
        positions.append((2*np.sqrt(3) * np.cos(theta), 2*np.sqrt(3) * np.sin(theta)))
    
    # Scale positions to ensure no overlap
    scaled_positions = [(x * scale, y * scale) for x, y in positions[:N]]
    
    # Create params array
    params = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(scaled_positions):
        params[3*i] = px
        params[3*i+1] = py
        params[3*i+2] = 0.0  # Angle 0
    
    # Outer container center and angle
    params[3*N] = 0.0
    params[3*N+1] = 0.0
    params[3*N+2] = 0.0
    
    return params

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple initial conditions with different scales and random perturbations
    np.random.seed(42)
    
    # Define a set of initial scales
    scales = [1.5, 1.2, 1.0, 0.9]
    
    for scale in scales:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
            
        # Generate base config
        base_params = generate_initial_config(scale)
        
        # Create a few perturbed versions
        for restart in range(3):
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            
            params = base_params.copy()
            if restart > 0:
                # Add small random noise
                noise = np.random.randn(3 * N + 3) * 0.1
                params += noise
            
            try:
                # Use Powell method which is derivative-free and robust
                res = minimize(objective, params, method='Powell', 
                              options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6})
                
                p = res.x
                inner = []
                for i in range(N):
                    cx = p[3*i]
                    cy = p[3*i+1]
                    ang = np.degrees(p[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                
                ox = p[3*N]
                oy = p[3*N+1]
                outer_angle = np.degrees(p[3*N+2]) % 360
                
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                
                # Check validity
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
                        
            except Exception as e:
                pass
    
    # Fallback: Use the largest scaled lattice that is valid
    if best_result is None:
        # Try scales from large to small until we find a valid one
        for scale in [2.0, 1.5, 1.2, 1.0]:
            params = generate_initial_config(scale)
            inner = []
            for i in range(N):
                cx = params[3*i]
                cy = params[3*i+1]
                ang = np.degrees(params[3*i+2]) % 360
                inner.append([cx, cy, ang])
            
            ox = params[3*N]
            oy = params[3*N+1]
            outer_angle = np.degrees(params[3*N+2]) % 360
            
            S = compute_outer_side_length(inner, (ox, oy), outer_angle)
            
            if check_validity(inner, (ox, oy), S, outer_angle):
                best_result = (inner, (ox, oy), S, outer_angle)
                best_score = S
                break
        
        if best_result is None:
            # Last resort: very large scale
            params = generate_initial_config(5.0)
            inner = []
            for i in range(N):
                cx = params[3*i]
                cy = params[3*i+1]
                ang = np.degrees(params[3*i+2]) % 360
                inner.append([cx, cy, ang])
            ox = params[3*N]
            oy = params[3*N+1]
            outer_angle = np.degrees(params[3*N+2]) % 360
            S = compute_outer_side_length(inner, (ox, oy), outer_angle)
            best_result = (inner, (ox, oy), S, outer_angle)
    
    return best_result

# EVOLVE-BLOCK-END
