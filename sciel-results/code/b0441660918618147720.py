import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin for 120s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _check_overlap(poly1, poly2):
    """
    Check if two convex polygons overlap using Separating Axis Theorem (SAT).
    Returns True if they overlap, False otherwise.
    """
    # Get all edges from both polygons
    edges = []
    for i in range(len(poly1)):
        p1 = poly1[i]
        p2 = poly1[(i+1)%len(poly1)]
        edge = p2 - p1
        edges.append(edge)
    for i in range(len(poly2)):
        p1 = poly2[i]
        p2 = poly2[(i+1)%len(poly2)]
        edge = p2 - p1
        edges.append(edge)
        
    # For each edge, get the normal axis
    for edge in edges:
        # Normal to the edge
        axis = np.array([-edge[1], edge[0]])
        # Project both polygons onto this axis
        proj1 = np.dot(poly1, axis)
        proj2 = np.dot(poly2, axis)
        
        # Check for separation
        max1 = np.max(proj1)
        min1 = np.min(proj1)
        max2 = np.max(proj2)
        min2 = np.min(proj2)
        
        if max1 < min2 or max2 < min1:
            return False # Separating axis found, no overlap
            
    return True # No separating axis, they overlap

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices for all hexagons
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Transform to outer hexagon coordinate system
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Support function for regular hexagon centered at origin
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _smooth_penalty(params, beta=50.0):
    """
    Compute a smooth penalty for overlaps and outer boundary.
    params: [x0, y0, a0, ..., x10, y10, a10]
    """
    pos = params[0:2*N].reshape(N, 2)
    ang = params[2*N:3*N]
    
    # 1. Outer Boundary Penalty
    # We want to minimize the outer side length.
    # Instead of hard max, we use a log-sum-exp style approximation or just the max itself 
    # if we are using a gradient method that can handle kinks (L-BFGS-B might struggle with kinks, 
    # but the outer boundary function is actually piecewise linear, so it's convex).
    # However, the "min outer side" is the max of support functions.
    # Let's compute the actual side length and use it as part of the objective.
    side = _min_outer_side(pos, ang, (0.0, 0.0), 0.0)
    
    # 2. Overlap Penalty
    # We use a smooth approximation of the overlap constraint.
    # For two hexagons, if the distance between centers is less than 2*sqrt(3) (touching distance for parallel hexagons),
    # they might overlap. 
    # A robust smooth penalty is based on the minimum distance between polygons.
    # However, computing exact min distance is expensive.
    # Alternative: Use a pairwise potential that grows rapidly if they get too close.
    # Since we start from a valid lattice, we can use a simple repulsive force model.
    # If dist < d_min, penalty = (d_min - dist)^2 * K
    
    penalty = 0.0
    # Optimal distance for non-overlap of unit hexagons in a lattice is 2*sqrt(3) ~ 3.464.
    # But rotated hexagons can be closer.
    # Let's use a conservative threshold for the "soft" constraint.
    # Actually, for L-BFGS, we need differentiability.
    # Let's use the SAT check only for validation, and for optimization,
    # we rely on the fact that we start from a valid lattice and move slowly.
    # To make it robust, we add a penalty if any vertex of hex i is inside hex j.
    # This is still non-smooth.
    
    # Better approach for this specific strategy:
    # Since we start on a lattice, the hexagons are disjoint.
    # We optimize the positions to minimize the outer side.
    # We add a small repulsive term to prevent crossing during optimization.
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Threshold for "too close" (approximate min distance for any rotation is > 1.732, 
            # but let's use 2.5 as a safe buffer for the soft constraint)
            threshold = 2.5 
            if dist < threshold:
                # Smooth quadratic penalty
                penalty += (threshold - dist) ** 2 * 10.0
                
    return side + penalty

def _initial_lattice_configuration():
    """
    Generate a valid initial configuration using a hexagonal lattice.
    This guarantees no overlaps and a known starting score.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # Hexagonal lattice vectors
    # v1 = (2*sqrt(3), 0)
    # v2 = (sqrt(3), 3)
    # v3 = (-sqrt(3), 3)
    
    # We want to pack 11 hexagons.
    # Center at (0,0)
    pos[0] = [0, 0]
    
    # Ring 1: 6 hexagons
    # In a lattice, neighbors are at distance 2*sqrt(3)
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # Ring 2: 4 hexagons to make 11 total.
    # We can place them in the "holes" or further out.
    # The standard "flower" has 7. To get 11, we add 4 more.
    # Let's place them at the next shell positions that keep the shape compact.
    # Positions for ring 2 in a hex lattice:
    # (4*sqrt3, 0), (2*sqrt3, 3), (-2*sqrt3, 3), (-4*sqrt3, 0), (-2*sqrt3, -3), (2*sqrt3, -3)
    # That's 6 more, total 13.
    # We only need 4 more. Let's pick 4 symmetrically to maintain balance.
    # Pick angles: 0, 90, 180, 270? No, lattice is 60 deg symmetry.
    # Pick 4 from the 6 available ring 2 spots.
    # Let's use indices 0, 2, 3, 5 of the ring 2 set (symmetric).
    ring2_angles = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    # Actually, let's just place them at distance 4*sqrt3 at angles 0, pi/2, pi, 3pi/2?
    # No, they must be on the lattice to be disjoint from ring 1.
    # Ring 1 is at dist 2*sqrt3.
    # Ring 2 valid spots are at dist 4*sqrt3 (along axes) or sqrt((2sqrt3)^2 + 3^2) = sqrt(12+9)=sqrt(21) ~ 4.58?
    # Lattice points:
    # (0,0)
    # (2s, 0), (s, 3), (-s, 3), (-2s, 0), (-s, -3), (s, -3)  where s=sqrt(3)
    # Next layer:
    # (4s, 0), (3s, 3), (2s, 6), (-s, 6), (-2s, 3), (-3s, 3) ... wait.
    # Let's just use the 6 spots at distance 4*sqrt(3) along the 6 main axes?
    # No, (4s, 0) is a lattice point. (2s, 3) is NOT a lattice point. (s,3) is.
    # Lattice points (x, y) where x = s*(i+j), y = 3/2*(i-j) * 2? 
    # Standard hex grid coords:
    # x = s * (q + r/2)
    # y = 3/2 * r * s? No.
    # Let's stick to the explicit coordinates generated by vectors.
    # v1 = (2s, 0), v2 = (s, 3).
    # Points: n1*v1 + n2*v2.
    # (0,0)
    # (2s, 0), (s, 3), (-s, 3), (-2s, 0), (-s, -3), (s, -3)
    # (4s, 0), (3s, 3), (2s, 6), (-s, 6), (-2s, 3), (-3s, 3) ... 
    # Wait, (2s, 6) = 1*v1 + 2*v2 = (2s, 0) + (2s, 6) = (4s, 6)? No.
    # v2 = (s, 3). 2*v2 = (2s, 6). v1 + 2*v2 = (2s, 0) + (2s, 6) = (4s, 6).
    # Let's just pick 4 points from the second shell that are far enough.
    # The 6 points in the second shell (distance from origin > 2s) that are lattice points:
    # (4s, 0), (3s, 3), (2s, 6), (-s, 6), (-2s, 3), (-3s, 3) -- these are not all same distance.
    # Distances:
    # (4s, 0): 4s
    # (3s, 3): sqrt(27+9) = sqrt(36) = 6
    # (2s, 6): sqrt(12+36) = sqrt(48)
    # It's easier to just place them in the "gaps" of the first ring if they fit.
    # But for N=11, the optimal is likely the 7 in the center + 4 on the outside.
    # Let's place the 4 extra hexagons at the vertices of a larger hexagon, slightly rotated or just at the lattice points.
    # To ensure validity, let's place them at (4s, 0), (-4s, 0), (0, 4s*sqrt(3)/2 * 2)? No.
    
    # Let's use a simpler valid config:
    # 1 center
    # 6 around it
    # 4 more: place them at (4*sqrt(3), 0), (2*sqrt(3), 6), (-2*sqrt(3), 6), (-4*sqrt(3), 0)?
    # Let's check (2*sqrt(3), 6).
    # Is it a lattice point?
    # x = 2s, y = 6.
    # y = 3 * r => r = 2.
    # x = s * (q + r/2) => 2s = s * (q + 1) => q = 1.
    # So (1, 2) in axial coords. Yes, it is a lattice point.
    # Similarly, (-2s, 6) is q=-3, r=2?
    # x = -2s = s(q + 1) => q = -3. r=2.
    # (-4s, 0) is q=-4, r=0.
    # (4s, 0) is q=4, r=0.
    
    # So we add:
    pos[7] = [4*sqrt3, 0]
    pos[8] = [2*sqrt3, 6]
    pos[9] = [-2*sqrt3, 6]
    pos[10] = [-4*sqrt3, 0]
    
    return pos, ang

def construct():
    global START_TIME, DEADLINE
    
    # Initialize with the standard lattice pattern
    pos0, ang0 = _initial_lattice_configuration()
    params0 = np.concatenate([pos0.ravel(), ang0])
    
    # Initial objective
    best_params = params0.copy()
    best_obj = _smooth_penalty(best_params)
    
    # Multi-start optimization using L-BFGS-B
    # L-BFGS-B is good for large scale problems with bounds.
    # We don't have explicit bounds, but we can set loose bounds to prevent explosion.
    
    starts = [params0.copy()]
    
    # Add some random perturbations to escape local minima
    np.random.seed(42)
    for _ in range(20):
        p = params0.copy()
        # Perturb positions slightly
        p[0:2*N] += np.random.normal(0, 0.5, (2*N,))
        # Perturb angles slightly
        p[2*N:3*N] += np.random.normal(0, 5.0, (N,))
        starts.append(p)
    
    # Optimization loop
    for start in starts:
        if time.time() > DEADLINE:
            break
        
        try:
            # Use L-BFGS-B
            # maxiter is per function call.
            res = minimize(_smooth_penalty, start, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > DEADLINE:
            break
    
    # Extract best solution
    pos_best = best_params[0:2*N].reshape(N, 2)
    ang_best = best_params[2*N:3*N]
    
    # Verify validity (no overlap)
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            dx = pos_best[i,0] - pos_best[j,0]
            dy = pos_best[i,1] - pos_best[j,1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 10.0: # 3.16^2, safe upper bound for overlap check
                v1 = _hex_vertices(pos_best[i,0], pos_best[i,1], 1.0, ang_best[i])
                v2 = _hex_vertices(pos_best[j,0], pos_best[j,1], 1.0, ang_best[j])
                if _check_overlap(v1, v2):
                    valid = False
                    break
        if not valid:
            break
    
    if not valid:
        # Fallback to initial configuration
        pos_best = pos0
        ang_best = ang0
    
    # Calculate the actual side length without penalty
    side_best = _min_outer_side(pos_best, ang_best, (0.0, 0.0), 0.0)
    
    inner_list = [[float(pos_best[i,0]), float(pos_best[i,1]), float(ang_best[i])] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(side_best), 0.0

if __name__ == "__main__":
    construct()
