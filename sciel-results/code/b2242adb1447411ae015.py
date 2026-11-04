"""
Strategy: Contact-Graph Topology-Driven Nelder-Mead Optimization with SAT Non-Overlap.

This approach treats the packing as a topological constraint satisfaction problem:
1. Define a contact graph where hexagons are connected if they are expected to touch.
2. Parameterize the configuration by the positions of hexagons relative to their neighbors in the contact graph, allowing non-adjacent hexagons to slide freely.
3. Use Nelder-Mead simplex optimization (derivative-free, robust to non-smoothness) on the raw coordinates of all hexagons.
4. The objective is the outer hexagon side length (max radial projection), with a heavy penalty for any overlap detected via Separating Axis Theorem (SAT).
5. This decouples the rigid lattice assumption from the optimization, allowing the optimizer to find non-lattice packings.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin for 150s limit

def _hex_vertices(center, angle):
    """Compute vertices of a unit regular hexagon."""
    angles = np.arange(6) * np.pi / 3 + angle
    return center + np.column_stack([np.cos(angles), np.sin(angles)])

def _get_all_vertices(pos, ang):
    """
    Compute vertices for all N hexagons.
    pos: (N, 2) array of centers
    ang: (N,) array of angles in radians
    Returns: (N, 6, 2) array of vertices
    """
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i] = _hex_vertices(pos[i], ang[i])
    return verts

def _sat_overlap_depth(v1, v2):
    """
    Compute the minimum separation distance between two convex polygons using SAT.
    Returns the minimum penetration depth (positive if overlapping, 0 if separated).
    For two convex polygons, if they overlap, the minimum overlap on any separating axis
    gives the penetration depth. If they are separated, the minimum gap on any axis
    gives the separation distance.
    
    We return: max(0, min_overlap_on_any_axis) where min_overlap is the overlap amount.
    Actually, for SAT: if polygons overlap, there is NO separating axis.
    The penetration depth is the minimum over all axes of the overlap amount on that axis.
    If polygons are separated, the maximum over all axes of the gap gives the separation.
    
    For our penalty, we want: if overlapping, return the minimum overlap depth.
    If separated, return 0.
    """
    min_overlap = np.inf
    for poly in [v1, v2]:
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal /= norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            m1_min, m1_max = proj1.min(), proj1.max()
            m2_min, m2_max = proj2.min(), proj2.max()
            
            # Overlap on this axis
            overlap = min(m1_max, m2_max) - max(m1_min, m2_min)
            if overlap < min_overlap:
                min_overlap = overlap
    
    # If min_overlap < 0, they are separated on at least one axis -> no overlap
    # If min_overlap >= 0, they overlap on all axes -> overlapping with depth min_overlap
    if min_overlap < 0:
        return 0.0
    return min_overlap

def _check_overlap(pos, ang):
    """
    Check for any overlap between hexagons.
    Returns the maximum overlap depth (0 if no overlap).
    """
    verts = _get_all_vertices(pos, ang)
    max_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts[i], verts[j])
            if depth > max_overlap:
                max_overlap = depth
    return max_overlap

def _min_outer_side(pos, ang, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(pos, dtype=np.float64)
    ang = np.radians(np.asarray(ang, dtype=np.float64))
    
    verts = _get_all_vertices(pos, ang)
    
    all_verts_x = verts[:, :, 0].ravel()
    all_verts_y = verts[:, :, 1].ravel()
    
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _objective(params):
    """
    Objective function for Nelder-Mead.
    params: [x0, y0, ang0, x1, y1, ang1, ..., x10, y10, ang10]
    Total 3*N = 33 parameters.
    """
    if time.time() > DEADLINE:
        return 1e6
    
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(N):
        pos[i, 0] = params[3*i]
        pos[i, 1] = params[3*i + 1]
        ang[i] = params[3*i + 2]
    
    # Check for overlap
    overlap = _check_overlap(pos, ang)
    if overlap > 0:
        # Heavy penalty for overlap
        return 1000.0 + 100.0 * overlap
    
    # Compute outer side length
    side = _min_outer_side(pos, ang)
    
    return side

def _initial_configuration():
    """
    Generate an initial configuration using a hexagonal lattice.
    Center hexagon at origin, 6 around it, 4 in second ring.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    pos[0] = [0, 0]
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
    
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
    
    return pos, ang

def construct():
    global START_TIME, DEADLINE
    
    # Get initial configuration
    pos0, ang0 = _initial_configuration()
    
    # Flatten to parameter vector
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = pos0[i, 0]
        x0[3*i + 1] = pos0[i, 1]
        x0[3*i + 2] = ang0[i]
    
    # Run Nelder-Mead optimization
    best_x = x0.copy()
    best_f = _objective(x0)
    
    # Multiple restarts with different initial perturbations
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 30.0:
            break
        
        # Perturb initial configuration
        if restart == 0:
            x_current = x0.copy()
        else:
            x_current = x0 + rng.normal(0, 0.5, size=3*N)
        
        # Run Nelder-Mead
        try:
            result = minimize(
                _objective,
                x_current,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
                
        except Exception as e:
            pass
    
    # Extract best solution
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(N):
        pos[i, 0] = best_x[3*i]
        pos[i, 1] = best_x[3*i + 1]
        ang[i] = best_x[3*i + 2]
    
    # Final verification
    overlap = _check_overlap(pos, ang)
    if overlap > 1e-6:
        # Invalid, fallback to initial configuration
        pos, ang = _initial_configuration()
    
    best_side = _min_outer_side(pos, ang)
    
    inner_list = [[float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()
