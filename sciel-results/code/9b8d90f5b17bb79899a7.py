# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute the 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _get_all_vertices(positions, angles, side=1.0):
    """
    positions: (N,2), angles: (N,) in degrees. Returns (N*6, 2).
    """
    verts = []
    for i in range(len(positions)):
        verts.append(_hex_vertices(positions[i,0], positions[i,1], side, angles[i]))
    return np.vstack(verts)

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    For a regular hex with side s and angle 0, the 6 edge inward normals point at
    angles 30, 90, 150, 210, 270, 330 degrees. The distance from center to each edge
    is s * cos(30°) = s * sqrt(3)/2.
    So we need s such that for every vertex v, the projection onto each normal is
    <= s * sqrt(3)/2.
    Thus s = max over all vertices and all normals of (dot(v, normal) / (sqrt(3)/2)).
    """
    normals = np.array([
        [np.cos(np.radians(30)), np.sin(np.radians(30))],
        [np.cos(np.radians(90)), np.sin(np.radians(90))],
        [np.cos(np.radians(150)), np.sin(np.radians(150))],
        [np.cos(np.radians(210)), np.sin(np.radians(210))],
        [np.cos(np.radians(270)), np.sin(np.radians(270))],
        [np.cos(np.radians(330)), np.sin(np.radians(330))],
    ])
    # For each vertex, compute max projection onto each normal
    # We need the max over all vertices of the max projection onto any normal
    # Then s = that max / (sqrt(3)/2)
    projections = vertices @ normals.T  # (N*6, 6)
    max_proj = np.max(np.abs(projections), axis=1)  # max abs projection per vertex
    max_overall = np.max(max_proj)
    return max_overall / (np.sqrt(3) / 2)

def _check_separation(positions, angles, side=1.0, tol=1e-6):
    """
    Check that no two inner hexagons overlap using separating axis test.
    Returns True if all pairs are separated.
    """
    n = len(positions)
    all_verts = _get_all_vertices(positions, angles, side)
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(positions[i,0], positions[i,1], side, angles[i])
            vj = _hex_vertices(positions[j,0], positions[j,1], side, angles[j])
            # Get edge normals for both hexagons
            # For hexagon vi, edges are between consecutive vertices
            edges_i = np.diff(np.vstack([vi, vi[0]]), axis=0)
            edges_j = np.diff(np.vstack([vj, vj[0]]), axis=0)
            # Normals are perpendicular to edges
            norms_i = np.column_stack([-edges_i[:,1], edges_i[:,0]])
            norms_j = np.column_stack([-edges_j[:,1], edges_j[:,0]])
            # Normalize
            norms_i = norms_i / np.linalg.norm(norms_i, axis=1, keepdims=True)
            norms_j = norms_j / np.linalg.norm(norms_j, axis=1, keepdims=True)
            all_norms = np.vstack([norms_i, norms_j])
            for norm in all_norms:
                proj_i = vi @ norm
                proj_j = vj @ norm
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    break
            else:
                return False
    return True

def _is_valid(positions, angles, side=1.0):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(angles)):
        return False
    if not _check_separation(positions, angles, side):
        return False
    all_verts = _get_all_vertices(positions, angles, side)
    s = _min_outer_side(all_verts)
    if s < np.sqrt(12) - 1e-8:
        return False
    # Check all inner vertices are inside outer hexagon
    outer_verts = _hex_vertices(0, 0, s, 0)
    for v in all_verts:
        inside = True
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k+1) % 6]
            cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
            if cross < -1e-6:
                inside = False
                break
        if not inside:
            return False
    return True

def _compute_score(positions, angles, side=1.0):
    """Compute 1/s for the given packing. Returns 0 if invalid."""
    if not _is_valid(positions, angles, side):
        return 0.0
    all_verts = _get_all_vertices(positions, angles, side)
    s = _min_outer_side(all_verts)
    return 1.0 / s

def _repulsion_forces(positions, angles, side=1.0, k=100.0, min_dist=1.8):
    """
    Compute repulsion forces between hexagons to push them apart.
    """
    n = len(positions)
    forces = np.zeros_like(positions)
    for i in range(n):
        for j in range(i+1, n):
            d = positions[i] - positions[j]
            dist = np.linalg.norm(d)
            if dist < min_dist and dist > 1e-10:
                force = k * (min_dist - dist) / dist
                forces[i] += force * d
                forces[j] -= force * d
    return forces

def _attract_to_center(positions, center, k=1.0):
    """Compute attractive forces toward center."""
    return k * (center - positions)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds deadline
    
    # Initial configuration: hexagonal grid arrangement
    # 12 hexagons in a compact arrangement
    # Use a pattern that's close to optimal
    # Center + 6 around + 5 more
    
    # Start with a known good configuration
    # Hexagonal lattice: each hex has 6 neighbors
    # For 12 hexagons, a good arrangement is:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons (offset)
    # But we need to fit them tightly.
    
    # Let's use a physics-based approach
    # Initial positions in a hexagonal pattern
    sqrt3 = np.sqrt(3)
    
    # Hexagonal grid positions for 12 hexagons
    # Using axial coordinates
    positions = np.array([
        [0, 0],
        [2.5, 0],
        [-2.5, 0],
        [1.25, 2.165],
        [-1.25, 2.165],
        [1.25, -2.165],
        [-1.25, -2.165],
        [3.75, 2.165],
        [-3.75, 2.165],
        [3.75, -2.165],
        [-3.75, -2.165],
        [0, -4.33],
    ])
    angles = np.zeros(12)
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_score = _compute_score(best_positions, best_angles)
    
    # Physics-based optimization
    dt = 0.01
    repulsion_k = 50.0
    attraction_k = 0.5
    damping = 0.9
    
    velocities = np.zeros_like(positions)
    
    max_iters = 50000
    for iteration in range(max_iters):
        if time.time() > deadline - 60:
            break
        
        # Compute forces
        rep_forces = _repulsion_forces(positions, angles, k=repulsion_k, min_dist=1.8)
        att_forces = _attract_to_center(positions, np.zeros(2), k=attraction_k)
        
        total_forces = rep_forces + att_forces
        
        # Update velocities and positions
        velocities = damping * velocities + total_forces * dt
        positions = positions + velocities * dt
        
        # Keep angles at 0 for simplicity (all hexagons aligned)
        
        # Check if we've found a better configuration
        current_score = _compute_score(positions, angles)
        if current_score > best_score:
            best_score = current_score
            best_positions = positions.copy()
            best_angles = angles.copy()
        
        # Adaptive step size: reduce if score decreases
        if current_score < best_score * 0.95:
            dt *= 0.5
            repulsion_k *= 0.9
        else:
            # Gradually increase attraction to compact
            attraction_k *= 1.001
    
    # Final polish: local optimization
    # Try to further optimize by adjusting positions slightly
    polish_dt = 0.001
    polish_iters = 5000
    positions = best_positions.copy()
    velocities = np.zeros_like(positions)
    
    for iteration in range(polish_iters):
        if time.time() > deadline - 30:
            break
        
        rep_forces = _repulsion_forces(positions, angles, k=repulsion_k * 0.5, min_dist=1.8)
        att_forces = _attract_to_center(positions, np.zeros(2), k=attraction_k * 0.5)
        total_forces = rep_forces + att_forces
        
        velocities = 0.9 * velocities + total_forces * polish_dt
        positions = positions + velocities * polish_dt
        
        current_score = _compute_score(positions, angles)
        if current_score > best_score:
            best_score = current_score
            best_positions = positions.copy()
            best_angles = angles.copy()
    
    # Use the best found configuration
    inner_hex_data = np.column_stack([best_positions, best_angles])
    all_verts = _get_all_vertices(best_positions, best_angles, side=1.0)
    s = _min_outer_side(all_verts)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
