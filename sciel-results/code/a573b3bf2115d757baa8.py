# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(cx, cy, s, a_deg):
    """Return 6 edge outward normals (unit vectors) for a hexagon."""
    verts = _hex_vertices(cx, cy, s, a_deg)
    normals = []
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        edge = v2 - v1
        # Outward normal: perpendicular to edge, pointing away from center
        n = np.array([-edge[1], edge[0]])
        n = n / np.linalg.norm(n)
        # Ensure it points outward (away from center)
        mid = (v1 + v2) / 2
        if np.dot(n, mid - np.array([cx, cy])) < 0:
            n = -n
        normals.append(n)
    return np.array(normals)


def _hexagons_separated(pos1, ang1, pos2, ang2, tol=1e-6):
    """Check if two unit hexagons are separated (not intersecting) using 12 edge normals."""
    verts1 = _hex_vertices(pos1[0], pos1[1], 1.0, ang1)
    verts2 = _hex_vertices(pos2[0], pos2[1], 1.0, ang2)
    normals1 = _hex_edge_normals(pos1[0], pos1[1], 1.0, ang1)
    normals2 = _hex_edge_normals(pos2[0], pos2[1], 1.0, ang2)
    all_normals = np.vstack([normals1, normals2])
    for n in all_normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Separated if intervals are apart
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False


def _hexagon_in_outer(cx, cy, ang, outer_center, outer_side, outer_ang, tol=1e-6):
    """Check if unit hexagon is inside outer hexagon."""
    verts = _hex_vertices(cx, cy, 1.0, ang)
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_ang)
    # For each outer edge, check that all inner vertices are on the correct side
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        # Normal pointing inward (toward center)
        n = np.array([-edge[1], edge[0]])
        n = n / np.linalg.norm(n)
        # Ensure inward: dot with (center - p1) should be positive
        if np.dot(n, outer_center - p1) < 0:
            n = -n
        # All inner vertices must satisfy: dot(p - p1, n) >= -tol
        for v in verts:
            if np.dot(v - p1, n) < -tol:
                return False
    return True


def _valid_packing(inner_data, outer_center, outer_side, outer_ang):
    """Full validity check."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_center)):
        return False
    if outer_side < np.sqrt(11) - 1e-9:
        return False
    # Check all pairs are separated
    n = len(inner_data)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_separated(inner_data[i, :2], inner_data[i, 2],
                                       inner_data[j, :2], inner_data[j, 2]):
                return False
    # Check all inner hexagons are inside outer
    for i in range(n):
        if not _hexagon_in_outer(inner_data[i, 0], inner_data[i, 1], inner_data[i, 2],
                                 outer_center, outer_side, outer_ang):
            return False
    return True


def _minimal_enclosing_hexagon(inner_data):
    """Find the minimal regular hexagon (centered at origin, axis-aligned) that contains all inner hexagons."""
    # For a hexagon centered at origin with side s and angle 0:
    # Vertices at s*(cos(60i), sin(60i)) for i=0..5
    # A point (x,y) is inside if for all 6 edges, dot(p - v_i, n_i) >= 0
    # The constraint reduces to: max over 6 directions of (|x*cos(a) + y*sin(a)|) <= s * cos(30 deg) = s*sqrt(3)/2
    # Actually, for a regular hexagon with side s centered at origin, the inradius is s*sqrt(3)/2
    # and the apothem (distance from center to each edge) is s*sqrt(3)/2
    # A point is inside iff max_i |dot(p, n_i)| <= s*sqrt(3)/2 where n_i are the 6 outward normals
    # The 6 outward normals for a hexagon with angle 0 are at angles 30, 90, 150, 210, 270, 330 degrees
    # Wait, let me think more carefully.
    # Hexagon vertices at angle 0: (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # Edge from (s,0) to (s/2, s*sqrt(3)/2): direction = (-s/2, s*sqrt(3)/2), normal pointing outward
    # Actually, the apothem (inradius) of a regular hexagon with side s is s * sqrt(3) / 2.
    # The condition for a point p to be inside is: for each of the 6 edge normals n_i, dot(p, n_i) <= apothem
    # where n_i are unit normals pointing outward.
    # For a hexagon with angle a, the outward normals are at angles a + 30 + 60*i for i=0..5
    # Let's just compute the required side length by checking all 6 half-planes.
    
    best_side = 0.0
    for i in range(len(inner_data)):
        verts = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        for v in verts:
            # For outer hexagon centered at origin with angle 0:
            # Outward normals at angles 30, 90, 150, 210, 270, 330 degrees
            for k in range(6):
                ang = np.radians(30 + 60 * k)
                n = np.array([np.cos(ang), np.sin(ang)])
                dist = np.dot(v, n)
                # Need apothem >= dist, so s * sqrt(3)/2 >= dist, s >= 2*dist/sqrt(3)
                req_side = 2.0 * dist / np.sqrt(3)
                if req_side > best_side:
                    best_side = req_side
    return best_side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses rigid-body dynamics with symplectic integrator.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    n = 11
    
    # Initial positions: hexagonal lattice arrangement
    # Center + 6 around + 4 more in second ring
    init_positions = np.array([
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
    ], dtype=float)
    
    init_angles = np.zeros(n)
    
    best_score = 0.0
    best_inner = None
    
    # Symplectic (velocity Verlet) integrator
    dt = 0.005
    damping = 0.999  # slight damping to settle
    repulsion_strength = 50.0
    center_attraction = 0.1
    
    def compute_forces(positions, angles):
        """Compute forces on each hexagon."""
        forces = np.zeros_like(positions)
        torques = np.zeros(n)
        
        # Pairwise repulsion: push apart if overlapping
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.array([1e-10, 0])
                    dist = 1e-10
                # Overlap threshold: two unit hexagons touching have center distance ~2.5 (for aligned)
                # Use a soft repulsion that activates when centers are too close
                threshold = 2.4  # approximate minimum center distance for non-overlap
                if dist < threshold:
                    # Strong repulsive force
                    f = repulsion_strength * (threshold - dist) / dist
                    forces[i] -= f * d
                    forces[j] += f * d
                    # Also add torque to encourage alignment
                    torque_mag = 0.5 * (threshold - dist)
                    # Compute relative angle
                    da = (angles[j] - angles[i]) % 360
                    if da > 180:
                        da -= 360
                    # Torque to align angles
                    torque = torque_mag * np.sin(np.radians(da))
                    torques[i] -= torque
                    torques[j] += torque
        
        # Attractive force toward center
        for i in range(n):
            forces[i] -= center_attraction * positions[i]
        
        return forces, torques
    
    # Run dynamics
    positions = init_positions.copy()
    velocities = np.zeros_like(positions)
    angles = init_angles.copy()
    angular_velocities = np.zeros(n)
    
    # Simulate for a fixed number of steps, checking deadline
    max_steps = 200000
    step = 0
    while step < max_steps:
        if time.time() > deadline:
            break
        
        forces, torques = compute_forces(positions, angles)
        
        # Velocity Verlet
        # v_half = v + 0.5*dt*a
        velocities_half = velocities + 0.5 * dt * forces
        angular_velocities_half = angular_velocities + 0.5 * dt * torques
        
        # x_new = x + dt*v_half
        positions_new = positions + dt * velocities_half
        angles_new = angles + dt * angular_velocities_half
        
        # Compute new forces at new position
        forces_new, torques_new = compute_forces(positions_new, angles_new)
        
        # v_new = v_half + 0.5*dt*a_new
        velocities = velocities_half + 0.5 * dt * forces_new
        angular_velocities = angular_velocities_half + 0.5 * dt * torques_new
        
        # Apply damping
        velocities *= damping
        angular_velocities *= damping
        
        positions = positions_new
        angles = angles_new
        
        step += 1
        
        # Check validity periodically
        if step % 500 == 0:
            outer_side = _minimal_enclosing_hexagon(np.column_stack([positions, angles]))
            if _valid_packing(np.column_stack([positions, angles]), np.array([0.0, 0.0]), outer_side, 0.0):
                score = 1.0 / outer_side
                if score > best_score:
                    best_score = score
                    best_inner = np.column_stack([positions, angles]).copy()
    
    # If we found a valid packing, use it
    if best_inner is not None:
        outer_side = _minimal_enclosing_hexagon(best_inner)
        outer_data = np.array([0.0, 0.0, 0.0])
        return best_inner, outer_data, outer_side
    
    # Fallback: use initial grid arrangement
    inner_hex_data = np.column_stack([init_positions, init_angles])
    outer_side = _minimal_enclosing_hexagon(inner_hex_data)
    if outer_side < np.sqrt(11):
        outer_side = np.sqrt(11) + 1.0
    outer_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_data, outer_side


# EVOLVE-BLOCK-END
