# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    """Compute all 66 vertices for 11 hexagons. positions (11,2), angles (11,)"""
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    """Check all pairs separated using SAT on 12 edge normals per pair."""
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    
    # Precompute edge normals for each hexagon (6 normals each)
    # Outward normal of edge i points at angle a + 30 + 60*i degrees
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]  # (6,2)
            vj = all_verts[j]
            # 12 normals: 6 from hex i, 6 from hex j
            normals = np.vstack([hex_normals[i], hex_normals[j]])  # (12,2)
            
            # Project all vertices onto each normal
            proj_i = vi @ normals.T  # (6, 12)
            proj_j = vj @ normals.T  # (6, 12)
            
            max_i = np.max(proj_i, axis=0)  # (12,)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            
            # Check if any normal separates them
            # Separated if max_i < min_j - tol OR max_j < min_i - tol
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue  # separated
            
            return False  # not separated
    return True

def _min_enclosing_side(positions, angles):
    """Min side length of outer hexagon (centered at origin, angle=0) containing all vertices."""
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    
    # Outer hexagon at angle 0: Edge normals at angles 30, 90, 150, 210, 270, 330
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    
    projs = all_verts @ normals.T  # (66, 6)
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    """Check all inner vertices inside outer hexagon (center=origin, angle=0)."""
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _compute_forces(positions, angles, outer_side):
    """
    Compute forces on each hexagon based on:
    1. Separation constraints (push apart if overlapping/too close)
    2. Containment constraints (push inward if hitting boundary)
    3. Centering force (pull towards origin to minimize outer hex size)
    """
    n = len(positions)
    forces = np.zeros((n, 2))
    
    # 1. Separation Forces
    # We use a soft penalty based on minimum distance between vertices.
    # For efficiency, we check center-to-center distance first.
    # If centers are close, we do a detailed vertex check.
    all_verts = _all_vertices(positions, angles)
    
    # Precompute normals for SAT
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            
            # Quick reject: if center distance > 2 * sqrt(3) + epsilon, no force
            # Max distance between vertices of two touching hexes is roughly 2*sqrt(3) ~ 3.46
            # Actually, if they are separated, distance between centers must be > 2*sqrt(3) if aligned?
            # No, for unit hexes, min center distance for non-overlap is 2*sqrt(3) if flat-to-flat?
            # Distance between parallel edges is sqrt(3). So center distance is 2*sqrt(3).
            # If distance > 3.5, they are definitely separated.
            dist_sq = np.sum((positions[i] - positions[j])**2)
            if dist_sq > 12.25: # 3.5^2
                continue
                
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            
            # Find the axis of minimum separation (or maximum overlap)
            # Separation vector for axis k:
            # If max_i < min_j, separation is min_j - max_i (positive means separated)
            # If max_j < min_i, separation is max_i - max_j (positive means separated)
            
            sep_vectors = np.zeros(12)
            # i pushes j away? No, we want force on i and j.
            # Let's define overlap amount.
            # Overlap on axis k = max(min_i - max_j, min_j - max_i)
            # If overlap > 0, they are intersecting.
            
            overlap_ij = np.max(min_i - max_j) # j is "left" of i? No, min_i < max_j means overlap
            overlap_ji = np.max(min_j - max_i)
            
            # Actually, let's just find the axis that gives the smallest "gap" (which can be negative).
            # Gap k = min( min_j - max_i, max_i - max_j )
            # If gap > 0, separated. If gap < 0, overlapping.
            
            gap_i_to_j = min_j - max_i
            gap_j_to_i = max_i - max_j
            
            # We want to push i and j apart along the axis that minimizes the gap.
            # Let's check all 12 axes.
            
            # Axis 0-5: normals of i.
            # If max_i < min_j - tol, separated.
            # If not, find axis where separation is minimal.
            
            # Let's compute the "penetration depth" and direction.
            # For each axis k:
            # If proj_i is to the left of proj_j (max_i < min_j), gap = min_j - max_i.
            # If proj_j is to the left of proj_i (max_j < min_i), gap = max_i - max_j.
            
            # We want to move i in direction -normal_k and j in direction +normal_k if i is "behind" j?
            # Let's simplify: Use the axis that gives the largest overlap.
            
            # Overlap vector direction:
            # If max_i > min_j and max_j > min_i, they overlap.
            
            # Let's just calculate the force as the sum of all "repulsions" from all axes that are close.
            # This is a soft body simulation approach.
            
            # For each axis k:
            # Projected centers? No, projected extents.
            
            # Let's use a simpler heuristic:
            # If they are close, compute the vector from center i to center j.
            # If the distance is less than a threshold (e.g., 2*sqrt(3) + 0.1), apply a repulsive force.
            # This is less precise but much faster and robust for local search.
            
            dist = np.sqrt(dist_sq)
            if dist < 3.5:
                # Repulsive force proportional to (threshold - dist)
                # Threshold for touching flat-to-flat is 2*sqrt(3) ~ 3.464
                # Threshold for touching vertex-to-vertex is 2*1.5? No.
                # Let's use 2*sqrt(3) as the "resting" distance.
                rest_dist = 2 * np.sqrt(3)
                if dist < rest_dist:
                    force_mag = (rest_dist - dist) * 10.0
                    dir_vec = (positions[i] - positions[j]) / dist
                    forces[i] += dir_vec * force_mag
                    forces[j] -= dir_vec * force_mag
                elif dist < rest_dist + 0.5:
                    # Small repulsion to keep them separated
                    force_mag = (rest_dist + 0.5 - dist) * 1.0
                    dir_vec = (positions[i] - positions[j]) / dist
                    forces[i] += dir_vec * force_mag
                    forces[j] -= dir_vec * force_mag

    # 2. Containment Forces
    # Outer hexagon side length S.
    # Vertices of outer hexagon are at distance S from center.
    # Apothem is S * sqrt(3)/2.
    # For a point (x,y), the distance to the boundary along the normal of the edge it's closest to.
    # If the point is outside, push it in.
    
    apothem = outer_side * np.sqrt(3) / 2
    
    # Normals of outer hexagon edges (pointing inward? No, outward from center)
    # Edge normals at angles 30, 90, 150, 210, 270, 330
    outer_normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    
    for i in range(n):
        # Check all vertices of hex i
        vi = all_verts[i]
        for v in vi:
            for k in range(6):
                n_vec = outer_normals[k]
                # Distance from center to plane of edge k is apothem.
                # Projection of v on n_vec.
                proj = np.dot(v, n_vec)
                # If proj > apothem, v is outside.
                if proj > apothem:
                    # Push hex i inward (opposite to n_vec)
                    # Force magnitude proportional to penetration
                    pen = proj - apothem
                    forces[i] -= n_vec * pen * 50.0
    
    # 3. Centering Force
    # Pull all hexes towards origin to minimize the bounding hexagon.
    for i in range(n):
        forces[i] -= positions[i] * 0.5
    
    return forces

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0
    
    d = np.sqrt(3)
    
    # Initial arrangement: Center + 6 ring + 4 in gaps
    # All angles = 0
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    
    positions[0] = [0.0, 0.0]
    for k in range(6):
        a = np.radians(30 + 60 * k)
        positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
    for k in range(0, 6, 2):
        a = np.radians(30 + 60 * k)
        positions[7 + k // 2] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
    a = np.radians(30 + 60 * 1)
    positions[10] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = _min_enclosing_side(best_positions, best_angles)
    
    # Ensure initial validity
    if not _check_separation_fast(best_positions, best_angles):
        # Scale up slightly
        scale = 1.0
        while not _check_separation_fast(best_positions, best_angles) and scale < 5.0:
            best_positions = best_positions * 1.01
            scale += 1.01
        best_positions[0] = [0.0, 0.0]
        best_side = _min_enclosing_side(best_positions, best_angles)
    
    # Simulation Parameters
    dt = 0.01
    damping = 0.9
    velocities = np.zeros((11, 2))
    
    # Adaptive step size
    step_scale = 1.0
    
    # Main Loop: Local Search via Force Simulation
    while time.time() < deadline - 10.0:
        # Compute forces
        forces = _compute_forces(best_positions, best_angles, best_side)
        
        # Update velocities
        velocities = velocities * damping + forces * dt
        
        # Limit velocity to prevent instability
        max_vel = 0.5
        vel_mags = np.linalg.norm(velocities, axis=1, keepdims=True)
        mask = vel_mags > max_vel
        if np.any(mask):
            velocities[mask] = velocities[mask] / vel_mags[mask] * max_vel
        
        # Propose new position
        new_positions = best_positions + velocities
        
        # Check separation
        if _check_separation_fast(new_positions, best_angles):
            # Check if it reduces the outer side
            new_side = _min_enclosing_side(new_positions, best_angles)
            
            # Accept if it improves the side length or if it's the same but more compact?
            # Since we are minimizing side, accept if new_side < best_side.
            # But we also need to ensure we don't get stuck in a local minimum where forces are zero but side is not minimal.
            # The centering force helps with that.
            
            if new_side < best_side - 1e-9:
                best_positions = new_positions
                best_side = new_side
                step_scale = min(step_scale * 1.05, 10.0) # Increase step if improving
            else:
                # If not improving, maybe we are stuck. Try a random kick.
                if np.random.random() < 0.1:
                    kick = np.random.randn(11, 2) * 0.1
                    trial_pos = best_positions + kick
                    if _check_separation_fast(trial_pos, best_angles):
                        trial_side = _min_enclosing_side(trial_pos, best_angles)
                        if trial_side < best_side:
                            best_positions = trial_pos
                            best_side = trial_side
                            velocities = np.zeros((11, 2))
                step_scale = max(step_scale * 0.95, 0.1) # Decrease step if not improving
        else:
            # Collision detected, reduce step and back off
            step_scale = max(step_scale * 0.5, 0.01)
            velocities = velocities * 0.5
            # Revert to previous position
            # best_positions remains unchanged
            # But we need to update velocities to reflect the bounce
            # Simple approach: just dampen velocity
            velocities = velocities * 0.5
        
        # Periodically try to optimize angles?
        # For now, keep angles fixed to 0 as per the successful strategy of the parent.
        # If we have time, we could try small angle perturbations.
        
        if time.time() < deadline - 30.0 and np.random.random() < 0.05:
            # Try rotating a random hexagon
            i = np.random.randint(11)
            delta_angle = np.random.choice([-5, 5, -10, 10])
            trial_angles = best_angles.copy()
            trial_angles[i] = (best_angles[i] + delta_angle) % 360
            
            if _check_separation_fast(best_positions, trial_angles):
                trial_side = _min_enclosing_side(best_positions, trial_angles)
                if trial_side < best_side - 1e-9:
                    best_angles = trial_angles
                    best_side = trial_side
    
    # Final verification and adjustment
    final_side = _min_enclosing_side(best_positions, best_angles)
    
    # Ensure strict separation
    if not _check_separation_fast(best_positions, best_angles):
        # Fallback to known valid configuration
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
        for k in range(0, 6, 2):
            a = np.radians(30 + 60 * k)
            positions[7 + k // 2] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        a = np.radians(30 + 60 * 1)
        positions[10] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)
    
    # Ensure containment
    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001
    
    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
