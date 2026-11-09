"""
Strategy: Binary Search on Outer Side Length + Force-Directed Relaxation.
Instead of minimizing a penalty function directly, we fix the outer hexagon size 
and use physics simulation to push inner hexagons into a valid, non-overlapping 
configuration within that boundary. If a valid configuration is found, we shrink 
the boundary and repeat. This separates the "feasibility" check from the 
"optimization" step, often leading to tighter packs.
"""

import numpy as np
import time
import random
import math
import sys

N = 11
DEADLINE = 145.0  # Safety margin
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _get_face_normals(angle_deg):
    """Return unit normals for the 6 faces of a hexagon at given angle."""
    angle_rad = np.radians(angle_deg)
    normals = []
    for k in range(6):
        # Face normal is perpendicular to edge. 
        # For a hexagon with vertex at 0 deg, face normal is at 30 deg.
        a = angle_rad + np.pi/6 + k * np.pi / 3
        normals.append([np.cos(a), np.sin(a)])
    return np.array(normals)

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """
    Calculate the minimum side length required to contain all inner hexagons.
    Returns the exact minimum side length based on vertex projections.
    """
    cx, cy = outer_center
    face_normals = _get_face_normals(outer_angle)
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        iverts_rel = iverts - np.array(outer_center)
        
        # Project vertices onto face normals
        # For a hexagon, the distance from center to face is inradius = s * sqrt(3)/2
        # A point is inside if dot(p, n) <= inradius for all normals n.
        # So s >= 2 * max(dot(p, n)) / sqrt(3)
        
        # We need max over all vertices and all normals.
        # Note: The set of normals must cover the directions of the faces.
        # Since the hexagon is regular, the support function in direction n is 
        # determined by the vertex farthest in that direction.
        
        for n in face_normals:
            projs = iverts_rel @ n
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
                
    return 2.0 * max_proj / np.sqrt(3)

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """
    Strictly check if the configuration is valid:
    1. All inner hexagons are inside the outer hexagon.
    2. No inner hexagons overlap.
    """
    cx, cy = outer_center
    outer_normals = _get_face_normals(outer_angle)
    inradius = outer_side * np.sqrt(3) / 2.0
    
    # Pre-compute vertices for all inner hexagons
    all_verts = []
    for i in range(N):
        ix, iy, iangle = inner_positions[i]
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        all_verts.append(verts)
        
    # 1. Check Containment
    for i in range(N):
        verts = all_verts[i]
        verts_rel = verts - np.array(outer_center)
        for n in outer_normals:
            projs = verts_rel @ n
            if np.max(projs) > inradius + tol:
                return False
                
    # 2. Check Disjointness (SAT - Separating Axis Theorem)
    for i in range(N):
        for j in range(i + 1, N):
            v1 = all_verts[i]
            v2 = all_verts[j]
            
            # Get edge normals for both hexagons
            # Normal to edge k is perpendicular to edge vector
            # For a regular hexagon, the normals are simply the face normals
            # derived from the angle.
            n1 = _get_face_normals(inner_positions[i][2])
            n2 = _get_face_normals(inner_positions[j][2])
            
            disjoint = False
            for n in np.vstack([n1, n2]):
                proj1 = v1 @ n
                proj2 = v2 @ n
                
                # If there is a gap in projection, they are disjoint
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    disjoint = True
                    break
            
            if not disjoint:
                return False
                
    return True

def _force_directed_relaxation(initial_positions, outer_side, outer_center, outer_angle, max_iters=2000, dt=0.01):
    """
    Simulate forces to push hexagons apart and into the container.
    Returns final positions if valid, else None.
    """
    # State: x, y, angle for each hex
    # We use a simple Euler integration with damping
    pos = np.array(initial_positions, dtype=float)
    vel = np.zeros_like(pos)
    
    # Parameters
    k_repel = 10.0   # Repulsion strength
    k_contain = 50.0 # Containment strength (stronger to keep inside)
    damping = 0.9    # Velocity damping
    
    cx, cy = outer_center
    outer_normals = _get_face_normals(outer_angle)
    inradius = outer_side * np.sqrt(3) / 2.0
    
    # Pre-compute some constants
    sqrt3 = np.sqrt(3)
    
    for step in range(max_iters):
        if _remaining_time() < 5.0:
            return None
            
        forces = np.zeros_like(pos)
        
        # 1. Containment Forces
        # If a vertex is outside, push the center back.
        for i in range(N):
            x, y, a = pos[i]
            verts = _hex_vertices(x, y, 1.0, a)
            verts_rel = verts - np.array(outer_center)
            
            # Calculate penetration for each face
            push_x = 0.0
            push_y = 0.0
            
            for k in range(6):
                n = outer_normals[k]
                projs = verts_rel @ n
                max_proj = np.max(projs)
                
                if max_proj > inradius:
                    penetration = max_proj - inradius
                    # Force is proportional to penetration, direction is -n
                    push_x -= k_contain * penetration * n[0]
                    push_y -= k_contain * penetration * n[1]
            
            forces[i, 0] += push_x
            forces[i, 1] += push_y
            
        # 2. Repulsion Forces between hexagons
        # Approximate using distance between centers. 
        # For hexagons of side 1, min center distance for no overlap is 2*sqrt(3) ~ 3.464 if aligned.
        # But with rotation, it varies. 
        # A more robust "soft" repulsion: 
        # If distance < D_min, apply force.
        # D_min depends on orientation. Let's use a conservative D_min = 3.0 for force start.
        # Actually, let's use a potential based on the minimum distance between vertices if possible,
        # but that's expensive. 
        # Heuristic: Use center distance.
        # If d < 2 * circumradius (2.0), definitely overlap.
        # If d < 2 * inradius (1.732), definitely overlap.
        # Let's use a smooth repulsion: F = k * (R - d) if d < R.
        # R should be roughly the sum of circumradii = 2.0.
        # However, hexagons can interlock.
        # Let's use a stronger repulsion if they are very close.
        
        for i in range(N):
            for j in range(i + 1, N):
                di = pos[i]
                dj = pos[j]
                diff = di[:2] - dj[:2]
                d = np.linalg.norm(diff)
                
                # Ideal distance is roughly 2*sqrt(3) for side-by-side, 
                # but they can be closer if rotated.
                # Let's define a "hard" radius for repulsion.
                # If d < 2.0, they are likely overlapping or very close.
                # If d < 1.5, definitely overlapping.
                
                # Use a potential that grows as d decreases.
                # U = 0.5 * k * (R - d)^2 if d < R
                # F = k * (R - d) * (di - dj)/d
                
                # We need to account for the fact that hexagons are not circles.
                # But for relaxation, circular approximation works if we tune k.
                # Let's use R = 2.2 (slightly larger than circumradius sum) to be safe.
                R = 2.2 
                
                if d < R:
                    if d < 1e-6:
                        # Random push
                        angle = random.uniform(0, 2*np.pi)
                        fx = np.cos(angle)
                        fy = np.sin(angle)
                    else:
                        fx = diff[0] / d
                        fy = diff[1] / d
                        
                    # Force magnitude
                    # Stronger force for deeper overlap
                    f_mag = k_repel * (R - d)
                    
                    # Apply Newton's 3rd law
                    forces[i, 0] += f_mag * fx
                    forces[i, 1] += f_mag * fy
                    forces[j, 0] -= f_mag * fx
                    forces[j, 1] -= f_mag * fy
                    
                    # Torque? 
                    # If we don't model torque, rotations are fixed.
                    # The task allows rotation. 
                    # For simplicity in this "first try" of the new strategy, 
                    # we might fix rotations to 0 or random initial values 
                    # and only optimize positions. 
                    # However, the current best program optimizes angles.
                    # Let's try fixing angles to 0 first to see if position-only 
                    # relaxation works. If not, we can add angle dynamics.
                    # Given the complexity, let's assume optimal packing 
                    # for hexagons often aligns them.
                    # We will keep angles fixed from initialization in this phase.
        
        # Update velocities and positions
        vel += forces * dt
        vel *= damping
        pos += vel * dt
        
        # Constrain positions to a reasonable area to prevent drift
        # The outer hexagon side is S. Max distance from center is S.
        # Let's bound x, y to [-S, S]
        bound = outer_side + 2.0
        pos[:, 0] = np.clip(pos[:, 0], -bound, bound)
        pos[:, 1] = np.clip(pos[:, 1], -bound, bound)
        
        # Check for convergence/validity periodically
        if step % 50 == 0:
            if _check_validity(pos, outer_center, outer_side, outer_angle, tol=1e-4):
                return pos.tolist()
                
    # Final check
    if _check_validity(pos, outer_center, outer_side, outer_angle, tol=1e-4):
        return pos.tolist()
        
    return None

def _initial_config_random(outer_side):
    """Generate a random initial configuration inside the outer hexagon."""
    pos = []
    # Center one in the middle
    pos.append([0.0, 0.0, 0.0])
    
    for i in range(1, N):
        # Sample random point inside the hexagon
        # Rejection sampling: sample in square, check if inside
        while True:
            x = random.uniform(-outer_side, outer_side)
            y = random.uniform(-outer_side, outer_side)
            
            # Check if (x,y) is inside the outer hexagon
            # Using support function
            inside = True
            for n in _get_face_normals(0.0):
                if x * n[0] + y * n[1] > outer_side * np.sqrt(3) / 2:
                    inside = False
                    break
            
            if inside:
                break
                
        # Random angle
        angle = random.uniform(0, 360)
        pos.append([x, y, angle])
        
    return pos

def _initial_config_lattice(outer_side):
    """Generate a lattice-like initial configuration."""
    pos = []
    pos.append([0.0, 0.0, 0.0])
    
    # Hexagonal lattice spacing for side 1 is 2*sqrt(3)
    dist = 2.0 * np.sqrt(3) * 0.95 # Slightly compressed to encourage relaxation
    
    for k in range(6):
        a = np.radians(60 * k)
        pos.append([dist * np.cos(a), dist * np.sin(a), 0.0])
        
    # Add more if needed (N=11)
    # The lattice above has 7. We need 4 more.
    # Place them in the "gaps" or further out.
    dist2 = dist * 1.5
    for k in range(4):
        a = np.radians(30 + 60 * k)
        pos.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
        
    return pos

def construct():
    """
    Main entry point.
    1. Binary search on outer side length S.
    2. For each S, try multiple random/lattice initializations.
    3. Use force-directed relaxation to find a valid packing.
    4. If found, shrink S. If not, expand S.
    """
    random.seed(42)
    np.random.seed(42)
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Initial bounds for binary search
    # Lower bound: theoretical minimum for 11 hexagons.
    # Area of 11 hexagons = 11 * (3*sqrt(3)/2) ~ 28.5
    # Area of outer hexagon with side S = (3*sqrt(3)/2) * S^2
    # S^2 >= 28.5 / (3*sqrt(3)/2) ~ 11. S >= ~3.3
    # However, packing efficiency is < 1. 
    # Let's start low and go up if it fails, or start high and go down.
    # Starting high and going down is safer for "finding" a solution.
    
    # Known good score is around 5.0. 
    # Let's search between 4.5 and 6.0.
    lo = 4.5
    hi = 6.0
    
    best_side = hi
    best_pos = None
    
    # We will perform a series of "shrinking" attempts.
    # Instead of strict binary search which might get stuck, 
    # we will try to find a valid config for a given S, 
    # and if successful, set S = S * 0.98 and repeat.
    # If we fail for S, we might increase S slightly.
    
    current_side = 6.0
    
    # Try to find a valid config at current_side
    # If found, shrink. If not, expand.
    
    attempts = 0
    max_attempts = 100
    
    while _remaining_time() > 10.0 and attempts < max_attempts:
        attempts += 1
        
        found_valid = False
        valid_pos = None
        
        # Try multiple initializations
        # 1. Lattice
        # 2. Random x3
        inits = [
            _initial_config_lattice(current_side),
            _initial_config_random(current_side),
            _initial_config_random(current_side),
            _initial_config_random(current_side)
        ]
        
        for init in inits:
            if _remaining_time() < 5.0:
                break
                
            res = _force_directed_relaxation(init, current_side, outer_center, outer_angle, max_iters=500)
            if res is not None:
                # Verify strictly
                if _check_validity(res, outer_center, current_side, outer_angle, tol=1e-5):
                    found_valid = True
                    valid_pos = res
                    break
        
        if found_valid:
            # Update best
            if current_side < best_side:
                best_side = current_side
                best_pos = valid_pos
                
            # Shrink
            current_side *= 0.99
        else:
            # Failed to find valid config. Expand.
            current_side *= 1.02
            
            # If we've expanded too much, reset to a known loose config?
            # Or just keep expanding until we find one, then shrink.
            
            # Safety: if we go above 7.0, something is wrong with the simulator
            if current_side > 7.0:
                current_side = 6.0
                # Force a restart with different random seed?
                random.seed(random.randint(0, 10000))
                np.random.seed(random.randint(0, 10000))
                
    if best_pos is None:
        # Fallback: return a very loose configuration
        # Just place them in a circle
        pos = []
        pos.append([0.0, 0.0, 0.0])
        for k in range(10):
            a = 2 * np.pi * k / 10
            r = 4.0
            pos.append([r * np.cos(a), r * np.sin(a), 0.0])
        side = _min_enclosing_side(pos, outer_center, outer_angle)
        return pos, outer_center, side, outer_angle
        
    return best_pos, outer_center, best_side, outer_angle

if __name__ == "__main__":
    res = construct()
    print(res)
