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
    """positions: (N,2), angles: (N,) in degrees. Returns (N*6, 2)."""
    verts = []
    for i in range(len(positions)):
        verts.append(_hex_vertices(positions[i,0], positions[i,1], side, angles[i]))
    return np.vstack(verts)

def _min_outer_side(vertices):
    """Compute minimum side length s of a regular hexagon centered at origin (angle 0)."""
    normals = np.array([
        [np.cos(np.radians(30)), np.sin(np.radians(30))],
        [np.cos(np.radians(90)), np.sin(np.radians(90))],
        [np.cos(np.radians(150)), np.sin(np.radians(150))],
        [np.cos(np.radians(210)), np.sin(np.radians(210))],
        [np.cos(np.radians(270)), np.sin(np.radians(270))],
        [np.cos(np.radians(330)), np.sin(np.radians(330))],
    ])
    projections = vertices @ normals.T
    max_proj = np.max(np.abs(projections), axis=1)
    max_overall = np.max(max_proj)
    return max_overall / (np.sqrt(3) / 2)

def _check_nonoverlap(positions, angles, side=1.0, tol=1e-6):
    """Fast SAT check: returns True if no two hexagons overlap."""
    n = len(positions)
    for i in range(n):
        vi = _hex_vertices(positions[i,0], positions[i,1], side, angles[i])
        for j in range(i+1, n):
            vj = _hex_vertices(positions[j,0], positions[j,1], side, angles[j])
            edges_i = np.diff(np.vstack([vi, vi[0]]), axis=0)
            edges_j = np.diff(np.vstack([vj, vj[0]]), axis=0)
            norms_i = np.column_stack([-edges_i[:,1], edges_i[:,0]])
            norms_j = np.column_stack([-edges_j[:,1], edges_j[:,0]])
            norms_i /= np.linalg.norm(norms_i, axis=1, keepdims=True)
            norms_j /= np.linalg.norm(norms_j, axis=1, keepdims=True)
            all_norms = np.vstack([norms_i, norms_j])
            separated = False
            for norm in all_norms:
                proj_i = vi @ norm
                proj_j = vj @ norm
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    separated = True
                    break
            if not separated:
                return False
    return True

def _compute_all_vertices(positions, angles, side=1.0):
    return _get_all_vertices(positions, angles, side)

def _initial_honeycomb():
    """Generate a good initial honeycomb configuration for 12 hexagons."""
    # Hexagonal lattice with spacing 2 (touching)
    # Center + ring of 6 + partial second ring
    sqrt3 = np.sqrt(3)
    # Axial to cartesian: x = 2.5*q + 1.25*r, y = (3/2)*r * (2/sqrt3) ... 
    # Simpler: use known positions for 12 hexagons in compact hex arrangement
    # Row-based: 3-4-3-2 or similar
    # Let's use a proper hex grid
    positions = np.array([
        [0.0, 0.0],
        [2.5, 0.0],
        [-2.5, 0.0],
        [1.25, 2.1650635],
        [-1.25, 2.1650635],
        [1.25, -2.1650635],
        [-1.25, -2.1650635],
        [3.75, 2.1650635],
        [-3.75, 2.1650635],
        [3.75, -2.1650635],
        [-3.75, -2.1650635],
        [0.0, 4.3301270],
    ])
    angles = np.zeros(12)
    return positions, angles

def _objective(positions, angles, side=1.0):
    """Compute 1/s if valid, else 0."""
    if not _check_nonoverlap(positions, angles, side):
        return 0.0
    all_verts = _compute_all_vertices(positions, angles, side)
    s = _min_outer_side(all_verts)
    if s < np.sqrt(12) - 1e-8:
        return 0.0
    # Verify containment
    outer_verts = _hex_vertices(0, 0, s, 0)
    for v in all_verts:
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k+1) % 6]
            cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
            if cross < -1e-6:
                return 0.0
    return 1.0 / s

def hexagon_packing_12():
    """Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon."""
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    positions, angles = _initial_honeycomb()
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_score = _objective(best_positions, best_angles)
    
    # Simulated annealing with position and angle optimization
    n = 12
    T = 0.5  # initial temperature
    cooling = 0.9995
    min_T = 0.001
    max_iters = 200000
    
    # Precompute edge directions for SAT (they depend on angle)
    for iteration in range(max_iters):
        if time.time() > deadline - 30:
            break
        if T < min_T:
            break
        
        # Generate random perturbation
        new_positions = positions.copy()
        new_angles = angles.copy()
        
        # Perturb 1-3 random hexagons
        n_perturb = np.random.randint(1, 4)
        indices = np.random.choice(n, size=n_perturb, replace=False)
        
        for idx in indices:
            # Position perturbation
            scale = T * 2.0
            new_positions[idx] += np.random.normal(0, scale, 2)
            # Angle perturbation (small, in degrees)
            new_angles[idx] += np.random.normal(0, T * 10.0)
        
        # Evaluate
        new_score = _objective(new_positions, new_angles)
        
        # Accept if better or with probability exp((new_score - best_score)/T)
        if new_score >= best_score:
            positions = new_positions
            angles = new_angles
            if new_score > best_score:
                best_score = new_score
                best_positions = positions.copy()
                best_angles = angles.copy()
        else:
            delta = new_score - best_score
            if delta > -T * 0.5 and np.random.random() < np.exp(delta / max(T, 1e-10)):
                positions = new_positions
                angles = new_angles
        
        T *= cooling
    
    # Final local refinement with smaller steps
    for refinement_round in range(5):
        if time.time() > deadline - 15:
            break
        T_ref = 0.01 * (0.5 ** refinement_round)
        for _ in range(5000):
            if time.time() > deadline - 15:
                break
            new_positions = best_positions.copy()
            new_angles = best_angles.copy()
            idx = np.random.randint(0, n)
            new_positions[idx] += np.random.normal(0, T_ref, 2)
            new_angles[idx] += np.random.normal(0, T_ref * 5.0)
            new_score = _objective(new_positions, new_angles)
            if new_score > best_score:
                best_score = new_score
                best_positions = new_positions
                best_angles = new_angles
    
    inner_hex_data = np.column_stack([best_positions, best_angles])
    all_verts = _get_all_vertices(best_positions, best_angles, side=1.0)
    s = _min_outer_side(all_verts)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
