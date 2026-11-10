# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _compute_min_side(inner, outer_center=None):
    """Compute the minimum side length of outer hexagon needed to contain all inner hexagons."""
    if outer_center is None:
        outer_center = np.array([0.0, 0.0])
    
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= outer_center[0]
    inner_shifted[:, 1] -= outer_center[1]
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(11):
        verts = _hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], 1.0, inner_shifted[i, 2])
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # The 6 edge normals of a regular hexagon centered at origin (angle 0)
    normals = np.column_stack([np.cos(np.radians(30 + 60 * np.arange(6))), 
                                 np.sin(np.radians(30 + 60 * np.arange(6)))])
    
    # For each normal direction, compute max projection
    projections = all_verts @ normals.T  # shape (66, 6)
    max_projs = projections.max(axis=0)  # shape (6,)
    
    # The side length s must satisfy: s * sqrt(3)/2 >= max_projs
    # So s >= max_projs * 2 / sqrt(3)
    min_side = np.max(max_projs) * 2.0 / np.sqrt(3)
    
    return min_side


def _check_separation(inner, tol=1e-6):
    """Check if all pairs of inner hexagons are separated."""
    for i in range(11):
        for j in range(i+1, 11):
            vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            
            # Get edge normals for both hexagons
            def get_edge_normals(verts):
                normals = []
                for k in range(6):
                    e = verts[(k+1) % 6] - verts[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
                return normals
            
            ni = get_edge_normals(vi)
            nj = get_edge_normals(vj)
            
            # Check on at least one of 12 edge normals the projected intervals are apart by > tol
            separated = False
            for n in ni + nj:
                proj_i = vi @ n
                proj_j = vj @ n
                max_i = proj_i.max()
                min_i = proj_i.min()
                max_j = proj_j.max()
                min_j = proj_j.min()
                
                if max_i < min_j - tol or max_j < min_i - tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _build_rigid_cluster(global_rot_deg, satellite_offsets, flower_gap=0.001):
    """
    Build the 11 hexagons from a rigid cluster topology.
    
    Topology: 7 hexagons in a flower (center + 6 around at distance 2+gap)
              + 4 satellite hexagons at specified offsets from center.
    
    All hexagons share the same angle = global_rot_deg.
    
    satellite_offsets: array of shape (4, 2) with (dx, dy) for each satellite.
    flower_gap: small gap added to flower hexagon distances to ensure separation.
    """
    sqrt3 = np.sqrt(3)
    
    # Flower positions (relative to center, before rotation)
    # Distance from center to neighbors is 2 + flower_gap to ensure separation
    d = 2.0 + flower_gap
    flower = np.array([
        [0.0, 0.0],           # center
        [d, 0.0],             # 0 deg
        [d/2, d*sqrt3/2],     # 60 deg
        [-d/2, d*sqrt3/2],    # 120 deg
        [-d, 0.0],            # 180 deg
        [-d/2, -d*sqrt3/2],   # 240 deg
        [d/2, -d*sqrt3/2],    # 300 deg
    ])
    
    # Rotate flower by global_rot_deg
    theta = np.radians(global_rot_deg)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    flower_rotated = (rot_matrix @ flower.T).T
    
    # Add satellites
    satellites = satellite_offsets.copy()
    satellites_rotated = (rot_matrix @ satellites.T).T
    
    # Combine all 11 hexagons
    positions = np.vstack([flower_rotated, satellites_rotated])
    
    # All have the same angle
    inner = np.column_stack([positions[:, 0], positions[:, 1], 
                             np.full(11, global_rot_deg)])
    
    return inner


def _local_search(inner, deadline, rng, max_iter=2000):
    """Local search to minimize outer side length."""
    current_side = _compute_min_side(inner)
    best_inner = inner.copy()
    best_side = current_side
    
    x = inner[:, 0].copy()
    y = inner[:, 1].copy()
    a_deg = inner[:, 2].copy()
    
    no_improve = 0
    max_no_improve = 50
    step_size = 0.1
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break
        
        improved = False
        for trial in range(100):
            if time.time() > deadline:
                break
            
            # Random perturbation
            dx = rng.normal(0, step_size, 11)
            dy = rng.normal(0, step_size, 11)
            da = rng.normal(0, step_size, 11)
            
            # Keep center fixed for symmetry
            dx[0] = 0
            dy[0] = 0
            da[0] = 0
            
            trial_x = x + dx
            trial_y = y + dy
            trial_a = a_deg + da
            
            trial_inner = np.column_stack([trial_x, trial_y, trial_a])
            
            # Quick separation check
            if not _check_separation(trial_inner):
                continue
            
            trial_side = _compute_min_side(trial_inner)
            
            if trial_side < current_side - 1e-12:
                x = trial_x.copy()
                y = trial_y.copy()
                a_deg = trial_a.copy()
                current_side = trial_side
                best_inner = trial_inner.copy()
                best_side = trial_side
                improved = True
                no_improve = 0
        
        if not improved:
            no_improve += 1
            step_size *= 0.95  # Decrease step size
    
    return best_inner, best_side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    sqrt3 = np.sqrt(3)
    rng = np.random.RandomState(42)
    
    best_inner = None
    best_side = np.inf
    
    # Try multiple initial configurations and optimize each
    
    # Configuration 1: Flower with satellites at extended positions
    for global_rot in [0, 15, 30, 45, 60, 75, 90]:
        if time.time() > deadline - 30:
            break
        
        # Different satellite offset patterns
        satellite_patterns = [
            # Pattern 1: Satellites at 30, 150, 210, 330 degrees, distance ~3.5
            np.array([
                [3.5 * np.cos(np.radians(30)), 3.5 * np.sin(np.radians(30))],
                [3.5 * np.cos(np.radians(150)), 3.5 * np.sin(np.radians(150))],
                [3.5 * np.cos(np.radians(210)), 3.5 * np.sin(np.radians(210))],
                [3.5 * np.cos(np.radians(330)), 3.5 * np.sin(np.radians(330))],
            ]),
            # Pattern 2: Satellites at 0, 90, 180, 270 degrees, distance ~3.2
            np.array([
                [3.2, 0],
                [0, 3.2],
                [-3.2, 0],
                [0, -3.2],
            ]),
            # Pattern 3: Satellites at 30, 90, 150, 270 degrees, distance ~3.0
            np.array([
                [3.0 * np.cos(np.radians(30)), 3.0 * np.sin(np.radians(30))],
                [0, 3.0],
                [3.0 * np.cos(np.radians(150)), 3.0 * np.sin(np.radians(150))],
                [0, -3.0],
            ]),
            # Pattern 4: Linear extension
            np.array([
                [3.0, 0],
                [-3.0, 0],
                [1.5, 2.6],
                [-1.5, -2.6],
            ]),
        ]
        
        for sat_offsets in satellite_patterns:
            if time.time() > deadline - 30:
                break
            
            inner = _build_rigid_cluster(global_rot, sat_offsets, flower_gap=0.001)
            
            if not _check_separation(inner):
                continue
            
            side = _compute_min_side(inner)
            
            if side < best_side:
                best_side = side
                best_inner = inner.copy()
    
    # If we have a valid configuration, do local search
    if best_inner is not None:
        # Do local search on the best configuration
        improved_inner, improved_side = _local_search(best_inner, deadline - 10, rng)
        
        if improved_side < best_side:
            best_inner = improved_inner
            best_side = improved_side
    
    # Fallback if no valid configuration found
    if best_inner is None:
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        best_side = 8.0
    
    # Final verification
    if not _check_separation(best_inner):
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        best_side = 8.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return best_inner, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END
