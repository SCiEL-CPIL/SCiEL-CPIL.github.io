# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0

# Precompute constants
_HEX_ANGLES = np.deg2rad(60.0 * np.arange(6))
_HEX_COS = np.cos(_HEX_ANGLES)
_HEX_SIN = np.sin(_HEX_ANGLES)
_HEX_VERTS_OFFSET = np.column_stack([_HEX_COS, _HEX_SIN])

# Normals for separation (SAT)
_NORMAL_ANGLES = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
_NORMAL_COS = np.cos(_NORMAL_ANGLES)
_NORMAL_SIN = np.sin(_NORMAL_ANGLES)
_NORMALS = np.column_stack([_NORMAL_COS, _NORMAL_SIN])

# Outer hexagon normals (for containment)
# Vertices of outer hex at angle 0 are at angles 0, 60, ...
# Edges are at angles 30, 90, ...
# Normal to edge connecting v0 and v1 (at 0 and 60 deg) points to 30 deg.
# So normals are same as _NORMALS.

def _min_containing_hex_side(centers):
    """
    Compute the minimum side length s of a regular hexagon centered at (0,0), angle 0,
    that contains all unit hexagons with given centers.
    """
    # Vertices of all inner hexagons
    # (12, 6, 2)
    verts = centers[:, np.newaxis, :] + _HEX_VERTS_OFFSET[np.newaxis, :, :]
    verts_flat = verts.reshape(-1, 2) # (72, 2)
    
    # Project onto 6 normal directions
    projections = verts_flat @ _NORMALS.T # (72, 6)
    max_projections = np.max(projections, axis=0) # (6,)
    
    # For a regular hexagon of side s, the distance from center to edge is s * sqrt(3) / 2.
    # The max projection of vertices on a normal is s * sqrt(3) / 2.
    # So s = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
    max_proj = np.max(max_projections)
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s

def _separation_violation(centers):
    """
    Compute total separation violation (how much pairs overlap).
    Returns 0 if all pairs are separated, positive value otherwise.
    Uses SAT.
    """
    n = len(centers)
    # Vertices
    verts = centers[:, np.newaxis, :] + _HEX_VERTS_OFFSET[np.newaxis, :, :]
    
    total_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = verts[i] # (6, 2)
            vj = verts[j] # (6, 2)
            
            # Project onto 6 normals
            proj_i = vi @ _NORMALS.T # (6, 6)
            proj_j = vj @ _NORMALS.T # (6, 6)
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            # Overlap on each axis
            overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            max_overlap = np.max(overlaps)
            
            if max_overlap > 0:
                total_violation += max_overlap ** 2
                
    return total_violation

def _check_separation(centers, tol=1e-6):
    """Check if all pairs of unit hexagons are separated using SAT."""
    n = len(centers)
    verts = centers[:, np.newaxis, :] + _HEX_VERTS_OFFSET[np.newaxis, :, :]
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = verts[i]
            vj = verts[j]
            
            proj_i = vi @ _NORMALS.T
            proj_j = vj @ _NORMALS.T
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            # Separated if on any axis, one interval is completely to the left of the other
            # i.e., max_i < min_j - tol OR max_j < min_i - tol
            separated = np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol)
            if not separated:
                return False
    return True

def _objective(params):
    """
    Objective function for optimization.
    Minimizes s + penalty * violation.
    """
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    violation = _separation_violation(centers)
    # Strong penalty for overlap
    penalty = 1000.0 * violation
    return s + penalty

def _make_initial_config_1():
    """
    Hexagonal lattice: 1 center + 6 ring + 5 ring 2.
    This is a valid packing.
    """
    sqrt3 = np.sqrt(3)
    return np.array([
        [0.0, 0.0],
        [1.5, 0.866025],
        [1.5, -0.866025],
        [0.0, 1.732051],
        [0.0, -1.732051],
        [-1.5, 0.866025],
        [-1.5, -0.866025],
        [3.0, 0.0],
        [2.25, 2.598076],
        [0.75, 3.464102],
        [-1.5, 2.598076],
        [-3.0, 0.0],
    ])

def _make_initial_config_2():
    """
    Slightly perturbed lattice to break symmetry.
    """
    base = _make_initial_config_1()
    rng = np.random.default_rng(123)
    return base + rng.normal(0, 0.1, size=base.shape)

def _make_initial_config_3():
    """
    Random cluster.
    """
    rng = np.random.default_rng(456)
    # Start with a cluster and expand
    centers = np.zeros((12, 2))
    centers[0] = [0,0]
    for i in range(1, 12):
        # Place near a random existing hex
        parent = rng.integers(0, i)
        angle = rng.uniform(0, 2*np.pi)
        dist = 1.8 # slightly more than touching
        centers[i] = centers[parent] + dist * np.array([np.cos(angle), np.sin(angle)])
    return centers

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    best_centers = None
    best_s = np.inf
    
    initial_configs = [
        _make_initial_config_1(),
        _make_initial_config_2(),
        _make_initial_config_3()
    ]
    
    for init_idx, init_centers in enumerate(initial_configs):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
            
        # Initial check
        if _check_separation(init_centers):
            s_init = _min_containing_hex_side(init_centers)
            if s_init < best_s:
                best_s = s_init
                best_centers = init_centers.copy()
        
        # Optimization
        x0 = init_centers.flatten()
        
        try:
            # Use SLSQP for gradient-based optimization
            # We don't have explicit constraints, so we use penalty method in objective
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'disp': False
                }
            )
            
            if result.success:
                candidate = result.x.reshape(12, 2)
                if _check_separation(candidate):
                    s_cand = _min_containing_hex_side(candidate)
                    if s_cand < best_s:
                        best_s = s_cand
                        best_centers = candidate.copy()
                        
        except Exception as e:
            pass
            
        # Also try Nelder-Mead from the best found so far if we have time
        if best_centers is not None:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                x0_nm = best_centers.flatten()
                try:
                    result_nm = minimize(
                        _objective,
                        x0_nm,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 2000,
                            'maxfev': 20000,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True,
                        }
                    )
                    if result_nm.success:
                        candidate_nm = result_nm.x.reshape(12, 2)
                        if _check_separation(candidate_nm):
                            s_cand_nm = _min_containing_hex_side(candidate_nm)
                            if s_cand_nm < best_s:
                                best_s = s_cand_nm
                                best_centers = candidate_nm.copy()
                except Exception:
                    pass

    # Fallback
    if best_centers is None:
        best_centers = _make_initial_config_1()
        best_s = _min_containing_hex_side(best_centers)
        
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
