import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0

# Precompute geometry constants
_BASE_ANGLES = np.deg2rad(60.0 * np.arange(6))
_BASE_COS = np.cos(_BASE_ANGLES)
_BASE_SIN = np.sin(_BASE_ANGLES)
_BASE_VERTS = np.column_stack([_BASE_COS, _BASE_SIN])  # (6, 2) unit hex vertices

_NORMAL_ANGLES = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
_NORMALS = np.column_stack([np.cos(_NORMAL_ANGLES), np.sin(_NORMAL_ANGLES)])  # (6, 2)


def _min_containing_hex_side(centers):
    """
    Compute minimum side length s of a regular hexagon centered at (0,0), angle 0,
    that contains all unit hexagons with given centers.
    """
    n = len(centers)
    # All vertices: (n, 6, 2)
    verts = centers[:, np.newaxis, :] + _BASE_VERTS[np.newaxis, :, :]
    verts_flat = verts.reshape(-1, 2)  # (36, 2)
    projections = _NORMALS @ verts_flat.T  # (6, 36)
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _separation_violation(centers):
    """Total overlap violation across all pairs (for penalty)."""
    n = len(centers)
    verts = centers[:, np.newaxis, :] + _BASE_VERTS[np.newaxis, :, :]  # (n, 6, 2)
    
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            proj_i = verts[i] @ _NORMALS.T  # (6,)
            proj_j = verts[j] @ _NORMALS.T  # (6,)
            min_i = proj_i.min()
            max_i = proj_i.max()
            min_j = proj_j.min()
            max_j = proj_j.max()
            # For SAT: separated if any normal has gap
            # Overlap on each axis:
            overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            max_overlap = np.max(overlaps)
            if max_overlap > 0:
                total += max_overlap ** 2
    return total


def _check_separation(centers, tol=1e-6):
    """Check if all pairs of unit hexagons are separated using SAT."""
    n = len(centers)
    verts = centers[:, np.newaxis, :] + _BASE_VERTS[np.newaxis, :, :]
    
    for i in range(n):
        for j in range(i + 1, n):
            proj_i = verts[i] @ _NORMALS.T
            proj_j = verts[j] @ _NORMALS.T
            min_i = proj_i.min()
            max_i = proj_i.max()
            min_j = proj_j.min()
            max_j = proj_j.max()
            # Separated if there exists a normal where intervals don't overlap
            # i.e., max_i < min_j - tol OR max_j < min_i - tol
            if not (max_i < min_j - tol or max_j < min_i - tol):
                return False
    return True


def _hex_lattice_12():
    """
    Hexagonal lattice: 1 center + 6 ring-1 + 5 ring-2 (asymmetric ring-2 for 12 total).
    Actually let's use a proper 12-hex arrangement.
    Standard: center + 6 neighbors + 5 of the 12 ring-2 positions.
    But better: use a 3x3-ish arrangement clipped.
    
    For 12 hexagons in hexagonal close packing:
    - 1 at center
    - 6 at distance sqrt(3) (ring 1)
    - 5 at ring 2 (distances 2*sqrt(3) or sqrt(3+3)=sqrt(6)...)
    
    Actually the standard hex packing of 12:
    Row structure (offset):
    Row 0: 3 hexes
    Row 1: 4 hexes  
    Row 2: 3 hexes
    Row 3: 2 hexes
    Total: 12
    
    Let me use the axial coordinate approach.
    """
    sqrt3 = np.sqrt(3)
    # Use a hexagonal arrangement
    # Center at origin, ring 1 at 6 positions, ring 2 at 12 positions
    # Pick 1 center + 6 ring-1 + 5 ring-2
    
    positions = []
    # Center
    positions.append([0.0, 0.0])
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(60.0 * k)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        positions.append([x, y])
    
    # Ring 2: pick 5 of the 12 positions
    # Ring 2 positions are at:
    # 6 at distance 2*sqrt(3) (same angles as ring 1)
    # 6 at distance sqrt(3)*sqrt(3) = 3 (angles offset by 30)
    # Actually ring 2 in hex lattice:
    # For hex with side 1, adjacent hex centers are at distance sqrt(3)
    # Ring 2: 12 positions at various distances
    
    # Let me just use a known compact arrangement
    # 12 hexagons: use a "flower" with partial second ring
    # Positions for ring 2 (distance from center):
    # 6 at 2*sqrt(3) at angles 0,60,...
    # 6 at 3 at angles 30,90,...
    # Pick 5 from these 12
    
    ring2_positions = []
    for k in range(6):
        angle = np.deg2rad(60.0 * k)
        ring2_positions.append([2.0 * sqrt3 * np.cos(angle), 2.0 * sqrt3 * np.sin(angle)])
    for k in range(6):
        angle = np.deg2rad(60.0 * k + 30.0)
        ring2_positions.append([3.0 * np.cos(angle), 3.0 * np.sin(angle)])
    
    # Pick 5 ring-2 positions that are most "inward" or symmetric
    # For symmetry, pick every other one from the 12
    # Actually for 12 total, let's just pick 5 that give good coverage
    # Use positions 0, 2, 4, 6, 8 (alternating)
    for idx in [0, 2, 4, 6, 8]:
        positions.append(ring2_positions[idx])
    
    return np.array(positions)


def _compact_12():
    """
    A more compact initial arrangement using a triangular/hexagonal pattern.
    4-4-4 rows or 3-4-3-2 or similar.
    """
    sqrt3 = np.sqrt(3)
    # Use a rectangular-ish arrangement optimized for hex packing
    # Row spacing: sqrt(3) for adjacent rows
    # Column spacing: 1.5 for same row, offset by 0.75 for alternating
    
    # Let's do: 4 in row 0, 4 in row 1, 4 in row 2 (with offsets)
    # Actually for hex packing, let's use:
    # 3 in top row, 4 in second, 3 in third, 2 in bottom
    positions = []
    
    # Row 0 (top): 3 hexes
    for i in range(3):
        positions.append([i * 1.5 - 1.5, 0.0])
    
    # Row 1: 4 hexes, offset by 0.75 in x, shifted by sqrt(3)/2 in y
    for i in range(4):
        positions.append([i * 1.5 - 2.25, -sqrt3 / 2])
    
    # Row 2: 3 hexes
    for i in range(3):
        positions.append([i * 1.5 - 1.5, -sqrt3])
    
    # Row 3: 2 hexes
    for i in range(2):
        positions.append([i * 1.5 - 0.75, -1.5 * sqrt3])
    
    return np.array(positions)


def _slsqp_objective(params):
    """Objective for SLSQP: minimize containing hex side length."""
    centers = params.reshape(12, 2)
    return _min_containing_hex_side(centers)


def _slsqp_constraints(centers):
    """
    Build constraint functions for SLSQP.
    For each pair (i,j) and each of the 6 SAT normals,
    we need: max_proj_i - min_proj_j <= -tol  OR  max_proj_j - min_proj_i <= -tol
    This is an OR constraint which is tricky.
    
    Instead, use a penalty-based approach with SLSQP:
    Minimize s + lambda * sum of violation^2
    
    Actually, let's use a different approach:
    For each pair, the minimum distance between the two hexagons must be >= 0.
    The distance between two convex polygons can be computed, but it's complex.
    
    Alternative: use the fact that for two unit hexagons with the same orientation,
    they are separated iff their centers are at distance >= sqrt(3) in the hex lattice sense.
    But that's not quite right for arbitrary positions.
    
    Let me use a simpler constraint: for each pair, the minimum over all 6 normals of
    (separation on that normal) must be >= 0.
    Separation on normal k = min(max_i_k - min_j_k, max_j_k - min_i_k)
    But this needs to be >= 0 for AT LEAST ONE normal.
    
    This is hard to express as SLSQP constraints directly.
    Let me use a penalty approach instead.
    """
    pass


def _penalty_objective(params, penalty_weight=50.0):
    """Objective with penalty for separation violations."""
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    violation = _separation_violation(centers)
    return s + penalty_weight * violation


def _slsqp_with_constraints(params):
    """
    Use SLSQP with inequality constraints for separation.
    For each pair (i,j), we add a constraint that the minimum gap
    across all SAT axes is >= -tol.
    
    The gap on axis k for pair (i,j):
    gap_k = min(max_i_k - min_j_k, max_j_k - min_i_k)
    We need max_k(gap_k) >= 0 (i.e., at least one axis separates them).
    
    This is equivalent to: NOT (for all k, gap_k < 0)
    Which is: OR over k of (gap_k >= 0)
    
    SLSQP doesn't handle OR constraints well. Let's use a different formulation.
    
    Alternative: For two convex polygons with the same orientation (both angle 0),
    they are separated iff the distance between their centers projected onto
    any of the 6 edge normals gives a gap >= 0.
    
    Actually, for two congruent regular hexagons with the same orientation,
    the SAT test simplifies: they are separated iff the center-to-center distance
    projected onto some normal is >= 2 * apothem = 2 * (sqrt(3)/2) * 1 = sqrt(3).
    
    Wait, that's for the case where the hexagons are axis-aligned. Let me verify:
    For a unit hexagon (side 1), the apothem (inradius) is sqrt(3)/2.
    The width across flats is sqrt(3).
    So two such hexagons with the same orientation are separated iff
    the projection of their center difference onto some edge normal >= sqrt(3).
    
    The 6 edge normals are at angles 30, 90, 150, 210, 270, 330.
    
    So the constraint for pair (i,j) is:
    max over k of (dot(center_j - center_i, normal_k)) >= sqrt(3)
    OR
    max over k of (dot(center_i - center_j, normal_k)) >= sqrt(3)
    
    Which is equivalent to:
    max over k of |dot(center_j - center_i, normal_k)| >= sqrt(3)
    
    This is a nice simplification! For two unit hexagons with the same orientation,
    they are separated iff the maximum projection of their center difference onto
    the 6 edge normals is >= sqrt(3).
    
    Let me verify: if two hexagons are just touching (edges aligned), the center
    distance along the normal is exactly 2 * apothem = sqrt(3). Yes!
    
    So the constraint is:
    max_k |dot(d_ij, n_k)| >= sqrt(3) for all pairs (i,j).
    
    This can be written as 6 constraints per pair (one for each normal), but with an OR.
    However, we can use the fact that:
    max_k |dot(d, n_k)| >= sqrt(3)
    is equivalent to:
    NOT (for all k, |dot(d, n_k)| < sqrt(3))
    
    For SLSQP, let's use a smooth approximation or a penalty.
    """
    pass


def _hex_separation_constraint(params, i, j, k):
    """
    Constraint for pair (i,j) on normal k:
    |dot(center_j - center_i, normal_k)| >= sqrt(3) - tol
    """
    centers = params.reshape(12, 2)
    d = centers[j] - centers[i]
    proj = np.abs(np.dot(d, _NORMALS[k]))
    return proj - np.sqrt(3.0) + 1e-7


def _make_slsqp_constraints():
    """
    Create all constraint functions.
    For each pair (i,j) and each normal k, we have a constraint.
    But we only need ONE of the 6 to be satisfied (OR logic).
    
    To handle this, we'll use a different approach:
    For each pair, define a single constraint as:
    max_k |dot(d_ij, n_k)| - sqrt(3) >= 0
    
    This is a max of 6 functions, which is non-smooth but SLSQP can handle it
    with finite differences.
    """
    constraints = []
    n = 12
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(params, i=i, j=j):
                centers = params.reshape(12, 2)
                d = centers[j] - centers[i]
                projs = np.abs(d @ _NORMALS.T)  # (6,)
                return np.max(projs) - np.sqrt(3.0) + 1e-7
            constraints.append({'type': 'ineq', 'fun': constraint_func})
    return constraints


def _nelder_mead_objective(params):
    """Nelder-Mead objective with strong penalty."""
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    violation = _separation_violation(centers)
    return s + 1000.0 * violation


def _local_search(centers, n_iters=200, step_size=0.05):
    """
    Simple local search: try small perturbations and keep improvements.
    """
    centers = centers.copy()
    current_s = _min_containing_hex_side(centers)
    if not _check_separation(centers):
        return None
    
    rng = np.random.default_rng(123)
    
    for it in range(n_iters):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Try perturbing one or two random centers
        n_perturb = rng.integers(1, 3)
        indices = rng.choice(12, size=n_perturb, replace=False)
        perturbations = rng.normal(0, step_size, size=(n_perturb, 2))
        
        new_centers = centers.copy()
        new_centers[indices] += perturbations
        
        if _check_separation(new_centers):
            new_s = _min_containing_hex_side(new_centers)
            if new_s < current_s - 1e-12:
                centers = new_centers
                current_s = new_s
                step_size *= 0.95  # Shrink step on success
            else:
                step_size *= 0.99
        else:
            step_size *= 0.98
    
    return centers


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    """
    best_centers = None
    best_s = np.inf
    
    rng = np.random.default_rng(42)
    
    initial_configs = [
        _hex_lattice_12(),
        _compact_12(),
    ]
    
    # Add some randomized initial configs
    for r in range(4):
        base = _compact_12()
        noise = rng.normal(0, 0.2 + 0.1 * r, size=base.shape)
        initial_configs.append(base + noise)
    
    for init_idx, init_centers in enumerate(initial_configs):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # First, check if the initial config is valid
        if _check_separation(init_centers):
            s_init = _min_containing_hex_side(init_centers)
            if s_init < best_s:
                best_s = s_init
                best_centers = init_centers.copy()
        else:
            # Need to fix overlaps - use a quick relaxation
            # Simple iterative push-apart
            centers = init_centers.copy()
            for it in range(500):
                elapsed = time.time() - _START_TIME
                if elapsed > _DEADLINE - _SAFETY_MARGIN:
                    break
                
                if _check_separation(centers):
                    break
                
                # For each overlapping pair, push apart
                improved = False
                for i in range(12):
                    for j in range(i + 1, 12):
                        d = centers[j] - centers[i]
                        projs = np.abs(d @ _NORMALS.T)
                        max_proj = np.max(projs)
                        
                        if max_proj < np.sqrt(3.0) - 1e-7:
                            # Find the normal with max projection
                            k_max = np.argmax(projs)
                            n_dir = _NORMALS[k_max]
                            # Push apart along this normal
                            if np.dot(d, n_dir) > 0:
                                centers[i] -= 0.1 * n_dir
                                centers[j] += 0.1 * n_dir
                            else:
                                centers[i] += 0.1 * n_dir
                                centers[j] -= 0.1 * n_dir
                            improved = True
                
                if not improved:
                    break
            
            if _check_separation(centers):
                s_init = _min_containing_hex_side(centers)
                if s_init < best_s:
                    best_s = s_init
                    best_centers = centers.copy()
        
        # Phase 1: SLSQP optimization with separation constraints
        if best_centers is not None:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                x0 = best_centers.flatten()
                try:
                    constraints = _make_slsqp_constraints()
                    result = minimize(
                        _slsqp_objective,
                        x0,
                        method='SLSQP',
                        constraints=constraints,
                        options={
                            'maxiter': 3000,
                            'ftol': 1e-12,
                        }
                    )
                    
                    if result.success or result.fun < best_s:
                        candidate = result.x.reshape(12, 2)
                        if _check_separation(candidate):
                            s_cand = _min_containing_hex_side(candidate)
                            if s_cand < best_s:
                                best_s = s_cand
                                best_centers = candidate.copy()
                except Exception:
                    pass
        
        # Phase 2: Nelder-Mead with penalty (from current best)
        if best_centers is not None:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                x0 = best_centers.flatten()
                try:
                    result = minimize(
                        _nelder_mead_objective,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 10000,
                            'maxfev': 100000,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )
                    
                    candidate = result.x.reshape(12, 2)
                    if _check_separation(candidate):
                        s_cand = _min_containing_hex_side(candidate)
                        if s_cand < best_s:
                            best_s = s_cand
                            best_centers = candidate.copy()
                except Exception:
                    pass
        
        # Phase 3: Local search refinement
        if best_centers is not None:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                refined = _local_search(best_centers, n_iters=500, step_size=0.03)
                if refined is not None:
                    s_ref = _min_containing_hex_side(refined)
                    if s_ref < best_s:
                        best_s = s_ref
                        best_centers = refined.copy()
        
        # Phase 4: SLSQP from the Nelder-Mead result
        if best_centers is not None:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                x0 = best_centers.flatten()
                try:
                    constraints = _make_slsqp_constraints()
                    result = minimize(
                        _slsqp_objective,
                        x0,
                        method='SLSQP',
                        constraints=constraints,
                        options={
                            'maxiter': 5000,
                            'ftol': 1e-14,
                        }
                    )
                    
                    candidate = result.x.reshape(12, 2)
                    if _check_separation(candidate):
                        s_cand = _min_containing_hex_side(candidate)
                        if s_cand < best_s:
                            best_s = s_cand
                            best_centers = candidate.copy()
                except Exception:
                    pass
    
    # Fallback
    if best_centers is None:
        best_centers = _compact_12().copy()
        best_s = _min_containing_hex_side(best_centers)
    
    # Final polish with SLSQP
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        x0_final = best_centers.flatten()
        try:
            constraints = _make_slsqp_constraints()
            result_final = minimize(
                _slsqp_objective,
                x0_final,
                method='SLSQP',
                constraints=constraints,
                options={
                    'maxiter': 8000,
                    'ftol': 1e-15,
                }
            )
            
            candidate = result_final.x.reshape(12, 2)
            if _check_separation(candidate):
                s_cand = _min_containing_hex_side(candidate)
                if s_cand < best_s:
                    best_s = s_cand
                    best_centers = candidate.copy()
        except Exception:
            pass
    
    # Final local search
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        refined = _local_search(best_centers, n_iters=1000, step_size=0.01)
        if refined is not None:
            s_ref = _min_containing_hex_side(refined)
            if s_ref < best_s:
                best_s = s_ref
                best_centers = refined.copy()
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
