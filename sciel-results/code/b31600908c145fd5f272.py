"""Packing 12 unit hexagons in a regular hexagon using caging formulation.
Minimizes outer side by computing minimal enclosing hexagon from inner vertices.
Uses multi-start differential evolution + local refinement.
"""
import numpy as np
import time
import math
from scipy.optimize import differential_evolution

N = 12
SQRT3 = math.sqrt(3.0)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_all_inner_vertices(inner_positions):
        """Get all 72 vertices from 12 inner hexagons."""
        verts = []
        for i in range(N):
            cx, cy, angle = inner_positions[i]
            v = hexagon_vertices(cx, cy, 1.0, angle)
            verts.append(v)
        return np.vstack(verts)

    def min_enclosing_hexagon_side(verts, center_x, center_y, angle_deg):
        """
        Compute minimum side length of a regular hexagon centered at (cx, cy)
        with given rotation that contains all vertices.
        The hexagon has 6 half-planes; side = max over all constraints.
        """
        angle_rad = math.radians(angle_deg)
        # Hexagon normals at 30, 90, 150, 210, 270, 330 degrees + rotation
        normal_angles = np.radians(np.arange(30, 360, 60)) + angle_rad
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Translate vertices to center
        v_local = verts - np.array([center_x, center_y])
        
        # For each normal direction, compute max projection
        # side >= max_projection / (sqrt(3)/2) = max_projection * 2/sqrt(3)
        max_proj = np.max(v_local @ normals.T, axis=0)
        # side = max_proj * 2/sqrt(3), but also need to handle negative projections
        # Actually for a regular hexagon centered at origin, the constraint is:
        # n_i · v <= apothem for all i, where apothem = side * sqrt(3)/2
        # So side >= max(n_i · v) * 2/sqrt(3)
        min_side = np.max(max_proj) * 2.0 / SQRT3
        return max(min_side, 0.0)

    def objective(params):
        """
        Objective: minimize outer hexagon side length.
        params: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, cx, cy, outer_angle]
        """
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        cx = params[3*N]
        cy = params[3*N+1]
        oa = params[3*N+2]
        
        # Check pairwise disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not polygons_disjoint_sat_fast(inner[i], inner[j]):
                    return 1e6  # large penalty for overlap
        
        verts = get_all_inner_vertices(inner)
        side = min_enclosing_hexagon_side(verts, cx, cy, oa)
        return side

    def polygons_disjoint_sat_fast(pos1, pos2):
        """Fast SAT check for two hexagons."""
        v1 = hexagon_vertices(pos1[0], pos1[1], 1.0, pos1[2])
        v2 = hexagon_vertices(pos2[0], pos2[1], 1.0, pos2[2])
        
        # Precompute edge normals
        axes = []
        for i in range(6):
            edge = v1[(i+1)%6] - v1[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n / norm)
        for i in range(6):
            edge = v2[(i+1)%6] - v2[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n / norm)
        
        for axis in axes:
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            if np.max(proj1) < np.min(proj2) - 1e-10 or np.max(proj2) < np.min(proj1) - 1e-10:
                return True
        return False

    def is_valid_config(inner, cx, cy, side, oa):
        """Validate configuration."""
        # Check pairwise disjoint
        for i in range(N):
            for j in range(i+1, N):
                if not polygons_disjoint_sat_fast(inner[i], inner[j]):
                    return False
        # Check containment
        verts = get_all_inner_vertices(inner)
        actual_side = min_enclosing_hexagon_side(verts, cx, cy, oa)
        return actual_side <= side + 1e-6

    def hexagonal_lattice_config(seed_offset=0):
        """Generate a hexagonal lattice configuration."""
        positions = []
        # Use a 4x3 grid with hexagonal spacing
        dx = 3.0
        dy = 3.0 * SQRT3 / 2
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * dx
                y = (row - 1) * dy
                if row % 2 == 1:
                    x += dx / 2
                positions.append([x, y, 0.0])
        return positions

    def ring_config():
        """Central + 6 ring + 5 outer."""
        positions = [[0.0, 0.0, 0.0]]
        for k in range(6):
            angle = k * math.pi / 3
            x = 2.0 * math.cos(angle)
            y = 2.0 * math.sin(angle)
            positions.append([x, y, 0.0])
        for k in range(5):
            angle = k * math.pi / 3 + math.pi / 6
            x = 3.464 * math.cos(angle)
            y = 3.464 * math.sin(angle)
            positions.append([x, y, 0.0])
        return positions

    def two_rows_config():
        """Two staggered rows of 6."""
        positions = []
        for i in range(6):
            positions.append([(i - 2.5) * 2.2, -1.5, 0.0])
            positions.append([(i - 2.5) * 2.2 + 1.1, 1.5, 0.0])
        return positions

    def compact_grid_config():
        """Tighter 3x4 grid."""
        positions = []
        dx = 2.2
        dy = 2.7
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * dx
                y = (row - 1) * dy
                if row % 2 == 1:
                    x += dx / 2
                positions.append([x, y, 0.0])
        return positions

    # Generate initial configurations
    base_configs = [
        hexagonal_lattice_config(),
        ring_config(),
        two_rows_config(),
        compact_grid_config()
    ]

    best_side = float('inf')
    best_params = None

    # Try each base configuration with differential evolution
    for cfg_idx, base_pos in enumerate(base_configs):
        if time.time() > deadline - 30:
            break

        # Create initial parameter vector
        init_params = []
        for p in base_pos:
            init_params.extend([p[0], p[1], p[2]])
        init_params.extend([0.0, 0.0, 0.0])  # center and angle

        # Define bounds for differential evolution
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))  # x
            bounds.append((-10, 10))  # y
            bounds.append((0, 180))  # angle
        bounds.append((-5, 5))  # cx
        bounds.append((-5, 5))  # cy
        bounds.append((0, 180))  # outer angle

        # Run differential evolution with limited iterations
        max_iter = 50
        popsize = 15
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=max_iter,
                popsize=popsize,
                tol=0.01,
                mutation=(0.3, 0.8),
                recombination=0.7,
                seed=42 + cfg_idx,
                init='latinhypercube',
                polish=False,
                updating='immediate'
            )
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x.copy()
        except Exception as e:
            pass

    # Local refinement with Nelder-Mead on best result
    if best_params is not None:
        from scipy.optimize import minimize
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If no valid result, use fallback
    if best_params is None:
        positions = hexagonal_lattice_config()
        inner = [(p[0], p[1], p[2]) for p in positions]
        cx, cy = 0.0, 0.0
        oa = 0.0
        verts = get_all_inner_vertices(inner)
        side = min_enclosing_hexagon_side(verts, cx, cy, oa)
        best_params = np.array([p for pos in positions for p in pos] + [cx, cy, oa])
        best_side = side

    # Parse best params
    inner = []
    for i in range(N):
        inner.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
    cx = best_params[3*N]
    cy = best_params[3*N+1]
    oa = best_params[3*N+2]

    # Compute exact minimum side
    verts = get_all_inner_vertices(inner)
    side = min_enclosing_hexagon_side(verts, cx, cy, oa)

    # Validate and adjust if needed
    if not is_valid_config(inner, cx, cy, side, oa):
        # Expand side slightly
        side *= 1.01
        # If still invalid, spread inner hexagons
        if not is_valid_config(inner, cx, cy, side, oa):
            for i in range(N):
                inner[i] = (inner[i][0] * 1.1, inner[i][1] * 1.1, inner[i][2])
            verts = get_all_inner_vertices(inner)
            side = min_enclosing_hexagon_side(verts, cx, cy, oa) * 1.01

    return inner, (cx, cy), side, oa
