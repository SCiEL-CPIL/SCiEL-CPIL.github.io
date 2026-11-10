# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _compute_min_side_for_hexagon(all_vertices, a_deg, ox, oy):
    """
    Given all inner hexagon vertices, outer hexagon angle a_deg and center (ox,oy),
    compute the minimum side length s of the outer hexagon that contains all vertices.
    
    A regular hexagon with center (ox,oy), angle a, side s has 6 edges.
    The outward normal to edge i points in direction a + 60*i + 30 degrees.
    The apothem (distance from center to edge) = s * sqrt(3)/2.
    Constraint: for all vertices p and all edges i: (p - center) . n_i <= s * sqrt(3)/2
    So: s >= (2/sqrt(3)) * max_i max_p (p - center) . n_i
    """
    a = np.radians(a_deg)
    normals = []
    for i in range(6):
        theta = a + i * np.pi / 3 + np.pi / 6
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)  # (6, 2)
    
    diffs = all_vertices - np.array([ox, oy])  # (N, 2)
    projections = diffs @ normals.T  # (N, 6)
    max_proj = np.max(projections)
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return max(s, 1e-10)


def _optimize_outer_hexagon(all_vertices):
    """Optimize outer hexagon angle and center to minimize side length."""
    def objective(params):
        a_deg = params[0]
        ox, oy = params[1], params[2]
        return _compute_min_side_for_hexagon(all_vertices, a_deg, ox, oy)
    
    best_s = float('inf')
    best_params = None
    
    # Try multiple initializations
    for a0 in np.linspace(0, 60, 13):
        for ox0 in [-2, -1, 0, 1, 2]:
            for oy0 in [-2, -1, 0, 1, 2]:
                x0 = np.array([a0, ox0, oy0])
                try:
                    res = minimize(objective, x0, method='Nelder-Mead',
                                   options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 500})
                    if res.fun < best_s:
                        best_s = res.fun
                        best_params = res.x.copy()
                except:
                    pass
    
    return best_s, best_params


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    deadline = time.time() + 600
    
    # Honeycomb arrangement with unit hexagons
    # Center-to-center distance for touching unit hexagons = sqrt(3)
    d = np.sqrt(3)
    
    # Center + 6 ring1
    base_positions = [[0.0, 0.0]]
    for i in range(6):
        angle = i * np.pi / 3
        base_positions.append([d * np.cos(angle), d * np.sin(angle)])
    
    # Ring 2: 12 candidate positions
    ring2_candidates = []
    for i in range(6):
        angle = i * np.pi / 3
        ring2_candidates.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
        ring2_candidates.append([2 * d * np.cos(angle + np.pi/3), 2 * d * np.sin(angle + np.pi/3)])
    
    # Deduplicate
    ring2_unique = []
    for p in ring2_candidates:
        found = False
        for q in ring2_unique:
            if np.allclose(p, q, atol=1e-10):
                found = True
                break
        if not found:
            ring2_unique.append(p)
    
    # Try all C(12,4) = 495 combinations
    best_side = float('inf')
    best_inner = None
    best_outer_params = None
    
    for combo in combinations(range(len(ring2_unique)), 4):
        if time.time() > deadline - 30:
            break
        
        positions = base_positions + [ring2_unique[i] for i in combo]
        
        # Collect all vertices
        all_vertices_list = []
        for (x, y) in positions:
            verts = _hex_vertices(x, y, 1.0, 0)
            all_vertices_list.append(verts)
        all_vertices = np.vstack(all_vertices_list)
        
        s, params = _optimize_outer_hexagon(all_vertices)
        
        if s < best_side:
            best_side = s
            best_inner = np.array([[p[0], p[1], 0.0] for p in positions])
            best_outer_params = params
    
    if best_inner is None:
        # Fallback
        inner_hex_data = np.array(
            [
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
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    outer_hex_data = np.array([best_outer_params[1], best_outer_params[2], best_outer_params[0]])
    outer_hex_side_length = best_side
    
    return best_inner, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
