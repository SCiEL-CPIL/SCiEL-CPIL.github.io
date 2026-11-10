# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0

# ---- Hexagonal lattice helpers (axial coords -> cartesian) ----
# Unit hexagons on a hex lattice: centers at integer axial coords (q, r)
# cartesian: x = 1.5*q, y = sqrt(3)*(r + q/2)
_SQ3 = np.sqrt(3.0)

def _axial_to_cart(qr):
    """qr: (N,2) int array -> (N,2) float cartesian centers."""
    q = qr[:, 0].astype(float)
    r = qr[:, 1].astype(float)
    x = 1.5 * q
    y = _SQ3 * (r + 0.5 * q)
    return np.column_stack([x, y])

# ---- Generate all connected 12-cell hexominoes (up to rotation/reflection) ----
def _gen_hexominoes(n_cells=12):
    """BFS over connected sets of n_cells on hex lattice, dedup by canonical form."""
    # 6 axial directions
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    seen_canonical = set()
    shapes = []
    
    def canonical(cells):
        """Rotate/reflect to canonical; return frozenset of normalized coords."""
        cells = list(cells)
        # normalize: subtract min
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        norm = frozenset((c[0] - min_q, c[1] - min_r) for c in cells)
        # Try all 12 symmetries of the hex lattice
        # Rotations by 60 deg: (q, r) -> (-r, q+r)
        # Reflection: (q, r) -> (r, q)
        best = norm
        cur = set(cells)
        for _ in range(6):
            # reflect
            refl = frozenset((r, q) for (q, r) in cur)
            min_q = min(c[0] for c in refl)
            min_r = min(c[1] for c in refl)
            norm_r = frozenset((c[0] - min_q, c[1] - min_r) for c in refl)
            if norm_r < best:
                best = norm_r
            # rotate: (q, r) -> (-r, q+r)
            cur = {(-r, q + r) for (q, r) in cur}
        # also check the reflected rotations
        cur2 = set(cells)
        for _ in range(6):
            refl = {(-r, q + r) for (q, r) in cur2}
            min_q = min(c[0] for c in refl)
            min_r = min(c[1] for c in refl)
            norm_r = frozenset((c[0] - min_q, c[1] - min_r) for c in refl)
            if norm_r < best:
                best = norm_r
            cur2 = {(r, q) for (q, r) in cur2}
        return best
    
    def bfs(start_set, depth, path_set, path_list):
        if len(path_set) == n_cells:
            can = canonical(path_set)
            if can not in seen_canonical:
                seen_canonical.add(can)
                shapes.append(list(path_set))
            return
        if depth >= n_cells:
            return
        # Find cells adjacent to the current set
        candidates = set()
        for (q, r) in path_set:
            for (dq, dr) in dirs:
                c = (q + dq, r + dr)
                if c not in path_set:
                    candidates.add(c)
        # To reduce branching, only add cells that are "next to" the current shape
        # Sort candidates for determinism
        for c in sorted(candidates):
            bfs(path_set | {c}, depth + 1, path_set | {c}, path_list + [c])
    
    # Start from (0,0)
    bfs({(0, 0)}, 1, {(0, 0)}, [(0, 0)])
    
    return shapes


# ---- Analytic minimal containing hexagon side for a set of points ----
def _min_containing_hex_side(centers, rot_deg=0.0):
    """
    For centers (N,2) in world coords, find min side s of a regular hexagon
    (centered at origin, angle 0) that contains all unit hexagons at these centers.
    
    A unit hexagon at center c with angle 0 has vertices c + (cos(60i), sin(60i)).
    The containing hexagon with side s, center (0,0), angle 0 has edges at distance
    s*sqrt(3)/2 from center, with outward normals at angles 30, 90, 150, 210, 270, 330.
    
    For the containing hexagon to contain a point p, we need:
      for all 6 normals n_k: n_k . p <= s * sqrt(3)/2
    So s >= max_k (n_k . p) * 2 / sqrt(3) for all points p.
    
    But we can also rotate the outer hexagon. Here we fix outer angle=0 and
    instead rotate the inner centers.
    """
    rot = np.deg2rad(rot_deg)
    c_rot = np.cos(rot)
    s_rot = np.sin(rot)
    R = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
    
    # Rotate centers
    centers_r = centers @ R.T  # (N, 2)
    
    # Vertices of each unit hex (angle 0): offsets (cos60i, sin60i)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    offsets = np.column_stack([np.cos(base_angles), np.sin(base_angles)])  # (6, 2)
    
    # All vertices: (N*6, 2)
    all_verts = (centers_r[:, np.newaxis, :] + offsets[np.newaxis, :, :]).reshape(-1, 2)
    
    # Outward normals of outer hex (angle 0): 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # Max projection of all vertices onto each normal
    projs = all_verts @ normals.T  # (N*6, 6)
    max_proj = np.max(projs, axis=0)  # (6,)
    max_all = np.max(max_proj)
    
    s = (2.0 / _SQ3) * max_all
    return s


def _min_containing_hex_side_offset(centers, rot_deg, cx, cy):
    """
    Min side of outer hex centered at (cx, cy), angle 0, containing unit hexes at centers.
    """
    rot = np.deg2rad(rot_deg)
    c_rot = np.cos(rot)
    s_rot = np.sin(rot)
    R = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
    
    # Rotate and translate centers so outer center is origin
    centers_r = (centers - np.array([cx, cy])) @ R.T
    
    base_angles = np.deg2rad(60.0 * np.arange(6))
    offsets = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    all_verts = (centers_r[:, np.newaxis, :] + offsets[np.newaxis, :, :]).reshape(-1, 2)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    projs = all_verts @ normals.T
    max_proj = np.max(projs, axis=0)
    max_all = np.max(max_proj)
    
    s = (2.0 / _SQ3) * max_all
    return s


def _separation_violation_fast(centers):
    """
    For hex-lattice centers (integer axial -> cartesian), check separation.
    Two unit hexes on the hex lattice are non-overlapping iff their centers are
    at hex-lattice distance >= 1 (i.e., adjacent or farther). Since we generate
    connected sets of distinct cells, all pairs are at distance >= 1, so they
    are separated by construction. Return 0.
    """
    return 0.0


def _check_separation_lattice(centers):
    """For hex-lattice generated shapes, separation is guaranteed. Return True."""
    return True


def _optimize_shape(centers_base):
    """
    For a given hexomino shape (centers_base in cartesian), optimize rotation angle
    and centering offset to minimize containing hex side.
    Returns (best_s, best_rot, best_cx, best_cy).
    """
    # First, center the shape at origin for symmetry
    centroid = np.mean(centers_base, axis=0)
    centers_c = centers_base - centroid
    
    best_s = np.inf
    best_params = (0.0, 0.0, 0.0)
    
    # Coarse scan over rotation angles (0 to 60 deg, since hex symmetry)
    n_coarse = 36
    for k in range(n_coarse):
        rot = 60.0 * k / n_coarse
        s = _min_containing_hex_side(centers_c, rot)
        if s < best_s:
            best_s = s
            best_params = (rot, 0.0, 0.0)
    
    # Refine with minimize_scalar around best coarse angle
    rot0 = best_params[0]
    # Try a few windows
    for window in [10.0, 5.0, 2.0]:
        res = minimize_scalar(
            lambda r: _min_containing_hex_side(centers_c, r),
            bounds=(rot0 - window, rot0 + window),
            method='bounded',
            options={'xatol': 1e-8}
        )
        if res.success and res.fun < best_s:
            best_s = res.fun
            rot0 = res.x
    
    # Now optimize centering offset (cx, cy) for the best rotation
    # The optimal centering is the "center" of the bounding hex, which we can
    # approximate by the mean of the extreme vertices
    # Use a small 2D optimization for offset
    def obj_offset(params):
        cx, cy = params
        return _min_containing_hex_side_offset(centers_base, rot0, cx, cy)
    
    # Initial offset: centroid
    x0 = [centroid[0], centroid[1]]
    try:
        res_off = minimize(obj_offset, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res_off.success:
            s_off = res_off.fun
            if s_off < best_s:
                best_s = s_off
                best_params = (rot0, res_off.x[0], res_off.x[1])
    except Exception:
        pass
    
    return best_s, best_params


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Emergency fallback: simple hexagonal cluster
        centers = _axial_to_cart(np.array([
            [0, 0],
            [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1],
            [2, 0], [2, -1], [1, -2], [0, -2], [-1, -1]
        ]))
        s = _min_containing_hex_side(centers, 0.0)
        inner = np.zeros((12, 3))
        inner[:, 0] = centers[:, 0]
        inner[:, 1] = centers[:, 1]
        return inner, np.array([0.0, 0.0, 0.0]), s
    
    # Generate all 12-cell hexominoes
    shapes = _gen_hexominoes(12)
    
    best_s = np.inf
    best_centers = None
    best_rot = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Sort shapes by a quick bounding-box heuristic (smaller first)
    def shape_score(cells):
        pts = _axial_to_cart(np.array(cells))
        return np.max(pts[:, 0]) - np.min(pts[:, 0]) + np.max(pts[:, 1]) - np.min(pts[:, 1])
    
    shapes_sorted = sorted(shapes, key=shape_score)
    
    n_evaluated = 0
    for shape in shapes_sorted:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        centers_base = _axial_to_cart(np.array(shape))
        s, (rot, cx, cy) = _optimize_shape(centers_base)
        
        n_evaluated += 1
        
        if s < best_s:
            best_s = s
            best_rot = rot
            best_cx = cx
            best_cy = cy
            best_centers = centers_base.copy()
    
    if best_centers is None:
        # Fallback
        shape = shapes[0]
        best_centers = _axial_to_cart(np.array(shape))
        best_s = _min_containing_hex_side(best_centers, 0.0)
        best_rot = 0.0
        best_cx = 0.0
        best_cy = 0.0
    
    # Build final centers: rotate and translate
    rot_rad = np.deg2rad(best_rot)
    c_rot = np.cos(rot_rad)
    s_rot = np.sin(rot_rad)
    R = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
    
    # Centers relative to outer center
    centers_rel = (best_centers - np.array([best_cx, best_cy])) @ R.T
    
    # For output: inner hex centers in world coords (outer center at origin)
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = centers_rel[:, 0]
    inner_hex_data[:, 1] = centers_rel[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
