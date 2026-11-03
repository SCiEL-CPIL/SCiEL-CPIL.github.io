"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses spectral graph embedding: positions derived from Laplacian eigenvectors,
with edge weights and container orientation optimized via Nelder-Mead.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill
_EPS = 1e-9


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def hex_vertices_array(cx, cy, s, angle_deg):
    """Vectorized hex vertices as numpy array (6,2)."""
    a = math.radians(angle_deg)
    thetas = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(thetas), cy + s * np.sin(thetas)])


def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - _EPS or maxB < minA - _EPS:
                return False
    return True


def sat_gap(A, B):
    """Return minimum separation gap between two convex polygons.
    Positive gap means separated; <= 0 means overlapping/touching."""
    min_gap = float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-12:
                continue
            nx, ny = nx / norm, ny / norm
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            gap = min(minA - maxB, minB - maxA)
            if gap > 0:
                return gap  # found separating axis, they are disjoint
            min_gap = min(min_gap, gap)
    return min_gap


def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon (CCW or CW)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj


def smooth_max(values, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    m = np.max(arr)
    if m < -50.0 / beta:
        return m
    return m + math.log(np.sum(np.exp(beta * (arr - m)))) / beta


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def build_spectral_packing(edge_weights, rot_in_deg, rot_out_deg, spread=1.0):
    """
    Build positions using spectral graph embedding.
    
    edge_weights: array of shape (N, N) symmetric, non-negative.
      w[i,j] controls how "close" hexagons i and j should be.
    rot_in_deg: rotation angle for all inner hexagons.
    rot_out_deg: rotation angle for outer hexagon.
    spread: scaling factor for the spectral embedding.
    
    Returns: (inner_list, outer_center, outer_side, outer_angle)
    """
    W = np.asarray(edge_weights, dtype=float)
    W = 0.5 * (W + W.T)  # ensure symmetric
    W = np.maximum(W, 0.0)
    
    # Graph Laplacian: L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # The first eigenvector (smallest eigenvalue) is the constant vector (null space).
    # Use the next two eigenvectors for 2D embedding (Fiedler vectors).
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    
    # Take the 2nd and 3rd eigenvectors (indices 1 and 2) for 2D coordinates
    # These give the "most spread" 2D embedding that respects the graph structure
    x_coords = eigenvectors[:, 1]
    y_coords = eigenvectors[:, 2]
    
    # Center and scale
    x_coords = x_coords - x_coords.mean()
    y_coords = y_coords - y_coords.mean()
    
    # Scale so that the typical spacing matches what we need
    # For unit hexagons, minimum center distance ~ 2 (for aligned hexagons)
    # The eigenvectors are normalized, so we scale by spread
    scale_factor = spread
    x_coords *= scale_factor
    y_coords *= scale_factor
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append((float(x_coords[i]), float(y_coords[i]), rot_in_deg))
    
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out_deg)
    
    return inner, outer_center, outer_side, rot_out_deg


def build_ring12(scale, rot_in_deg, rot_out_deg):
    """12 hexagons in a ring around a center point."""
    positions = []
    for k in range(12):
        theta = math.radians(k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        positions.append((x, y))
    inner = [(x, y, rot_in_deg) for (x, y) in positions]
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out_deg)
    return inner, outer_center, outer_side, rot_out_deg


def build_hex6_plus6(scale, rot_in_deg, rot_out_deg):
    """6 in a hexagonal cluster + 6 in a ring around them."""
    positions = []
    # Inner hex cluster: center + 5 neighbors (or 6 in a ring)
    a = 2.0 * scale
    positions.append((0.0, 0.0))
    for k in range(5):
        theta = math.radians(k * 72.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    # Outer ring of 6
    b = 4.0 * scale
    for k in range(6):
        theta = math.radians(k * 60.0 + 30.0)
        positions.append((b * math.cos(theta), b * math.sin(theta)))
    inner = [(x, y, rot_in_deg) for (x, y) in positions]
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out_deg)
    return inner, outer_center, outer_side, rot_out_deg


def build_rect3x4(scale, rot_in_deg, rot_out_deg):
    """3x4 rectangular grid of hexagons."""
    positions = []
    dx = 2.0 * scale
    dy = 2.0 * scale * math.sqrt(3) / 2.0
    for i in range(3):
        for j in range(4):
            x = (j - 1.5) * dx
            y = (i - 1.0) * dy
            positions.append((x, y))
    inner = [(x, y, rot_in_deg) for (x, y) in positions]
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out_deg)
    return inner, outer_center, outer_side, rot_out_deg


def objective(params):
    """Objective for spectral packing optimization.
    params: [edge_weights flattened (N*N), rot_in_deg, rot_out_deg, spread]
    Returns outer side length + penalty for overlaps.
    """
    if time.time() > DEADLINE:
        return 1e6
    
    n_params = len(params)
    w_size = N * N
    edge_weights = np.array(params[:w_size]).reshape(N, N)
    rot_in_deg = params[w_size]
    rot_out_deg = params[w_size + 1]
    spread = params[w_size + 2]
    
    # Constrain
    if spread < 1.0 or spread > 10.0:
        return 1e6
    
    inner, outer_center, outer_side, outer_angle = build_spectral_packing(
        edge_weights, rot_in_deg, rot_out_deg, spread)
    
    # Check pairwise overlap and compute penalty
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0:
                overlap_penalty += (-gap) ** 2
    
    # Check containment
    outer_poly = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    contain_penalty = 0.0
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer_poly):
                # Estimate how far outside
                for (px2, py2) in outer_poly:
                    d = math.sqrt((px - px2) ** 2 + (py - py2) ** 2)
                    contain_penalty += d ** 2
                break
    
    total = outer_side + 100.0 * overlap_penalty + 50.0 * contain_penalty
    return total


def optimize_topology():
    """Search over spectral packing parameters and structured topologies."""
    best_side = 1e6
    best_result = None
    
    # Phase 1: Try structured topologies with parameter sweep
    for scale in np.linspace(0.9, 2.5, 8):
        for rot_in in [0.0, 30.0, 15.0, 45.0]:
            for rot_out in [0.0, 30.0, 15.0, 45.0]:
                if time.time() > DEADLINE:
                    return best_side, best_result
                
                for builder in [build_ring12, build_hex6_plus6, build_rect3x4]:
                    try:
                        inner, oc, side, oa = builder(scale, rot_in, rot_out)
                        if check_validity(inner, oc, side, oa):
                            if side < best_side:
                                best_side = side
                                best_result = (inner, oc, side, oa)
                    except Exception:
                        pass
    
    # Phase 2: Spectral embedding optimization
    if time.time() < DEADLINE - 30:
        # Initialize edge weights: use a hexagonal adjacency pattern
        # Hexagonal lattice adjacency for 12 nodes
        # Start with a ring graph + some cross connections
        W_init = np.zeros((N, N))
        # Ring connections
        for i in range(N):
            W_init[i, (i + 1) % N] = 1.0
            W_init[(i + 1) % N, i] = 1.0
        # Add some diagonal connections for compactness
        for i in range(N):
            W_init[i, (i + 2) % N] = 0.5
            W_init[(i + 2) % N, i] = 0.5
            W_init[i, (i + 4) % N] = 0.3
            W_init[(i + 4) % N, i] = 0.3
        
        w_flat = W_init.flatten()
        initial_params = list(w_flat) + [0.0, 0.0, 3.0]  # rot_in, rot_out, spread
        
        try:
            result = minimize(objective, initial_params, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5,
                                      'maxfev': 2000})
            if result.fun < best_side and result.fun < 1e5:
                # Extract and validate
                w_size = N * N
                W_opt = np.array(result.x[:w_size]).reshape(N, N)
                rot_in_opt = result.x[w_size]
                rot_out_opt = result.x[w_size + 1]
                spread_opt = result.x[w_size + 2]
                
                inner, oc, side, oa = build_spectral_packing(
                    W_opt, rot_in_opt, rot_out_opt, spread_opt)
                
                if check_validity(inner, oc, side, oa) and side < best_side:
                    best_side = side
                    best_result = (inner, oc, side, oa)
        except Exception:
            pass
    
    # Phase 3: If spectral didn't work well, try more structured searches
    if time.time() < DEADLINE - 10 and best_side > 10.0:
        # Try denser sweep of structured topologies
        for scale in np.linspace(0.85, 2.0, 12):
            for rot_in in [0.0, 30.0]:
                for rot_out in [0.0, 30.0]:
                    if time.time() > DEADLINE:
                        return best_side, best_result
                    for builder in [build_ring12, build_hex6_plus6, build_rect3x4]:
                        try:
                            inner, oc, side, oa = builder(scale, rot_in, rot_out)
                            if check_validity(inner, oc, side, oa):
                                if side < best_side:
                                    best_side = side
                                    best_result = (inner, oc, side, oa)
                        except Exception:
                            pass
    
    return best_side, best_result


def construct():
    t0 = time.time()
    
    # Try spectral + structured packing optimization
    try:
        best_side, best_result = optimize_topology()
        if best_result is not None and best_side < 1e5:
            return best_result
    except Exception:
        pass
    
    # Fallback: simple linear arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
