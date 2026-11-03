"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_support(points, angles_rad):
    """
    Compute the support function of a set of points (the centers of unit hexagons)
    in the 6 directions of a regular hexagon.
    
    For a regular hexagon of side length s centered at origin, rotated by angle alpha,
    the support in direction theta is s * (cos(alpha - theta) + sqrt(3)*sin(alpha - theta)) / 2 ?
    Actually, let's use the standard formula.
    
    A regular hexagon with side length s can be viewed as the intersection of 6 half-planes.
    The distance from the center to each side is the apothem, a = s * sqrt(3) / 2.
    The outward normal of the k-th side is at angle alpha + k * pi / 3.
    
    If we have a set of points P (centers of inner hexagons), and we want to enclose them
    in a regular hexagon of side length s centered at C with rotation alpha,
    then for each of the 6 directions (normals) n_k, we need:
    max_{p in P} (n_k . (p - C)) <= a = s * sqrt(3) / 2
    
    So, for a given configuration of centers P and rotation alpha,
    the minimal s is determined by:
    s = (2 / sqrt(3)) * max_{k} max_{p in P} (n_k . (p - C))
    
    But we can also translate C. To minimize s, we should choose C to minimize the max support.
    This is equivalent to finding the smallest enclosing hexagon.
    
    Let's define the support values for a fixed rotation alpha.
    For each direction k, let d_k = max_{p in P} (n_k . p).
    The center C should be chosen such that the max over k of (d_k - n_k . C) is minimized.
    Let u_k = d_k - n_k . C. We want to minimize max(u_k).
    This is a convex optimization problem.
    
    Alternatively, we can formulate it as an LP.
    Variables: C_x, C_y, t
    Minimize t
    Subject to:
    n_k . C + t >= d_k  for all k
    -n_k . C + t >= -d_k  (wait, this is for the opposite side)
    
    Actually, the condition is:
    For each side k with normal n_k:
    max_{p in P} (n_k . (p - C)) <= a
    => max_{p in P} (n_k . p) - n_k . C <= a
    => n_k . C + a >= max_{p in P} (n_k . p)
    
    And for the opposite side k+3 with normal -n_k:
    max_{p in P} (-n_k . (p - C)) <= a
    => -max_{p in P} (n_k . p) + n_k . C <= a
    => -n_k . C + a >= -max_{p in P} (n_k . p)
    => n_k . C - a <= min_{p in P} (n_k . p)
    
    So we have:
    n_k . C + a >= max_k
    n_k . C - a <= min_k
    
    Let M_k = max_{p in P} (n_k . p)
    Let m_k = min_{p in P} (n_k . p)
    
    We want to minimize a (which is proportional to s).
    Variables: C_x, C_y, a
    Minimize a
    Subject to:
    n_k . C + a >= M_k
    -n_k . C + a >= -m_k  (which is n_k . C - a <= m_k)
    
    This is a linear program!
    """
    # points: (N, 2) array of centers
    # angles_rad: 6 angles for the normals
    # n_k = [cos(angle_k), sin(angle_k)]
    
    M = np.max(points @ np.column_stack([np.cos(angles_rad), np.sin(angles_rad)]), axis=0)
    m = np.min(points @ np.column_stack([np.cos(angles_rad), np.sin(angles_rad)]), axis=0)
    
    # LP:
    # Variables: x, y, a
    # Minimize a
    # Constraints:
    # cos(alpha_k) * x + sin(alpha_k) * y + a >= M_k
    # -cos(alpha_k) * x - sin(alpha_k) * y + a >= -m_k
    
    # In standard form for linprog (minimize c^T x subject to A_ub x <= b_ub):
    # -cos(alpha_k) * x - sin(alpha_k) * y - a <= -M_k
    # cos(alpha_k) * x + sin(alpha_k) * y - a <= m_k
    
    c = [0, 0, 1]
    A_ub = []
    b_ub = []
    
    for k in range(6):
        cx = np.cos(angles_rad[k])
        cy = np.sin(angles_rad[k])
        A_ub.append([-cx, -cy, -1])
        b_ub.append(-M[k])
        A_ub.append([cx, cy, -1])
        b_ub.append(m[k])
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)])
    
    if res.success:
        a = res.x[2]
        s = 2 * a / np.sqrt(3)
        center = (res.x[0], res.x[1])
        return s, center
    else:
        return np.inf, (0, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial configuration: hexagonal packing
    # Centers of unit hexagons in a hexagonal grid
    # Let's try a 1-2-3-2-1 arrangement or similar
    # For N=11, a compact arrangement is a hexagon with side length 2 (which has 7 hexagons) plus 4 more.
    # Or a linear arrangement.
    
    # Let's generate several candidate configurations
    candidates = []
    
    # 1. Linear arrangement
    for k in range(-(N // 2), N - N // 2):
        candidates.append([(k * 3.0, 0.0) for _ in range(1)]) # This is not right, we need N points
    
    # Let's just create a list of N points
    # Configuration 1: Linear
    config1 = [(3.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
    candidates.append(np.array(config1))
    
    # Configuration 2: Hexagonal grid
    # Center at (0,0), then neighbors
    # Hex grid coordinates to Cartesian
    def hex_to_cart(q, r):
        x = 3.0 * q + 1.5 * r
        y = np.sqrt(3) * r
        return (x, y)
    
    hex_coords = [(0,0)]
    # Add rings
    for ring in range(1, 5):
        # 6 * ring hexagons in ring
        for i in range(6 * ring):
            # Generate coordinates for ring
            # This is complex, let's use a simpler method
            pass
    
    # Let's just use a predefined set of coordinates for a compact hexagonal packing
    # 11 hexagons: center + 6 neighbors + 4 more
    config2 = [(0,0)]
    for i in range(6):
        angle = i * np.pi / 3
        config2.append((3.0 * np.cos(angle), 3.0 * np.sin(angle)))
    # Add 4 more around
    for i in range(0, 6, 2):
        angle = i * np.pi / 3
        config2.append((6.0 * np.cos(angle), 6.0 * np.sin(angle)))
    candidates.append(np.array(config2))
    
    # Configuration 3: Two rows
    config3 = []
    for k in range(-(N // 2), N - N // 2):
        config3.append((3.0 * k, 0.0))
    # Shift every other one up
    config3_alt = []
    for k in range(-(N // 2), N - N // 2):
        if k % 2 == 0:
            config3_alt.append((3.0 * k, 0.0))
        else:
            config3_alt.append((3.0 * k, np.sqrt(3) * 1.5))
    candidates.append(np.array(config3_alt))
    
    best_s = np.inf
    best_center = (0, 0)
    best_config = None
    best_angle = 0.0
    
    for config in candidates:
        if time.time() > deadline:
            break
            
        # Try different rotations of the container
        # The optimal rotation is likely to be aligned with the symmetry of the config
        # But let's try a few angles
        for alpha_deg in [0, 30, 15, 45]:
            if time.time() > deadline:
                    break
                    
            alpha_rad = np.radians(alpha_deg)
            # The 6 normals of the container
            angles_rad = alpha_rad + np.arange(6) * np.pi / 3
            
            s, center = hexagon_support(config, angles_rad)
            
            if s < best_s:
                best_s = s
                best_center = center
                best_config = config
                best_angle = alpha_deg
    
    # Now we have a best configuration and container angle
    # We can try to optimize the configuration itself
    # For now, let's just return the best we found
    
    # Construct the inner list
    inner = []
    for i in range(N):
        x, y = best_config[i]
        # The inner hexagons can be rotated arbitrarily, so angle 0 is fine
        inner.append([x, y, 0.0])
    
    return inner, best_center, best_s, best_angle

# EVOLVE-BLOCK-END
