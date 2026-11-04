import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0,60,...300)
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])
# 12 separation directions (6 normals + 6 rotated by 30 deg)
# Rotation by 30 degrees matrix
R30 = np.array([
    [np.cos(np.pi/6), -np.sin(np.pi/6)],
    [np.sin(np.pi/6), np.cos(np.pi/6)]
])
N12 = np.vstack([NORM, NORM @ R30.T])

# unit hexagon vertices
UNIT = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def outer_side(C, A):
    """Smallest outer hexagon (centered at origin, rot 0) side length enclosing all hexagons."""
    m = 0.0
    for k in range(6):
        n = NORM[k]
        # Project centers
        proj = C @ n
        # Add the maximum projection of the hexagon itself onto this normal
        # For a unit hexagon rotated by angle A, the max projection onto a fixed normal n is:
        # max( |n . v| ) for v in vertices of rotated hex.
        # Since the hexagon is convex and symmetric, the max projection is simply the support function.
        # For a regular hexagon of side 1, the support function h(theta) = cos(theta - alpha) / cos(pi/6) ?
        # Actually, the distance from center to edge is S3/2. The distance to vertex is 1.
        # The projection of the hexagon onto a direction n is 2 * max( |n . v| ).
        # Let's calculate it properly.
        # Vertices of hex i: C[i] + R(A[i]) @ UNIT
        # We can just compute the max projection of the vertices directly.
        # But doing this for all 11 hexes * 6 vertices * 6 normals in the loop is slow.
        # Optimization: The max projection of a unit hexagon onto any direction is 1 (if aligned with vertex) or S3/2 * 2? No.
        # The width of a unit hexagon is 2*sqrt(3) (distance between parallel edges) and 2 (distance between opposite vertices).
        # The projection onto a normal n depends on the orientation A[i].
        
        # For simplicity and robustness in the objective function (which is called many times),
        # we can approximate or compute exactly.
        # Let's compute exactly for accuracy.
        
        # To speed up, we can precompute the rotated vertices for each hexagon once per call?
        # No, C and A change.
        
        # Let's use the property that the max projection of a unit hexagon on a unit vector u is:
        # h(u) = max( |u . v| ) where v are the vertices.
        # This is equivalent to max( |u . R(A) . v0| ) = max( |R(-A) . u . v0| )
        # So it's the support function of the unit hexagon evaluated at direction -A.
        
        # For a regular hexagon, the support function is piecewise cosine.
        # h(theta) = cos(theta mod pi/6 - pi/6) / cos(pi/6) ?
        # At theta=0 (vertex), h=1. cos(0)/cos(30) = 1/0.866 = 1.15? No.
        # Let's just calculate the max projection of the 6 vertices for each hex.
        # It's 11 * 6 * 6 = 396 dot products. Negligible.
        
        max_proj = 0.0
        for i in range(N):
            # Rotate unit vertices by A[i]
            angle = A[i] * np.pi / 180.0
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            verts = UNIT @ R.T
            # Project onto n
            ps = verts @ n
            # Max projection of this hexagon onto n is max(ps)
            # The extent in direction n is max(ps) - min(ps) = 2*max(|ps|) due to symmetry?
            # No, the hexagon is centered at 0. So max projection is max(ps).
            # The total projection of the hexagon centered at C[i] is C[i].n + max(ps).
            val = C[i] @ n + np.max(ps)
            if val > max_proj:
                max_proj = val
        m = max(m, max_proj)
    
    return 2.0 * m / S3


def separation_violation(C, A):
    """For each pair, the max separation along the 12 directions minus 2. Negative => overlap."""
    viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            # For exact non-overlap of two rotated hexagons, we need to check SAT on the union of normals.
            # However, for the objective function, we can use a slightly weaker but differentiable-ish penalty.
            # Using the 12 fixed directions is a good approximation for the penalty to guide the optimizer.
            sep = np.abs(d @ N12.T).max()
            if sep < 2.0:
                viol += (2.0 - sep) ** 2
    return viol


def objective_flat(flat):
    C = flat[0::3].reshape(N, 2)
    A = flat[1::3]
    side = outer_side(C, A)
    viol = separation_violation(C, A)
    return side + 100.0 * viol


def exact_nonoverlap(C, A, eps=1e-9):
    """Exact check: two convex polygons non-overlapping iff a separating axis exists.
    Use SAT on the 12 edge normals (union of both orientations)."""
    for i in range(N):
        for j in range(i + 1, N):
            rel = (A[i] - A[j]) * np.pi / 180.0
            c, s = np.cos(rel), np.sin(rel)
            R = np.array([[c, -s], [s, c]])
            n_j = NORM @ R
            axes = np.vstack([NORM, n_j])
            d = C[i] - C[j]
            sep = np.abs(d @ axes.T).max()
            if sep < 2.0 - eps:
                return False
    return True


def inside_outer(C, A, S, eps=1e-9):
    """All hexagons inside outer hexagon of side S centered at origin rot 0."""
    apothem = S * S3 / 2.0
    for k in range(6):
        n = NORM[k]
        # Calculate max projection of all hexagons onto this normal
        max_proj = -np.inf
        for i in range(N):
            angle = A[i] * np.pi / 180.0
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            verts = UNIT @ R.T
            ps = verts @ n
            val = C[i] @ n + np.max(ps)
            if val > max_proj:
                max_proj = val
        if max_proj > apothem + eps:
            return False
    return True


def valid(C, A, S):
    return exact_nonoverlap(C, A) and inside_outer(C, A, S)


def init_flower_plus():
    """
    Initialize with a 7-hex flower (center + 6) and 4 more hexes.
    Place the 4 extra hexes at distance ~2*sqrt(3) from center, spaced out.
    """
    C = [[0.0, 0.0]]
    # 6 surrounding
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    
    # 4 more. Let's place them at 0, 90, 180, 270 degrees but further out?
    # Or at the "gaps"?
    # The 7-hex flower has "corners" at 0, 60, 120, 180, 240, 300.
    # If we place hexes at 0, 120, 240, we add 3. We need 4.
    # Let's place them at 0, 90, 180, 270 with a larger radius to avoid overlap.
    # Distance from center to center of touching hexes is 2*sqrt(3) ~ 3.46.
    # The outer ring of the flower is at sqrt(3) ~ 1.73.
    # So we can place new hexes at distance 3.0 from center.
    # But they must not overlap with the ring.
    # Distance from a ring hex (at angle 0, dist 1.73) to a new hex (at angle 0, dist 3.0) is 1.27 < 2. Overlap!
    # So we need to place them in the gaps or further away.
    
    # Let's just place them randomly in a larger ring and let the optimizer fix it.
    # Or, use a known good configuration.
    # For N=11, a common packing is:
    # Center.
    # Ring 1 (6).
    # Ring 2 (4) placed at angles 0, 120, 240 and one more?
    
    # Let's try a "linear" cluster first as a robust fallback, then flower.
    # But flower is better.
    
    # Let's place the 4 extra hexes at:
    # (2*S3, 0), (-2*S3, 0), (S3, 3), (-S3, 3) ?
    # Let's just use a wider ring for the extra 4.
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6 # 30, 120, 210, 300
        # Place them at distance 2.5 * S3
        dist = 2.5 * S3
        C.append([dist * np.cos(a), dist * np.sin(a)])
        
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def optimize(C, A, deadline, maxiter=50000):
    flat0 = np.empty(3 * N)
    flat0[0::3] = C[:, 0]
    flat0[1::3] = A
    flat0[2::3] = C[:, 1]
    
    start_time = time.time()
    try:
        # Use Nelder-Mead
        res = minimize(objective_flat, flat0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-8,
                                "adaptive": True})
        
        # Check time
        if time.time() > deadline:
            # Return the best found so far? minimize doesn't easily give intermediate bests.
            # We'll just return the result.
            pass
    except Exception:
        return C, A
        
    f = res.x
    Cn = np.column_stack([f[0::3], f[2::3]])
    An = f[1::3]
    return Cn, An


def construct():
    start = time.time()
    deadline = start + 140.0

    # Try the flower + 4 initialization
    C, A = init_flower_plus()
    Cn, An = optimize(C, A, deadline, maxiter=50000)
    Sn = outer_side(Cn, An)
    
    if not valid(Cn, An, Sn):
        # Fallback to a simple linear packing if optimization fails
        Cn = np.zeros((N, 2))
        for i in range(N):
            Cn[i] = [i * 2.0 * S3, 0.0]
        An = np.zeros(N)
        Sn = outer_side(Cn, An)

    # One more refinement pass if time permits
    if time.time() < deadline - 10:
        Cn2, An2 = optimize(Cn, An, deadline, maxiter=20000)
        Sn2 = outer_side(Cn2, An2)
        if valid(Cn2, An2, Sn2) and Sn2 < Sn:
            Cn, An, Sn = Cn2, An2, Sn2

    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), float(An[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(Sn), 0.0
