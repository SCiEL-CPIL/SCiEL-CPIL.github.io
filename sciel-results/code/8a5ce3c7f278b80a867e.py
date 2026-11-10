<<<<<<< SEARCH
def _outer_side(inner, tol=1e-9):
    """Smallest outer side S (centered at origin) so every inner vertex is inside
    the outer hexagon of side S (flat-top, angle 0)."""
    X = inner[:, 0]
    Y = inner[:, 1]
    A = np.deg2rad(inner[:, 2])
    V = np.column_stack((X + np.cos(A), Y + np.sin(A)))
    # outer flat-top edges (outward normals)
    N = np.array([[math.cos(t), math.sin(t)] for t in
                  (90, 150, 210, 270, 330, 30)])
    D = V @ N.T
    M = D.max(axis=0)
    S = np.maximum(M / N[:, 1], M / N[:, 0])
    S = np.maximum(S, 0.0)
    S = np.max(S)
    if not np.isfinite(S):
        return np.inf
    return S + tol
=======
def _outer_side(inner, tol=1e-9):
    """Smallest outer side S (centered at origin) so every inner vertex is inside
    the outer hexagon of side S (flat-top, angle 0)."""
    X = inner[:, 0]
    Y = inner[:, 1]
    A = np.deg2rad(inner[:, 2])
    V = np.column_stack((X + np.cos(A), Y + np.sin(A)))
    # Outer hexagon with angle 0 (pointy-top? No, prompt says angle a is for vertices.
    # If a=0, vertex at 0 deg. Normals at 30, 90, ...
    # Apothem = S * sqrt(3) / 2.
    # Distance from center to edge = S * sqrt(3) / 2.
    # So S = (2 / sqrt(3)) * max_projection.
    N = np.array([[math.cos(np.deg2rad(t)), math.sin(np.deg2rad(t))] for t in
                  (30, 90, 150, 210, 270, 330)])
    D = V @ N.T
    M = D.max(axis=0)
    S = np.max(M) * (2.0 / math.sqrt(3.0))
    if not np.isfinite(S):
        return np.inf
    return S + tol
>>>>>>> REPLACE

