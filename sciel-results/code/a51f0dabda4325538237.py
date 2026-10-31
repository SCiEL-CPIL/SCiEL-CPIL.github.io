import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate positions on a hexagonal grid
    # Axial coordinates (q, r) to Cartesian
    # x = size * (sqrt(3) * q + sqrt(3)/2 * r)
    # y = size * (3/2 * r)
    # For unit hexagons, center distance is sqrt(3).
    # So size = 1.
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate a large enough grid of axial coordinates
    # For N=11, a radius of 3 is sufficient.
    # To be safe for slightly larger N, we can estimate D.
    # Hexagonal numbers: 1, 7, 19, 37...
    # D=1 -> 7, D=2 -> 19.
    # So for N=11, D=2 is enough (19 points).
    # Let's use D=3 to be safe.
    D = 3
    candidates = []
    for q in range(-D, D + 1):
        for r in range(-D, D + 1):
            # Axial distance
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist <= D:
                x, y = axial_to_cartesian(q, r)
                candidates.append((x, y, dist))
    
    # Sort by distance from origin to get a compact cluster
    candidates.sort(key=lambda p: p[2])
    
    # Take the first N
    if N > len(candidates):
        # Fallback if N is larger than generated grid
        # Extend D
        D = int(np.sqrt(N)) + 2
        candidates = []
        for q in range(-D, D + 1):
            for r in range(-D, D + 1):
                dist = (abs(q) + abs(r) + abs(q + r)) / 2
                if dist <= D:
                    x, y = axial_to_cartesian(q, r)
                    candidates.append((x, y, dist))
        candidates.sort(key=lambda p: p[2])

    selected = candidates[:N]
    
    # Extract x, y
    inner_x = [p[0] for p in selected]
    inner_y = [p[1] for p in selected]
    
    # All angles are 0
    inner = [[x, y, 0.0] for x, y in zip(inner_x, inner_y)]
    
    # Determine optimal outer hexagon size S and center
    # Normals for a hexagon with angle 0 (vertices at 30, 90, 150, ...)
    # Side normals are at 0, 60, 120...
    # Vertex directions are 30, 90, 150...
    # We use vertex directions for the support function calculation as derived.
    
    n1 = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
    n2 = np.array([np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))])
    n3 = np.array([np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))])
    
    S = 0.0
    u_vals = []
    for n in [n1, n2, n3]:
        projs = [x * n[0] + y * n[1] for x, y in zip(inner_x, inner_y)]
        max_p = max(projs)
        min_p = min(projs)
        width = max_p - min_p
        S_k = 1 + width / 2.0
        if S_k > S:
            S = S_k
        u_vals.append((max_p + min_p) / 2.0)
    
    # Solve for center (cx, cy)
    # u1 = cx cos30 + cy sin30
    # u2 = cy
    cy = u_vals[1]
    cx = (u_vals[0] - cy * np.sin(np.deg2rad(30))) / np.cos(np.deg2rad(30))
    
    return inner, (cx, cy), S, 0.0
# EVOLVE-BLOCK-END
