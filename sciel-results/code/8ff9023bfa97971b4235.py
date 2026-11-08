"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def hex_vertices(cx, cy, angle_deg=0.0):
        """Return 6 vertices of unit regular hexagon centered at (cx,cy) with given rotation."""
        rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = rad + i * math.pi / 3
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return verts

    def min_hex_side(centered_verts_list):
        """
        Find minimum side length s of a regular hexagon (any rotation, centered at origin)
        containing all vertices in centered_verts_list.
        centered_verts_list: list of lists of (x,y) tuples, all relative to container center.
        """
        best_s = float('inf')
        all_pts = [pt for verts in centered_verts_list for pt in verts]

        # Try container rotations from 0 to 60 degrees
        for phi_deg in range(0, 61, 1):
            phi = math.radians(phi_deg)
            thetas = [phi + k * math.pi / 3 for k in range(6)]
            cos_thetas = [math.cos(t) for t in thetas]
            sin_thetas = [math.sin(t) for t in thetas]
            s_needed = 0.0
            for (x, y) in all_pts:
                for ct, st in zip(cos_thetas, sin_thetas):
                    val = x * ct + y * st
                    if val > s_needed:
                        s_needed = val
            if s_needed < best_s:
                best_s = s_needed

        return best_s

    def score_arrangement(hexes):
        """
        hexes: list of (x, y, angle_deg)
        Returns (s, center) where s is min outer hex side, center is (cx, cy).
        """
        all_verts = []
        for (x, y, ang) in hexes:
            verts = hex_vertices(x, y, ang)
            all_verts.append(verts)

        # Find centroid of all vertices
        total_x = sum(v[0] for verts in all_verts for v in verts)
        total_y = sum(v[1] for verts in all_verts for v in verts)
        n_verts = len(all_verts) * 6
        cx = total_x / n_verts
        cy = total_y / n_verts

        # Translate vertices to be centered at origin
        centered_verts = []
        for verts in all_verts:
            centered_verts.append([(v[0] - cx, v[1] - cy) for v in verts])

        s = min_hex_side(centered_verts)
        return s, (cx, cy)

    def make_hexes(pattern):
        """pattern: list of (row_y, [x_positions])"""
        hexes = []
        for row_y, xs in pattern:
            for x in xs:
                hexes.append((x, row_y, 0.0))
        return hexes

    sqrt3 = math.sqrt(3)
    candidates = []

    # 1. Single row of 11
    xs = [2.0 * k for k in range(-5, 6)]
    candidates.append([(0.0, xs)])

    # 2. Two rows: 6 + 5 staggered (6 on bottom, 5 on top)
    row6 = [2.0 * k for k in range(-5, 1)]   # -5, -3, -1, 1, 3, 5
    row5 = [2.0 * k for k in range(-4, 5)]    # -4, -2, 0, 2, 4
    candidates.append([(0.0, row6), (sqrt3, row5)])

    # 3. Two rows: 5 + 6 staggered (5 on bottom, 6 on top)
    candidates.append([(0.0, row5), (sqrt3, row6)])

    # 4. Three rows: 4 + 3 + 4
    r4a = [-3.0, -1.0, 1.0, 3.0]
    r3a = [-2.0, 0.0, 2.0]
    r4b = [-3.0, -1.0, 1.0, 3.0]
    candidates.append([(0.0, r4a), (sqrt3, r3a), (2 * sqrt3, r4b)])

    # 5. Three rows: 3 + 4 + 4 (asymmetric)
    r3b = [-2.0, 0.0, 2.0]
    r4c = [-3.0, -1.0, 1.0, 3.0]
    r4d = [-4.0, -2.0, 0.0, 2.0]
    candidates.append([(0.0, r3b), (sqrt3, r4c), (2 * sqrt3, r4d)])

    # 6. Two rows: 4 + 7
    r4e = [-3.0, -1.0, 1.0, 3.0]
    r7 = [2.0 * k for k in range(-3, 4)]  # -6,-4,-2,0,2,4,6
    candidates.append([(0.0, r4e), (sqrt3, r7)])

    # 7. Two rows: 7 + 4
    candidates.append([(0.0, r7), (sqrt3, r4e)])

    # 8. Three rows: 5 + 1 + 5
    r5a = [2.0 * k for k in range(-2, 3)]  # -4,-2,0,2,4
    r1 = [0.0]
    r5b = [2.0 * k for k in range(-2, 3)]
    candidates.append([(0.0, r5a), (sqrt3, r1), (2 * sqrt3, r5b)])

    # 9. Three rows: 2 + 7 + 2
    r2 = [-1.0, 1.0]
    r7b = [2.0 * k for k in range(-3, 4)]
    candidates.append([(0.0, r2), (sqrt3, r7b), (2 * sqrt3, r2)])

    # 10. Four rows: 3 + 2 + 2 + 4 or similar
    r3c = [-2.0, 0.0, 2.0]
    r2a = [-1.0, 1.0]
    r2b = [-1.0, 1.0]
    r4f = [-3.0, -1.0, 1.0, 3.0]
    candidates.append([(0.0, r3c), (sqrt3, r2a), (2 * sqrt3, r2b), (3 * sqrt3, r4f)])

    best_s = float('inf')
    best_center = (0.0, 0.0)
    best_inner = None

    for i, pattern in enumerate(candidates):
        if time.time() > deadline:
            break
        hexes = make_hexes(pattern)
        s, center = score_arrangement(hexes)
        if s < best_s:
            best_s = s
            best_center = center
            best_inner = hexes

    # If no candidates worked (shouldn't happen), fall back to single row
    if best_inner is None:
        xs = [2.0 * k for k in range(-5, 6)]
        best_inner = [(x, 0.0, 0.0) for x in xs]
        best_s, best_center = score_arrangement(best_inner)

    # Convert inner to required format: list of [x, y, angle_degrees]
    inner = [[x, y, ang] for (x, y, ang) in best_inner]

    return inner, best_center, best_s, 0.0
# EVOLVE-BLOCK-END
