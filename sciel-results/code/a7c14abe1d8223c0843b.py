import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = math.sqrt(3)

    # Precompute unit hexagon vertices (pointy-top: vertices at 30, 90, 150, 210, 270, 330 deg)
    # For flat-top: vertices at 0, 60, 120, 180, 240, 300
    # We'll use flat-top convention
    def hex_verts(side, angle_deg):
        a = math.radians(angle_deg)
        return np.array([
            [side * math.cos(a + math.radians(k * 60)), side * math.sin(a + math.radians(k * 60))]
            for k in range(6)
        ])

    def translate_verts(verts, cx, cy):
        return verts + np.array([cx, cy])

    def rotate_verts(verts, angle_deg):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        return verts @ R.T

    def get_full_verts(cx, cy, side, angle_deg):
        v = hex_verts(side, 0)
        v = rotate_verts(v, angle_deg)
        v = v * side  # already scaled
        # Actually hex_verts already includes side, so just rotate and translate
        v = hex_verts(1.0, 0)
        v = rotate_verts(v, angle_deg)
        v = v * side
        v = v + np.array([cx, cy])
        return v

    def min_outer_side(positions, angles, outer_angle_deg):
        """Compute minimum side length of outer regular hexagon (centered at origin) containing all inner hexagons."""
        oa = math.radians(outer_angle_deg)
        # Normals to the 6 sides of the outer hexagon
        # For a regular hexagon with side s, the distance from center to each side is s * sqrt(3)/2
        # The 6 outward normals are at angles oa + 30 + k*60 for k=0..5
        normals = np.array([[math.cos(oa + math.radians(30 + k * 60)), math.sin(oa + math.radians(30 + k * 60))] for k in range(6)])
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_full_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            mp = np.max(projs)
            if mp > max_proj:
                max_proj = mp
        
        # The apothem (center-to-side distance) of outer hexagon with side s is s*sqrt(3)/2
        # So max_proj = s * sqrt(3) / 2  =>  s = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / sqrt3

    def pairwise_overlap_check(positions, angles):
        """Check if any two hexagons overlap using SAT on 12 axes."""
        verts_list = [get_full_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                # SAT: check all 12 edge normals
                overlap = True
                for k in range(6):
                    for (v, w) in [(v1, v2), (v2, v1)]:
                        e = v[(k + 1) % 6] - v[k]
                        n = np.array([-e[1], e[0]])
                        nn = np.linalg.norm(n)
                        if nn < 1e-12:
                            continue
                        n = n / nn
                        p1 = v @ n
                        p2 = w @ n
                        if np.max(p1) < np.min(p2) or np.max(p2) < np.min(p1):
                            overlap = False
                            break
                    if not overlap:
                        break
                if overlap:
                    return False
        return True

    # Two-ring topology: 1 center hexagon + 11 outer hexagons
    # Outer 11: arranged as a hexagonal ring of 12 minus 1 (to make 11)
    # Actually, let's do: center at origin, and 11 around it.
    # The natural hexagonal packing has 6 neighbors at distance 2*sqrt(3)/2 = sqrt(3) for flat-top... 
    # Let's think: two flat-top unit hexagons touching side-to-side have centers at distance sqrt(3).
    # For a ring of 6 around a center: centers at distance sqrt(3) from center, at angles 0,60,...,300.
    # For 11 outer: we can do a ring of 6 at radius r1, and 5 more at a larger radius r2.
    # Or: 6 at radius r, and 5 at radius r * something.
    
    # Simpler: use a "hexagonal number" arrangement. 1 + 6 = 7, then next ring has 12.
    # We need 12 total. So 1 center + 11 outer.
    # Let's place 6 at radius R at angles 0,60,120,180,240,300
    # And 5 at radius R2 at angles 30,90,150,210,270 (offset by 30 deg)
    
    def build_two_ring(scale, outer_rot_deg, hex_rot_deg):
        """
        scale: radial scale factor for the outer ring
        outer_rot_deg: rotation of the entire outer ring pattern
        hex_rot_deg: rotation angle for all inner hexagons
        """
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Center hexagon
        positions[0] = [0.0, 0.0]
        angles[0] = hex_rot_deg
        
        # 6 hexagons at radius R, angles 0,60,...,300 + outer_rot
        R = scale * sqrt3  # base distance for touching flat-top hexagons
        for k in range(6):
            ang = math.radians(outer_rot_deg + k * 60)
            positions[1 + k] = [R * math.cos(ang), R * math.sin(ang)]
            angles[1 + k] = hex_rot_deg + 30  # rotated 30 deg relative to center
        
        # 5 hexagons at radius R2, angles 30,90,150,210,270 + outer_rot
        R2 = scale * sqrt3 * 1.8  # further out
        for k in range(5):
            ang = math.radians(outer_rot_deg + 30 + k * 72)  # 5 evenly spaced
            positions[6 + k] = [R2 * math.cos(ang), R2 * math.sin(ang)]
            angles[6 + k] = hex_rot_deg + 30
        
        return positions, angles

    def evaluate(scale, outer_rot_deg, hex_rot_deg):
        positions, angles = build_two_ring(scale, outer_rot_deg, hex_rot_deg)
        if not pairwise_overlap_check(positions, angles):
            return float('inf')
        return min_outer_side(positions, angles, 0.0)

    best_side = float('inf')
    best_params = None

    # Grid search over scale, outer_rot, hex_rot
    # Scale: the outer ring radius. For touching hexagons, scale ~ 1.0
    # Let's search scale in [0.9, 1.6], outer_rot in [0, 60) step 5, hex_rot in [0, 60) step 10
    scales = np.linspace(0.9, 1.6, 15)
    outer_rots = np.arange(0, 60, 5)
    hex_rots = np.arange(0, 60, 10)

    for s in scales:
        for orot in outer_rots:
            for hrot in hex_rots:
                if time.time() > deadline - 20:
                    break
                side = evaluate(s, orot, hrot)
                if side < best_side:
                    best_side = side
                    best_params = (s, orot, hrot)

    # Local refinement around best
    if best_params is not None:
        s0, orot0, hrot0 = best_params
        # Fine grid
        for s in np.linspace(s0 - 0.1, s0 + 0.1, 11):
            for orot in np.linspace(orot0 - 5, orot0 + 5, 11):
                for hrot in np.linspace(hrot0 - 5, hrot0 + 5, 11):
                    if time.time() > deadline - 10:
                        break
                    side = evaluate(s, orot % 60, hrot % 60)
                    if side < best_side:
                        best_side = side
                        best_params = (s, orot % 60, hrot % 60)

    # Also try a few alternative topologies
    # Topology 2: 2 concentric rings of 6, no center
    def build_two_rings_66(scale1, scale2, rot1, rot2, hex_rot):
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            ang = math.radians(rot1 + k * 60)
            positions[k] = [scale1 * math.cos(ang), scale1 * math.sin(ang)]
            angles[k] = hex_rot
        for k in range(6):
            ang = math.radians(rot2 + k * 60)
            positions[6 + k] = [scale2 * math.cos(ang), scale2 * math.sin(ang)]
            angles[6 + k] = hex_rot + 30
        return positions, angles

    for s1 in np.linspace(0.5, 1.2, 8):
        for s2 in np.linspace(1.2, 2.0, 8):
            for r1 in np.arange(0, 60, 15):
                for r2 in np.arange(0, 60, 15):
                    if time.time() > deadline - 5:
                        break
                    positions, angles = build_two_rings_66(s1, s2, r1, r2, 0)
                    if pairwise_overlap_check(positions, angles):
                        side = min_outer_side(positions, angles, 0.0)
                        if side < best_side:
                            best_side = side
                            best_params = ('66', s1, s2, r1, r2)

    # Fallback: if no valid config found, use a large safe configuration
    if best_params is None or best_side == float('inf'):
        # Simple grid placement
        positions = np.array([
            [0, 0], [2.5, 0], [-2.5, 0],
            [1.25, 2.2], [-1.25, 2.2], [0, 4.4],
            [1.25, -2.2], [-1.25, -2.2], [0, -4.4],
            [3.75, 2.2], [-3.75, 2.2], [3.75, -2.2]
        ])
        angles = np.zeros(N)
        best_side = min_outer_side(positions, angles, 0.0)
    else:
        if isinstance(best_params[0], str):
            # 66 topology
            _, s1, s2, r1, r2 = best_params
            positions, angles = build_two_rings_66(s1, s2, r1, r2, 0)
        else:
            s, orot, hrot = best_params
            positions, angles = build_two_ring(s, orot, hrot)

    inner = []
    for i in range(N):
        inner.append([float(positions[i][0]), float(positions[i][1]), float(angles[i])])

    return inner, (0.0, 0.0), float(best_side), 0.0
