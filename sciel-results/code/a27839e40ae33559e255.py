"""Optimized packing of 12 unit hexagons inside a regular hexagon."""
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    rows, cols = 2, 6
    spacing_x = 1.5
    spacing_y = math.sqrt(3.0)
    angle = 0.0
    r = 1.0

    inner = []
    for i in range(rows):
        for j in range(cols):
            x = (j - (cols - 1) / 2.0) * spacing_x
            y = (i - (rows - 1) / 2.0) * spacing_y
            inner.append([x, y, angle])

    def required_side(inner_list, outer_a_deg, center=(0.0, 0.0)):
        oa = math.radians(outer_a_deg)
        dirs = []
        for k in range(6):
            th = oa + math.radians(90 + 60 * k)
            dirs.append((math.cos(th), math.sin(th)))
        best_inradius = 0.0
        for (px, py, aa) in inner_list:
            aa_r = math.radians(aa)
            for (dxn, dyn) in dirs:
                m = 0.0
                for k in range(6):
                    th = math.radians(60 * k) + aa_r
                    vx, vy = math.cos(th), math.sin(th)
                    val = dxn * (px - center[0] + vx) + dyn * (py - center[1] + vy)
                    if val > m:
                        m = val
                if m > best_inradius:
                    best_inradius = m
        return 2.0 * best_inradius / math.sqrt(3.0)

    best_inner = [list(p) for p in inner]
    best_s = required_side(best_inner, 0.0)
    best_outer_a = 0.0

    # Try rotating outer hexagon
    for deg in range(0, 60, 3):
        if time.time() > deadline:
            break
        s = required_side(best_inner, deg)
        if s < best_s - 1e-9:
            best_s = s
            best_outer_a = deg

    # Try small block translations
    for dx in [0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15]:
        for dy in [0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15]:
            if time.time() > deadline:
                break
            shifted = [[x + dx, y + dy, a] for (x, y, a) in best_inner]
            s = required_side(shifted, best_outer_a)
            if s < best_s - 1e-9:
                best_s = s
                best_inner = shifted

    return best_inner, (0.0, 0.0), best_s, best_outer_a
# EVOLVE-BLOCK-END
