"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """Pack 11 unit regular hexagons in a regular hexagon, minimizing outer side length."""
    
    sqrt3 = math.sqrt(3)
    
    def hex_vertices(cx, cy, side, angle_deg):
        a0 = math.radians(angle_deg)
        return np.array([(cx + side * math.cos(a0 + i * math.pi / 3),
                         cy + side * math.sin(a0 + i * math.pi / 3)) for i in range(6)])
    
    def overlap(v1, v2, v3, v4):
        """SAT overlap check for two unit hexagons at (v1,v2) and (v3,v4) with angle 0."""
        h1 = hex_vertices(v1, v2, 1.0, 0.0)
        h2 = hex_vertices(v3, v4, 1.0, 0.0)
        for h in [h1, h2]:
            for i in range(6):
                j = (i + 1) % 6
                ex, ey = h[j][0] - h[i][0], h[j][1] - h[i][1]
                nx, ny = -ey, ex
                n = math.sqrt(nx*nx + ny*ny)
                if n < 1e-10: continue
                nx, ny = nx/n, ny/n
                p1 = [x*nx + y*ny for x, y in h1]
                p2 = [x*nx + y*ny for x, y in h2]
                if max(p1) < min(p2) or max(p2) < min(p1):
                    return False
        return True
    
    def min_side(inner, cx, cy):
        """Min side of outer hexagon (angle 0, center cx,cy) containing all inner."""
        m = 0.0
        for x, y, _ in inner:
            for vx, vy in hex_vertices(x, y, 1.0, 0.0):
                dx, dy = vx - cx, vy - cy
                for i in range(6):
                    a = math.radians(30 + i * 60)
                    m = max(m, abs(dx * math.cos(a) + dy * math.sin(a)))
        return m / (sqrt3 / 2)
    
    def obj(p):
        inner = [(p[2*i], p[2*i+1], 0.0) for i in range(N)]
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if overlap(inner[i][0], inner[i][1], inner[j][0], inner[j][1]):
                    pen += 500
        cx = sum(x for x, y, _ in inner) / N
        cy = sum(y for x, y, _ in inner) / N
        return min_side(inner, cx, cy) + pen
    
    # Hex lattice initial: 3+4+3+1
    pos = []
    for k in range(3): pos.append((k*sqrt3 - sqrt3, 0))
    for k in range(4): pos.append((k*sqrt3 - 1.5*sqrt3, 1.5))
    for k in range(3): pos.append((k*sqrt3 - sqrt3, 3.0))
    pos.append((0, 4.5))
    
    best_p = [c for p in pos for c in p]
    best_v = obj(best_p)
    cur_p = best_p[:]
    cur_v = best_v
    T, cool, step = 1.0, 0.9995, 0.05
    
    for _ in range(15000):
        np_ = cur_p[:]
        i = np.random.randint(len(np_))
        np_[i] += np.random.normal(0, step)
        nv = obj(np_)
        if nv < cur_v or np.random.random() < math.exp(-(nv - cur_v)/T):
            cur_p, cur_v = np_, nv
            if cur_v < best_v:
                best_v, best_p = cur_v, cur_p[:]
        T *= cool
        if T < 0.005: break
    
    inner = [(best_p[2*i], best_p[2*i+1], 0.0) for i in range(N)]
    cx = sum(x for x, y, _ in inner) / N
    cy = sum(y for x, y, _ in inner) / N
    side = min_side(inner, cx, cy)
    return inner, (float(cx), float(cy)), float(side), 0.0
# EVOLVE-BLOCK-END
