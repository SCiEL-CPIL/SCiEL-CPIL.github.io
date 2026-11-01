import numpy as np
import time
from scipy.optimize import minimize, least_squares
from itertools import combinations, product
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hex_vertices(cx, cy, s=1.0, angle=0.0):
        """Return 6 vertices of unit hexagon centered at (cx, cy) with rotation angle."""
        a = np.radians(angle)
        pts = []
        for k in range(6):
            theta = a + np.radians(60 * k)
            pts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
        return np.array(pts)

    def required_side(positions, angles):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with angle 0 to contain all given hexagons.
        The outer hexagon has vertices at distance R = s from center? No, side length s.
        Distance from center to vertex is s. Distance to edge is s*sqrt(3)/2.
        The containment condition for a point (x,y) to be inside a regular hexagon
        of side s centered at origin with flat-top (vertices at 0, 60, ...) is:
        |x| <= s * sqrt(3)/2 is NOT correct for all directions.
        
        For a regular hexagon with side s, the support function h(u) in direction u is:
        h(u) = s * max(0, cos(theta - k*pi/3)) / sin(pi/6) ? No.
        
        Let's use the 6 half-planes.
        The edges are at distance apothem = s * sqrt(3)/2 from the center.
        The normals to the edges are at angles 30, 90, 150, 210, 270, 330 degrees.
        Wait, if vertices are at 0, 60, ..., the edges are between them.
        Edge between 0 and 60 deg: midpoint angle 30. Normal is 30 deg.
        So the constraints are:
        x * cos(30) + y * sin(30) <= s * sqrt(3)/2
        x * cos(90) + y * sin(90) <= s * sqrt(3)/2
        ...
        
        So s >= (x * cos(30) + y * sin(30)) / (sqrt(3)/2)
        s >= (x * cos(90) + y * sin(90)) / (sqrt(3)/2)
        
        Let's verify. If vertex is at (s, 0).
        Constraint 1: s * cos(30) + 0 <= s * sqrt(3)/2. True.
        Constraint 2: 0 + s * sin(90) = s. Is s <= s * sqrt(3)/2? No, s > s*0.866.
        So the vertex (s,0) violates the second constraint?
        
        Ah, the "flat-top" hexagon usually has flat edges on top/bottom.
        Vertices at 30, 90, 150, 210, 270, 330?
        If vertices are at 0, 60, 120... (pointy top), then edges are at 30, 90...
        If vertices are at 30, 90... (flat top), then edges are at 0, 60...
        
        The current program uses `hex_vertices` with `a=0`.
        k=0: (s, 0). k=1: (s/2, s*sqrt(3)/2).
        This is a "pointy-top" hexagon (vertex at 0 deg).
        The edges are between these.
        Edge 0-1: from (s,0) to (s/2, s*sqrt(3)/2).
        Midpoint: (3s/4, s*sqrt(3)/4). Angle: atan(sqrt(3)/3) = 30 deg.
        Normal: 30 deg.
        Distance from origin to this line:
        Line eq: x cos(30) + y sin(30) = d.
        Plug in (s, 0): s * sqrt(3)/2 = d.
        So apothem is s * sqrt(3)/2.
        
        The 6 normals are at 30, 90, 150, 210, 270, 330.
        Constraints for point (x,y) to be inside:
        x cos(30) + y sin(30) <= s * sqrt(3)/2
        x cos(90) + y sin(90) <= s * sqrt(3)/2
        x cos(150) + y sin(150) <= s * sqrt(3)/2
        ...
        
        So s >= (x cos(30) + y sin(30)) * 2/sqrt(3)
        s >= (x cos(90) + y sin(90)) * 2/sqrt(3)
        
        We need max over all vertices of all inner hexagons.
        """
        all_v = []
        for i in range(N):
            v = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_v.append(v)
        all_v = np.vstack(all_v)
        x, y = all_v[:, 0], all_v[:, 1]
        
        # Normals at 30, 90, 150, 210, 270, 330
        # cos(30)=sqrt(3)/2, sin(30)=1/2
        # cos(90)=0, sin(90)=1
        # cos(150)=-sqrt(3)/2, sin(150)=1/2
        # cos(210)=-sqrt(3)/2, sin(210)=-1/2
        # cos(270)=0, sin(270)=-1
        # cos(330)=sqrt(3)/2, sin(330)=-1/2
        
        s3 = np.sqrt(3.0)
        # Factor 2/sqrt(3)
        f = 2.0 / s3
        
        vals = np.array([
            x * (s3/2) + y * 0.5,
            x * 0 + y * 1.0,
            x * (-s3/2) + y * 0.5,
            x * (-s3/2) + y * (-0.5),
            x * 0 + y * (-1.0),
            x * (s3/2) + y * (-0.5)
        ])
        # The constraint is val <= s * sqrt(3)/2
        # So s >= val / (sqrt(3)/2) = val * 2/sqrt(3)
        max_val = np.max(vals)
        return max_val * f

    def sat_disjoint(v1, v2, tol=1e-7):
        """SAT: return True if two convex polygons are disjoint."""
        for poly in (v1, v2):
            for i in range(6):
                j = (i + 1) % 6
                ex, ey = poly[j] - poly[i]
                nx, ny = -ey, ex
                nl = np.hypot(nx, ny)
                if nl < 1e-12:
                    continue
                nx, ny = nx / nl, ny / nl
                p1 = v1 @ np.array([nx, ny])
                p2 = v2 @ np.array([nx, ny])
                if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
                    return True
        return False

    def all_disjoint(positions, angles, tol=1e-6):
        """Check all pairs are disjoint."""
        verts = [hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(verts[i], verts[j], tol):
                    return False
        return True

    def penalty(positions, angles):
        """Penalty for overlaps."""
        verts = [hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]) for i in range(N)]
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(verts[i], verts[j], 1e-7):
                    pen += 1.0
        return pen

    def objective(params):
        """
        params: 3*N array [x1, y1, a1, x2, y2, a2, ...]
        """
        positions = params[0::3].reshape(N, 2) # This is wrong, need to reshape properly
        # Better:
        params = np.array(params)
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        side = required_side(positions, angles)
        pen = penalty(positions, angles)
        return side + 100.0 * pen

    def generate_contact_graph_topology():
        """
        Generate a random valid contact graph (planar, max degree 6, etc.)
        and solve for positions.
        This is complex. Instead, we use a heuristic: 
        Start with a known good packing (hexagonal lattice) and perturb.
        """
        pass

    # Strategy: Hybrid Topological/Local Search
    # 1. Generate initial configurations based on "Contact Graph" heuristics
    #    - Hexagonal Lattice
    #    - Circular Ring
    #    - Random but constrained
    
    def init_hex_lattice():
        s3 = np.sqrt(3)
        pts = []
        angs = []
        # Center
        pts.append([0, 0])
        angs.append(0.0)
        # Ring 1 (6 hexes)
        for k in range(6):
            a = np.radians(60 * k)
            pts.append([s3 * np.cos(a), 1.5 * np.sin(a)])
            angs.append(0.0)
        # 5 more on outer ring
        for k in range(5):
            a = np.radians(60 * k + 30)
            r = 2 * 1.5 * s3 # approximate
            pts.append([r * np.cos(a), r * np.sin(a)])
            angs.append(0.0)
        return np.array(pts[:N]), np.array(angs[:N])

    def init_circular():
        pts = np.zeros((N, 2))
        angs = np.zeros(N)
        for i in range(N):
            a = 2 * np.pi * i / N
            r = 2.5
            pts[i] = [r * np.cos(a), r * np.sin(a)]
            angs[i] = 0.0
        return pts, angs

    def init_random(seed=42):
        np.random.seed(seed)
        pts = np.random.uniform(-4, 4, (N, 2))
        angs = np.random.uniform(0, 60, N) # Hexagons are symmetric under 60 deg rotation
        return pts, angs

    inits = [init_hex_lattice(), init_circular(), init_random(1), init_random(7), init_random(99), init_random(123)]

    best_pos = None
    best_ang = None
    best_side = float('inf')

    # Phase 1: Local Refinement with L-BFGS-B
    # We optimize positions and angles.
    for init_pos, init_ang in inits:
        if time.time() > deadline - 30:
            break

        # Ensure initial is valid or scale it
        pos0 = init_pos.copy()
        ang0 = init_ang.copy()
        valid = False
        for scale in [1.0, 1.2, 1.5, 2.0, 3.0]:
            if all_disjoint(pos0 * scale, ang0, 1e-5):
                pos0 = pos0 * scale
                valid = True
                break
        if not valid:
            continue

        params0 = np.concatenate([pos0.flatten(), ang0])
        
        # Try multiple penalty weights
        for pw in [50.0, 20.0, 10.0, 5.0]:
            if time.time() > deadline - 30:
                break
            
            def obj_pw(p):
                positions = p[:2*N].reshape(N, 2)
                angles = p[2*N:]
                side = required_side(positions, angles)
                pen = penalty(positions, angles)
                return side + pw * pen

            try:
                res = minimize(
                    obj_pw, params0,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 100}
                )
                pos = res.x[:2*N].reshape(N, 2)
                ang = res.x[2*N:]
                if all_disjoint(pos, ang, 1e-6):
                    s = required_side(pos, ang)
                    if s < best_side:
                        best_side = s
                        best_pos = pos
                        best_ang = ang
            except Exception:
                continue

    # Phase 2: Simulated Annealing on the best found configuration
    if best_pos is not None and time.time() < deadline - 20:
        cur_pos = best_pos.copy()
        cur_ang = best_ang.copy()
        cur_side = best_side
        T = 0.5
        for it in range(50000):
            if time.time() > deadline - 5:
                break
            
            # Move one hexagon
            idx = np.random.randint(N)
            dx, dy = np.random.normal(0, 0.3, 2)
            da = np.random.normal(0, 5.0) * np.pi / 180.0
            
            new_pos = cur_pos.copy()
            new_ang = cur_ang.copy()
            new_pos[idx] += [dx, dy]
            new_ang[idx] += da
            
            if not all_disjoint(new_pos, new_ang, 1e-6):
                continue
            
            ns = required_side(new_pos, new_ang)
            d = ns - cur_side
            if d < 0 or np.random.random() < np.exp(-d / max(T, 1e-8)):
                cur_pos = new_pos
                cur_ang = new_ang
                cur_side = ns
                if cur_side < best_side:
                    best_side = cur_side
                    best_pos = cur_pos.copy()
                    best_ang = cur_ang.copy()
            
            T *= 0.9999

    # Fallback
    if best_pos is None:
        best_pos = np.array([[i * 3.0, 0.0] for i in range(N)], dtype=float)
        best_ang = np.zeros(N)
        best_side = required_side(best_pos, best_ang)

    # Final check
    if not all_disjoint(best_pos, best_ang, 1e-5):
        best_pos = np.array([[i * 4.0, 0.0] for i in range(N)], dtype=float)
        best_ang = np.zeros(N)
        best_side = required_side(best_pos, best_ang)

    final_side = required_side(best_pos, best_ang)

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(np.degrees(best_ang[i]))])

    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
