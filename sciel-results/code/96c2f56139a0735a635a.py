import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hex_vertices(cx, cy, s=1.0):
        """Return 6 vertices of unit hexagon centered at (cx, cy), pointy-top orientation."""
        # Pointy-top: vertices at 30, 90, 150, 210, 270, 330 degrees
        a = np.radians(30)
        pts = np.array([[cx + s * np.cos(a + np.radians(60 * k)),
                         cy + s * np.sin(a + np.radians(60 * k))] for k in range(6)])
        return pts

    def required_side(positions):
        """Exact minimum outer side (angle 0, centered at origin) to contain all unit hexagons."""
        all_v = np.vstack([hex_vertices(*p) for p in positions])
        x, y = all_v[:, 0], all_v[:, 1]
        s3 = np.sqrt(3.0)
        # Outer hexagon (pointy-top) inequalities for containment:
        # |x| <= s
        # |y| <= 2s/sqrt(3)
        # |x - y/sqrt(3)| <= s/sqrt(3)  ->  sqrt(3)x - y <= s
        # |x + y/sqrt(3)| <= s/sqrt(3)  ->  sqrt(3)x + y <= s
        vals = np.array([
            np.abs(x),
            np.abs(y) * s3 / 2,
            np.abs(x * s3 - y),
            np.abs(x * s3 + y)
        ])
        return np.max(vals)

    def min_vertex_edge_dist(v1, v2):
        """
        Compute the minimum distance from any vertex of v1 to any edge of v2,
        and vice versa. This is a valid separation measure for convex polygons.
        If the minimum is < 0, they overlap.
        """
        min_dist = float('inf')
        
        # Vertices of v1 to edges of v2
        for v in v1:
            for i in range(6):
                p1 = v2[i]
                p2 = v2[(i + 1) % 6]
                edge_vec = p2 - p1
                edge_len_sq = np.dot(edge_vec, edge_vec)
                if edge_len_sq < 1e-10:
                    continue
                t = np.dot(v - p1, edge_vec) / edge_len_sq
                t = np.clip(t, 0, 1)
                closest = p1 + t * edge_vec
                dist = np.linalg.norm(v - closest)
                min_dist = min(min_dist, dist)
                
        # Vertices of v2 to edges of v1
        for v in v2:
            for i in range(6):
                p1 = v1[i]
                p2 = v1[(i + 1) % 6]
                edge_vec = p2 - p1
                edge_len_sq = np.dot(edge_vec, edge_vec)
                if edge_len_sq < 1e-10:
                    continue
                t = np.dot(v - p1, edge_vec) / edge_len_sq
                t = np.clip(t, 0, 1)
                closest = p1 + t * edge_vec
                dist = np.linalg.norm(v - closest)
                min_dist = min(min_dist, dist)
                
        return min_dist

    def penalty(positions, power=2):
        """
        Smooth penalty for overlaps.
        For each pair, if they are closer than 0 (overlapping), add (0 - dist)^power.
        We use a soft threshold to make it differentiable-ish for the optimizer.
        Actually, min_vertex_edge_dist is piecewise smooth.
        """
        verts = [hex_vertices(*p) for p in positions]
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = min_vertex_edge_dist(verts[i], verts[j])
                if d < 1e-6: # Overlapping or touching
                    # Penalty grows quadratically with penetration depth
                    pen += (1e-6 - d) ** power
        return pen

    def objective(params):
        positions = params.reshape(N, 2)
        side = required_side(positions)
        pen = penalty(positions)
        # Weight penalty heavily to ensure disjointness
        return side + 100.0 * pen

    def all_disjoint(positions, tol=1e-5):
        """Strict check for validity."""
        verts = [hex_vertices(*p) for p in positions]
        for i in range(N):
            for j in range(i + 1, N):
                d = min_vertex_edge_dist(verts[i], verts[j])
                if d < -tol: # Significant overlap
                    return False
        return True

    # Initial configurations
    def init_hex_lattice():
        """
        Hexagonal close-packing.
        For pointy-top hexagons, neighbors are at distance 2*apothem = sqrt(3) horizontally?
        No, for pointy-top, width is sqrt(3), height is 2.
        Horizontal distance between centers: sqrt(3).
        Vertical distance: 1.5.
        Offset: sqrt(3)/2.
        """
        s3 = np.sqrt(3)
        pts = []
        # Center
        pts.append([0, 0])
        # Ring 1 (6 hexes)
        # Neighbors of (0,0) in pointy-top hex grid:
        # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
        neighbors = [
            [s3, 0],
            [s3/2, 1.5],
            [-s3/2, 1.5],
            [-s3, 0],
            [-s3/2, -1.5],
            [s3/2, -1.5]
        ]
        pts.extend(neighbors)
        # We need 12. We have 7. Add 5 more.
        # Next ring or just extend?
        # Let's just add 5 more in a rough circle or extending the lattice.
        # To keep them disjoint, let's place them further out.
        # Ring 2 candidates:
        # (2*sqrt(3), 0), (1.5*sqrt(3), 3), (0, 3), (-1.5*sqrt(3), 3), (-2*sqrt(3), 0), etc.
        # Let's just use a spiral or random perturbations in init.
        # For now, duplicate some or place them in a larger circle.
        for k in range(5):
            a = 2 * np.pi * k / 5
            r = 3.0
            pts.append([r * np.cos(a), r * np.sin(a)])
        return np.array(pts[:N])

    def init_circular():
        pts = np.zeros((N, 2))
        for i in range(N):
            a = 2 * np.pi * i / N
            r = 2.5
            pts[i] = [r * np.cos(a), r * np.sin(a)]
        return pts

    def init_random(seed):
        np.random.seed(seed)
        # Random positions in a box
        return np.random.uniform(-4, 4, (N, 2))

    inits = [
        init_hex_lattice(),
        init_circular(),
        init_random(1),
        init_random(7),
        init_random(13),
        init_random(42)
    ]

    best_pos = None
    best_side = float('inf')

    # Optimization loop
    for init in inits:
        if time.time() > deadline - 20:
            break

        pos0 = init.copy()
        # Scale up if overlapping
        for scale in [1.0, 1.2, 1.5, 2.0, 3.0]:
            if all_disjoint(pos0, 1e-4):
                break
            pos0 = init * scale
        else:
            # If still overlapping, randomize
            pos0 = init_random(int(time.time() * 1000) % 10000)
            pos0 = pos0 * 2.0

        params0 = pos0.flatten()
        
        try:
            # Use L-BFGS-B for local optimization
            res = minimize(
                objective, params0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            pos = res.x.reshape(N, 2)
            
            # Check validity
            if all_disjoint(pos, 1e-5):
                s = required_side(pos)
                if s < best_side:
                    best_side = s
                    best_pos = pos.copy()
                    
        except Exception:
            continue

    # If no valid solution found, try a fallback: spread them out linearly
    if best_pos is None:
        # Fallback: place them in a line far apart
        best_pos = np.array([[i * 3.0, 0.0] for i in range(N)], dtype=float)
        best_side = required_side(best_pos)
    else:
        # Final refinement: try to reduce side length further with a tighter tolerance
        # This might be skipped if time is tight
        if time.time() < deadline - 5:
            # One more pass with the best found
            try:
                res = minimize(
                    objective, best_pos.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                pos = res.x.reshape(N, 2)
                if all_disjoint(pos, 1e-5):
                    s = required_side(pos)
                    if s < best_side:
                        best_side = s
                        best_pos = pos.copy()
            except Exception:
                pass

    final_side = required_side(best_pos)

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), 0.0])

    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
