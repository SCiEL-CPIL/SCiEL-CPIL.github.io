"""Combined program: hexagonal lattice packing + local optimization."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 150s limit, keep 10s margin

    # Lattice vectors for hexagonal packing
    # Distance between centers of touching unit hexagons is sqrt(3)
    # We add a small epsilon to ensure strict disjointness
    eps = 0.01
    d = math.sqrt(3) + eps

    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])

    # Build a cluster of 12 hexagons using ring-based construction
    centers = [np.array([0.0, 0.0])]

    # Ring 1: 6 neighbors of center
    basis = [u, v, v - u, -u, -v, u - v]
    ring1_vectors = [vec.copy() for vec in basis]
    centers.extend(ring1_vectors)

    # Ring 2: find neighbors of ring1 that are not in ring0 or ring1
    ring2_candidates = []
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            # Check if p2 is in Ring 0
            if np.allclose(p2, np.array([0.0, 0.0]), atol=eps*0.5):
                continue
            # Check if p2 is in Ring 1
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j, atol=eps*0.5):
                    is_r1 = True
                    break
            if is_r1:
                continue
            # Avoid duplicates
            duplicate = False
            for pc in ring2_candidates:
                if np.allclose(p2, pc, atol=eps*0.5):
                    duplicate = True
                    break
            if not duplicate:
                ring2_candidates.append(p2.copy())

    # Sort Ring 2 candidates by distance to origin and pick 5 closest
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(min(5, len(ring2_candidates))):
        centers.append(ring2_candidates[i].copy())

    # If we don't have exactly 12, add more from ring2 or generate additional
    while len(centers) < N:
        # Add more from remaining ring2 candidates
        if len(ring2_candidates) > len(centers) - 7:
            idx = len(centers) - 7
            if idx < len(ring2_candidates):
                centers.append(ring2_candidates[idx].copy())
            else:
                break
        else:
            # Generate a new position further out
            angle = random.uniform(0, 2*math.pi)
            dist = d * 2.5
            new_c = np.array([dist*math.cos(angle), dist*math.sin(angle)])
            centers.append(new_c)
        if len(centers) > 50:
            break

    centers = centers[:N]

    # Convert to list of [x, y, angle]
    # Rotate hexagons by 30 degrees so they are 'flat-top' relative to the x-axis
    inner = [[float(c[0]), float(c[1]), 30.0] for c in centers]

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon."""
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px, py) is inside or on boundary of hexagon."""
        dx = px - cx
        dy = py - cy
        angle = -math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a

        # For a unit hexagon centered at origin with angle=0:
        # Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
        # The hexagon is the intersection of 6 half-planes
        # Check using the 6 edge normals
        sqrt3 = math.sqrt(3)
        # Edge 1: from (1,0) to (0.5, sqrt3/2), normal points outward
        # The half-plane condition: for each edge, the point must be on the interior side
        
        # Simpler approach: check against the 6 supporting lines
        # For hexagon with vertices at angles 0,60,120,180,240,300 (side=1):
        # The apothem is sqrt(3)/2
        # Normal directions are at 30, 90, 150, 210, 270, 330 degrees
        # Projection onto each normal must be <= sqrt(3)/2 * side
        
        for k in range(6):
            normal_angle = math.radians(30 + 60 * k)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = lx * nx + ly * ny
            if proj > side * sqrt3 / 2 + 1e-9:
                return False
        return True

    def hexagons_disjoint(inner_positions, tol=1e-6):
        """Check if all unit hexagons are disjoint (no overlapping interiors)."""
        for i in range(len(inner_positions)):
            xi, yi, ai = inner_positions[i]
            vi = get_hex_vertices(xi, yi, 1.0, ai)
            for j in range(i + 1, len(inner_positions)):
                xj, yj, aj = inner_positions[j]
                vj = get_hex_vertices(xj, yj, 1.0, aj)
                
                # Check if any vertex of i is inside j
                for vx, vy in vi:
                    if point_in_hexagon(vx, vy, xj, yj, 1.0, aj):
                        return False
                # Check if any vertex of j is inside i
                for vx, vy in vj:
                    if point_in_hexagon(vx, vy, xi, yi, 1.0, ai):
                        return False
                
        return True

    def compute_outer_side(inner_positions):
        """Compute the minimum side length of a regular hexagon that contains all inner hexagons."""
        all_verts = []
        for x, y, a in inner_positions:
            verts = get_hex_vertices(x, y, 1.0, a)
            all_verts.extend(verts)
        
        if not all_verts:
            return 0.0, (0.0, 0.0), 0.0
        
        # Center should be near the centroid of all vertices
        cx = sum(v[0] for v in all_verts) / len(all_verts)
        cy = sum(v[1] for v in all_verts) / len(all_verts)
        
        best_side = float('inf')
        best_center = (cx, cy)
        best_angle = 0.0
        
        # Try different outer hexagon rotations
        for theta_deg in np.linspace(0, 60, 61):
            theta = math.radians(theta_deg)
            max_proj = 0.0
            for k in range(6):
                normal_angle = theta + math.pi/6 + k * math.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                for px, py in all_verts:
                    proj = (px - cx) * nx + (py - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
            s = 2.0 * max_proj / math.sqrt(3)
            if s < best_side:
                best_side = s
                best_center = (cx, cy)
                best_angle = float(theta_deg)
        
        return best_side, best_center, best_angle

    # Initial evaluation
    if hexagons_disjoint(inner):
        side, center, angle = compute_outer_side(inner)
        outer_side = side
        outer_center = center
        outer_angle = angle
    else:
        # Fallback: use a larger arrangement
        outer_side = 40.0
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        side, center, angle = compute_outer_side(inner)
        outer_side = side
        outer_center = center
        outer_angle = angle

    # Local optimization: perturb positions slightly
    if time.time() < deadline - 10:
        best_arr = [list(x) for x in inner]
        best_side = outer_side
        
        for iteration in range(2000):
            if time.time() > deadline - 5:
                break
            
            new_inner = [list(x) for x in best_arr]
            # Perturb a random hexagon
            idx = random.randint(0, N - 1)
            dx = random.uniform(-0.3, 0.3)
            dy = random.uniform(-0.3, 0.3)
            da = random.uniform(-20, 20)
            new_inner[idx][0] += dx
            new_inner[idx][1] += dy
            new_inner[idx][2] += da
            
            if hexagons_disjoint(new_inner):
                side, center, angle = compute_outer_side(new_inner)
                if side < best_side - 1e-6:
                    best_side = side
                    best_arr = new_inner
                    outer_side = side
                    outer_center = center
                    outer_angle = angle
                    inner = [list(x) for x in best_arr]
                # else revert (already a copy, so nothing to do)
            
            # Occasionally try a larger perturbation
            if iteration % 100 == 0:
                new_inner = [list(x) for x in best_arr]
                idx = random.randint(0, N - 1)
                dx = random.uniform(-1.0, 1.0)
                dy = random.uniform(-1.0, 1.0)
                da = random.uniform(-45, 45)
                new_inner[idx][0] += dx
                new_inner[idx][1] += dy
                new_inner[idx][2] += da
                
                if hexagons_disjoint(new_inner):
                    side, center, angle = compute_outer_side(new_inner)
                    if side < best_side - 1e-6:
                        best_side = side
                        best_arr = new_inner
                        outer_side = side
                        outer_center = center
                        outer_angle = angle
                        inner = [list(x) for x in best_arr]

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
