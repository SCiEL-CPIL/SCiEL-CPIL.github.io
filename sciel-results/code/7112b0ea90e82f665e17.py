import numpy as np
import time
import scipy.optimize as opt

def _hex_vertices(cx, cy, side, angle_deg):
    """
    Returns vertices of a regular hexagon.
    """
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _sat_gap(v1, v2):
    """
    Computes the minimum separation distance between two convex polygons v1, v2.
    Returns a positive value if separated, negative if overlapping.
    """
    min_gap = np.inf
    
    # Axes to test: normals of edges of v1 and v2
    axes = []
    for v in [v1, v2]:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            edge = p2 - p1
            # Normal to edge
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n = n / norm
                axes.append(n)
    
    # Project onto each axis
    for n in axes:
        proj1 = v1.dot(n)
        proj2 = v2.dot(n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Gap is the distance between the intervals
        # If max1 < min2, gap is min2 - max1
        # If max2 < min1, gap is min1 - max2
        # Otherwise, they overlap
        gap1 = min2 - max1
        gap2 = min1 - max2
        
        # The minimum gap on this axis is the smallest positive separation
        # If both are negative, they overlap on this axis.
        # We want the minimum gap over all axes.
        # If there is an axis where they are separated, the gap is positive.
        # If they overlap, all axes will show overlap (gap < 0)? No, SAT says if they are disjoint, there is at least one axis where they are separated.
        # If they are intersecting, they are not separated on ANY axis.
        # So, if they are disjoint, min_gap will be the smallest positive gap found.
        # If they are intersecting, all gaps will be negative?
        # Let's check. If they intersect, for any axis, the intervals overlap.
        # So min2 - max1 < 0 AND min1 - max2 < 0.
        # So we take the minimum of these values?
        # If they are disjoint, one of the gaps is positive.
        # We want the minimum separation.
        
        # Actually, the "gap" on an axis is:
        # If disjoint: min(min2 - max1, min1 - max2) is the separation distance on this axis.
        # If intersecting: both are negative. The "depth" of overlap is max(max1 - min2, max2 - min1).
        # We want a single metric.
        # Let's define gap_axis = min(min2 - max1, min1 - max2).
        # If they are disjoint, this is positive.
        # If they intersect, this is negative.
        # We want the minimum gap over all axes.
        # If the minimum gap is positive, they are disjoint by that amount.
        # If the minimum gap is negative, they intersect.
        
        gap_axis = min(gap1, gap2)
        min_gap = min(min_gap, gap_axis)
        
    return min_gap

def _compute_outer_side(positions, angles):
    """
    Computes the minimum side length s of a regular hexagon (center 0, angle 0)
    that contains all vertices of the hexagons defined by positions and angles.
    """
    max_proj = 0
    # Normals for outer hexagon (angle 0)
    # Vertices at 0, 60, ...
    # Edges are perpendicular to 30, 90, ...
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    for i in range(len(positions)):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        for n in normals:
            projs = verts.dot(n)
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
                
    # max_proj is the maximum projection onto any normal.
    # This must be <= apothem of outer hexagon.
    # apothem = s * sqrt(3) / 2
    # s = max_proj * 2 / sqrt(3)
    s = max_proj * 2.0 / np.sqrt(3.0)
    return s

def hexagon_packing_12():
    start_time = time.time()
    timeout = 590.0 # Leave 10s margin
    
    n = 12
    # Initial positions: Honeycomb grid
    # Side 1 hexagons.
    # Horizontal distance between centers: 1.5
    # Vertical distance: sqrt(3)
    
    # Let's generate a few random-ish initial configurations
    best_cost = np.inf
    best_x = None
    
    # Configuration 1: Standard honeycomb
    # 1 in center, 6 around, 5 more?
    # Let's just use a grid of 12 and let the optimizer move them.
    # Grid:
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # This is 4x3.
    
    # Better initial guess:
    # Center (0,0)
    # 6 neighbors
    # 5 more forming a ring?
    
    # Let's try a few random starts within a box
    rng = np.random.default_rng(42)
    
    def objective(x):
        # x: [x1, y1, a1, x2, y2, a2, ...]
        positions = x[0::3].reshape(-1, 2)
        angles = x[1::3]
        
        # Compute s
        s = _compute_outer_side(positions, angles)
        
        # Compute overlaps
        penalty = 0.0
        verts = []
        for i in range(n):
            verts.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
            
        for i in range(n):
            for j in range(i + 1, n):
                gap = _sat_gap(verts[i], verts[j])
                if gap < 0:
                    penalty += (-gap) ** 2
        
        # We want to minimize s + penalty
        # Scale penalty to be comparable to s
        return s + 10.0 * penalty

    # Run multiple restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Generate initial positions
        if r == 0:
            # Honeycomb-like
            pos = []
            # Center
            pos.append([0, 0])
            # Ring 1
            for k in range(6):
                angle = k * np.pi / 3
                pos.append([1.5 * np.cos(angle), 1.5 * np.sin(angle) * 2 / np.sqrt(3) * np.sqrt(3)]) # Wait, hex grid coords
            # Correct hex grid spacing:
            # dx = 1.5, dy = sqrt(3)
            # Let's just place them in a loose grid
            pos = []
            for i in range(4):
                for j in range(3):
                    x = i * 1.5 - 2.25
                    y = j * np.sqrt(3) - np.sqrt(3)
                    if j % 2 == 1:
                        x += 0.75
                    pos.append([x, y])
            pos = np.array(pos)
            angles = np.zeros(n)
        else:
            # Random positions in a circle
            pos = rng.uniform(-4, 4, (n, 2))
            angles = rng.uniform(0, 360, n)
            
        x0 = np.concatenate([pos.ravel(), angles])
        
        # Optimize
        try:
            res = opt.minimize(objective, x0, method='Powell', options={'maxiter': 200, 'disp': False, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
                [-1.25, 2.17, 0], [1.25, 2.17, 0],
                [-1.25, -2.17, 0], [1.25, -2.17, 0],
                [-3.75, 2.17, 0], [3.75, 2.17, 0],
                [-3.75, -2.17, 0], [3.75, -2.17, 0],
                [0, -4, 0]
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, 8.0

    # Extract best result
    positions = best_x[0::3].reshape(-1, 2)
    angles = best_x[1::3]
    
    # Refine with L-BFGS-B for better precision if time permits
    if time.time() - start_time < timeout - 20:
        try:
            res2 = opt.minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 100, 'disp': False})
            if res2.fun < best_cost:
                best_x = res2.x
                positions = best_x[0::3].reshape(-1, 2)
                angles = best_x[1::3]
        except Exception:
            pass

    # Final check and calculation of s
    s = _compute_outer_side(positions, angles)
    
    # Construct inner data
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, s

if __name__ == "__main__":
    hexagon_packing_12()
