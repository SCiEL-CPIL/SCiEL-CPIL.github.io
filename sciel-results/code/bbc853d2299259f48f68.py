import numpy as np
import time
import random

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    This provides a decent starting point.
    """
    pts = []
    # Start with a vertex or center? Center is safer.
    # Let's start with a random point, then add others greedily.
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        # Sample many candidates
        for _ in range(200):
            cand = np.array(_random_point_in_triangle(rng))
            # Calculate min distance to existing points
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses a "Rubber Band" / Spring Network model with local perturbations.
    The idea is that the optimal configuration has a quasi-lattice structure.
    We start with a regular grid, then optimize the positions by minimizing a potential energy
    that encourages points to stay near their grid positions (springs) while maximizing the minimum area.
    This reduces the effective search dimension and leverages the structural regularity.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Define the "ideal" grid structure.
    # For 11 points, a 3x4 grid doesn't fit perfectly, but we can approximate.
    # Let's use a 3x3 grid (9 points) plus 2 extra points, or a 4x3 grid (12 points) minus 1.
    # Actually, for Heilbronn, the optimal for N=11 is known to be close to a triangular lattice.
    # Let's try a few different initial topologies.
    
    topologies = []
    
    # Topology 1: 3x3 Grid + 2 points
    # Grid points in barycentric coordinates
    # Row 0 (y=0): 4 points? No, let's stick to a triangular lattice.
    # Triangular lattice points:
    # Row 0: (0,0), (1,0) -> 2 points? No, spacing.
    # Let's generate points on a triangular grid.
    
    def make_triangular_grid(rows, cols, inset=0.1):
        """
        Generate points on a triangular grid inside the triangle.
        rows: number of rows parallel to base.
        cols: number of points in the bottom row.
        """
        pts = []
        # Inset from edges to ensure they are strictly inside or on boundary
        # Barycentric coords (u, v) where P = u*B + v*C, A=(0,0), B=(1,0), C=(0.5, S3H)
        # Actually, let's just generate Cartesian coords.
        
        # Height of triangle
        h = S3H
        # Width of triangle
        w = 1.0
        
        # We want to place points in a triangular lattice.
        # Let's define the lattice by its step size.
        # If we have 'rows' rows, the vertical step is h/rows.
        # If we have 'cols' points in the first row, the horizontal step is w/cols.
        
        # For a triangular lattice, the horizontal offset of each row alternates.
        
        # Let's try a specific configuration for N=11.
        # A common good configuration is close to a 3x4 grid distorted.
        
        # Let's just use a simple grid for now and let the optimization fix it.
        # Grid: 3 rows, 4 cols.
        # Row 0 (bottom): 4 points
        # Row 1 (middle): 3 points
        # Row 2 (top): 1 point? Total 8. Not enough.
        
        # Let's try 4 rows.
        # Row 0: 4 points
        # Row 1: 3 points
        # Row 2: 2 points
        # Row 3: 1 point
        # Total: 10. Close.
        
        # Or:
        # Row 0: 4
        # Row 1: 4
        # Row 2: 3
        # Total: 11.
        
        # Let's generate a 3-row grid with 4, 4, 3 points.
        # Y coordinates: 0, h/2, h
        # But we need to inset from the boundary.
        
        # Let's use barycentric coordinates for easier inset.
        # A=(0,0), B=(1,0), C=(0.5, S3H)
        # Point = u*B + v*C
        # u, v >= 0, u+v <= 1
        
        # Row 0 (v=0): u = 0.1, 0.3, 0.6, 0.9 (4 points)
        # Row 1 (v=0.33): u = 0.1, 0.3, 0.6, 0.9 (4 points)
        # Row 2 (v=0.66): u = 0.2, 0.5, 0.8 (3 points)
        
        # Let's refine these positions.
        
        # Topology 1: 4-4-3
        t1 = []
        # Row 0
        for u in [0.1, 0.3, 0.6, 0.9]:
            t1.append([u + 0.5*0, 0]) # v=0
        # Row 1
        v1 = 1/3
        for u in [0.1, 0.3, 0.6, 0.9]:
            t1.append([u + 0.5*v1, S3H*v1])
        # Row 2
        v2 = 2/3
        for u in [0.2, 0.5, 0.8]:
            t2 = [u + 0.5*v2, S3H*v2]
            t1.append(t2)
            
        topologies.append(np.array(t1))
        
        # Topology 2: Random
        topologies.append(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
        # Topology 3: Greedy
        topologies.append(_greedy_init(rng))
        
        # Topology 4: 3-4-4 (inverted)
        t4 = []
        v0 = 2/3
        for u in [0.2, 0.5, 0.8]:
            t4.append([u + 0.5*v0, S3H*v0])
        v1 = 1/3
        for u in [0.1, 0.3, 0.6, 0.9]:
            t4.append([u + 0.5*v1, S3H*v1])
        v2 = 0
        for u in [0.1, 0.3, 0.6, 0.9]:
            t4.append([u + 0.5*v2, S3H*v2])
        topologies.append(np.array(t4))

    # For each topology, perform a local search using the "Rubber Band" idea.
    # The "Rubber Band" is a spring force pulling points back to their initial position.
    # We want to maximize min_area.
    # This is equivalent to minimizing -min_area.
    # Since min_area is non-smooth, we can't use gradient descent directly.
    # Instead, we use a local search:
    # 1. Start with initial points.
    # 2. Identify the "critical" triangle (the one with min area).
    # 3. Move the points of this triangle slightly to increase its area.
    # 4. Project back to the triangle.
    # 5. Repeat.
    
    # To make it more efficient, we can use a "Repulsion" model.
    # Force on point i is sum of repulsive forces from other points, weighted by the area of the triangle they form.
    # This is getting complicated. Let's stick to a simpler local search.
    
    # Local Search Strategy:
    # 1. Compute min area and the indices of the critical triangle.
    # 2. Generate random perturbations for the points in the critical triangle.
    # 3. If the new min area is larger, accept the move.
    # 4. If not, try a smaller perturbation.
    
    for s, init_pts in enumerate(topologies):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        pts = init_pts.copy()
        current_score = _min_triangle_area(pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            
        # Local Search
        step_size = 0.05
        no_improve_count = 0
        
        while time.time() - start_time < DEADLINE - 5:
            if no_improve_count > 100:
                break
                
            # Find critical triangle
            p_i = pts[_i_idx]
            p_j = pts[_j_idx]
            p_k = pts[_k_idx]
            
            dx1 = p_j - p_i
            dx2 = p_k - p_i
            cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            min_area = np.min(areas)
            min_idx = np.argmin(areas)
            
            # Get the indices of the critical triangle
            i, j, k = _triplets[min_idx]
            
            # Perturb the points i, j, k
            # We want to increase the area of this triangle.
            # The area is 0.5 * |cross(P_j - P_i, P_k - P_i)|.
            # To increase this, we can move P_j and P_k in the direction perpendicular to P_i-P_j and P_i-P_k respectively.
            
            # Simple approach: random perturbation
            new_pts = pts.copy()
            
            # Perturb i, j, k
            for idx in [i, j, k]:
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                new_pts[idx][0] += dx
                new_pts[idx][1] += dy
            
            new_pts = _project_to_triangle(new_pts)
            
            new_score = _min_triangle_area(new_pts)
            
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            else:
                no_improve_count += 1
                # Reduce step size if we are stuck
                if no_improve_count % 10 == 0:
                    step_size *= 0.9
                    if step_size < 1e-5:
                        break
                        
    # Final check
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
