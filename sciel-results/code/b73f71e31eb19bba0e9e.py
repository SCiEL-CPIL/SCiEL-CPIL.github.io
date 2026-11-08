"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import time
import numpy as np
from scipy.spatial import Voronoi

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, Delaunay
import time

N = 21

def _voronoi_inradii(centers, width, height):
    """
    For each center, compute the largest circle radius such that the circle
    stays inside the rectangle [0,width]x[0,height] and does not overlap
    any other circle (using Voronoi cell boundaries as proxy).
    
    Actually, the correct approach: for each circle i, its radius is limited by:
    1. Distance to the nearest boundary of the rectangle
    2. For each other circle j, the distance between centers minus r_j (but this is coupled)
    
    Better approach using Voronoi: The Voronoi cell of center i is the set of points
    closer to i than to any other center. The largest circle centered at i that fits
    in its Voronoi cell AND in the rectangle has radius = min(
        min distance from center i to each edge of the Voronoi cell,
        min distance from center i to rectangle boundaries
    )
    
    But we need the Voronoi cell clipped to the rectangle. Let's compute it properly.
    """
    n = len(centers)
    radii = np.zeros(n)
    
    # Add boundary points far away to create a bounded Voronoi diagram
    # Or better: compute Voronoi on the centers, then for each cell, find the
    # inradius considering the rectangle boundaries
    
    # Method: For each circle i, r_i = min over all constraints:
    # - distance to left boundary: centers[i,0]
    # - distance to right boundary: width - centers[i,0]
    # - distance to bottom boundary: centers[i,1]
    # - distance to top boundary: height - centers[i,1]
    # - For each other circle j: (dist(i,j) - r_j) ... but this is coupled
    
    # Using Voronoi: the Voronoi cell of i is bounded by perpendicular bisectors
    # with neighbors. The inradius of the Voronoi cell (clipped to rectangle) is what we want.
    
    # Simpler iterative approach: 
    # r_i = min(
    #     centers[i,0], width - centers[i,0],
    #     centers[i,1], height - centers[i,1],
    #     min over j != i of (||c_i - c_j|| - r_j)
    # )
    # But this is implicit. Let's use a different strategy.
    
    # For optimization purposes, we can define:
    # For each pair (i,j), the constraint is r_i + r_j <= ||c_i - c_j||
    # And r_i <= min(c_i[0], width-c_i[0], c_i[1], height-c_i[1])
    
    # The "Voronoi-derived" radius for circle i, given all other radii, is:
    # r_i = min(boundary_dist_i, min_j (dist_ij - r_j))
    
    # For the objective, we want to maximize sum(r_i). 
    # A good approach: use the fact that at optimum, each circle is "tight" against 
    # either the boundary or another circle. 
    
    # For smooth optimization, let's define the radius as:
    # r_i = min( boundary_dist_i, min_j soft_min(dist_ij - r_j) )
    # But this is still coupled.
    
    # Alternative: Use the Voronoi diagram directly.
    # Compute Voronoi of the centers. For each cell, compute the inradius of the
    # intersection of the Voronoi cell with the rectangle.
    
    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback: uniform radii based on boundary distances
        for i in range(n):
            radii[i] = min(centers[i,0], width-centers[i,0], centers[i,1], height-centers[i,1])
        return radii
    
    # For each point, find the inradius of its Voronoi cell clipped to [0,width]x[0,height]
    # The Voronoi cell is a convex polygon (possibly unbounded). We clip it to the rectangle.
    # The inradius is the maximum radius of a circle centered at the point that fits in the cell.
    
    # scipy.spatial.Voronoi gives regions as indices of vertices.
    # We need to handle unbounded regions by clipping.
    
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0:
            # Degenerate case
            radii[i] = min(centers[i,0], width-centers[i,0], centers[i,1], height-centers[i,1])
            continue
        
        # Get the vertices of the Voronoi cell
        # For unbounded regions, region contains -1 for vertices at infinity
        # We'll clip the cell to the rectangle
        
        # Get finite vertices
        finite_vertices = []
        for v in region:
            if v != -1:
                finite_vertices.append(vor.vertices[v])
        
        # The cell is the intersection of half-planes. For a simpler approach,
        # we can compute the distance from center i to each edge of the rectangle
        # and to each edge of the Voronoi cell.
        
        # Boundary distances
        bd = min(centers[i,0], width-centers[i,0], centers[i,1], height-centers[i,1])
        
        # For Voronoi edges: the distance from center i to each Voronoi edge
        # A Voronoi edge is the perpendicular bisector between center i and a neighbor.
        # The distance from center i to this bisector is half the distance to the neighbor.
        
        # So for each neighbor j of i in the Voronoi diagram, the constraint is:
        # r_i <= (||c_i - c_j|| - r_j) ... but again coupled.
        
        # For the inradius of the Voronoi cell (ignoring other radii for a moment),
        # the distance from center i to the Voronoi edge shared with j is ||c_i - c_j||/2.
        # So the inradius of the Voronoi cell is min over neighbors j of ||c_i - c_j||/2,
        # and also bounded by the rectangle.
        
        # But we need to account for the fact that other circles also have radii.
        # At the optimal packing, r_i + r_j = ||c_i - c_j|| for tight contacts.
        
        # For a smooth objective, let's use:
        # r_i = min(bd, min over neighbors j of (||c_i - c_j||/2))
        # This gives the inradius of the Voronoi cell clipped to the rectangle.
        # Then the actual circle with this radius will be disjoint from others IF
        # the Voronoi cells are disjoint (which they are) and each circle fits in its cell.
        
        # Wait: if r_i = inradius of Voronoi cell i, and r_j = inradius of Voronoi cell j,
        # then circle i fits in cell i and circle j fits in cell j. Since cells are disjoint,
        # the circles are disjoint. Yes! This works.
        
        # So: r_i = min( boundary_dist_i, min over Voronoi neighbors j of ||c_i - c_j||/2 )
        
        # Find Voronoi neighbors
        min_voronoi_dist = np.inf
        for j in range(n):
            if j == i:
                continue
            d = np.linalg.norm(centers[i] - centers[j])
            min_voronoi_dist = min(min_voronoi_dist, d / 2.0)
        
        radii[i] = min(bd, min_voronoi_dist)
    
    return radii


def _repulsion_forces(centers, width, height, radii):
    """
    Compute forces to push centers apart and maximize sum of radii.
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    for i in range(n):
        for j in range(i+1, n):
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.array([1e-8, 0])
                dist = 1e-8
            
            # The constraint is r_i + r_j <= dist
            # We want to maximize r_i + r_j, which means we want dist to be large,
            # but also bounded by the rectangle.
            
            # Force: push apart if they are too close (overlap in terms of max possible radii)
            # Max possible sum: r_i + r_j <= dist
            # Current radii: radii[i] + radii[j]
            # If radii[i] + radii[j] > dist * 0.99, push apart
            
            overlap = (radii[i] + radii[j]) - dist
            if overlap > 0:
                # Strong repulsion
                f = overlap * 10.0
            else:
                # Mild attraction to maintain contact (to maximize radii)
                f = -0.1 * (1.0 - dist / (radii[i] + radii[j] + 1e-6))
                f = max(f, -0.5)  # Cap attraction
            
            unit_d = d / dist
            forces[i] += f * unit_d
            forces[j] -= f * unit_d
    
    # Boundary forces: push centers away from boundaries if too close
    margin = 0.01
    for i in range(n):
        if centers[i,0] < margin:
            forces[i,0] += (margin - centers[i,0]) * 5.0
        if centers[i,0] > width - margin:
            forces[i,0] -= (centers[i,0] - (width - margin)) * 5.0
        if centers[i,1] < margin:
            forces[i,1] += (margin - centers[i,1]) * 5.0
        if centers[i,1] > height - margin:
            forces[i,1] -= (centers[i,1] - (height - margin)) * 5.0
    
    return forces


def _evaluate(centers, width, height):
    """Evaluate the sum of radii for a given configuration."""
    radii = _voronoi_inradii(centers, width, height)
    # Check validity: circles must be disjoint and inside rectangle
    for i in range(N):
        if radii[i] < 0:
            return 0.0
        if centers[i,0] - radii[i] < -1e-6 or centers[i,0] + radii[i] > width + 1e-6:
            return 0.0
        if centers[i,1] - radii[i] < -1e-6 or centers[i,1] + radii[i] > height + 1e-6:
            return 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < radii[i] + radii[j] - 1e-6:
                return 0.0
    return np.sum(radii)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_centers = None
    best_width = None
    best_height = None
    
    # Try different aspect ratios
    # Perimeter = 2*(w+h) = 4 => w+h = 2
    # w in (0, 2), h = 2 - w
    aspect_ratios = np.linspace(0.3, 1/0.3, 50)
    aspect_ratios = np.unique(np.round(aspect_ratios, 3))
    
    # Add some specific ratios
    extra_ratios = [1.0, 1.5, 2.0, 0.5, 0.667, 1.333, 0.75, 1.25, 0.4, 2.5, 3.0]
    aspect_ratios = np.unique(np.concatenate([aspect_ratios, extra_ratios]))
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
        
        # Initialize with grid
        side = int(np.ceil(np.sqrt(N * w / h)))
        if side < 3:
            side = 3
        if side > 7:
            side = 7
        
        # Create grid centers
        cols = side
        rows = int(np.ceil(N / cols))
        if rows < 1:
            rows = 1
        
        centers = []
        for i in range(cols):
            for j in range(rows):
                if len(centers) >= N:
                    break
                cx = (i + 0.5) * w / cols
                cy = (j + 0.5) * h / rows
                centers.append([cx, cy])
        
        centers = np.array(centers[:N])
        
        # Optimize with repulsion-based approach
        np.random.seed(42)
        
        best_local_score = _evaluate(centers, w, h)
        best_local_centers = centers.copy()
        
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            radii = _voronoi_inradii(centers, w, h)
            forces = _repulsion_forces(centers, w, h, radii)
            
            # Update centers
            centers = centers + lr * forces
            
            # Clamp to rectangle with small margin
            margin = 0.001
            centers[:,0] = np.clip(centers[:,0], margin, w - margin)
            centers[:,1] = np.clip(centers[:,1], margin, h - margin)
            
            # Evaluate
            score = _evaluate(centers, w, h)
            if score > best_local_score:
                best_local_score = score
                best_local_centers = centers.copy()
            
            # Adaptive learning rate
            if step % 100 == 99:
                lr *= 0.95
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_centers = best_local_centers.copy()
            best_width = w
            best_height = h
        
        # Also try with random perturbation
        if time.time() > deadline - 10.0:
            break
        
        centers_rand = best_local_centers + np.random.randn(N, 2) * 0.05
        centers_rand[:,0] = np.clip(centers_rand[:,0], 0.001, w - 0.001)
        centers_rand[:,1] = np.clip(centers_rand[:,1], 0.001, h - 0.001)
        
        best_local_score2 = _evaluate(centers_rand, w, h)
        best_local_centers2 = centers_rand.copy()
        
        lr = 0.005
        for step in range(1500):
            if time.time() > deadline - 5.0:
                break
            
            radii = _voronoi_inradii(centers_rand, w, h)
            forces = _repulsion_forces(centers_rand, w, h, radii)
            
            centers_rand = centers_rand + lr * forces
            centers_rand[:,0] = np.clip(centers_rand[:,0], 0.001, w - 0.001)
            centers_rand[:,1] = np.clip(centers_rand[:,1], 0.001, h - 0.001)
            
            score = _evaluate(centers_rand, w, h)
            if score > best_local_score2:
                best_local_score2 = score
                best_local_centers2 = centers_rand.copy()
            
            if step % 100 == 99:
                lr *= 0.95
        
        if best_local_score2 > best_score:
            best_score = best_local_score2
            best_centers = best_local_centers2.copy()
            best_width = w
            best_height = h
    
    # Fallback if no valid configuration found
    if best_centers is None:
        # Use simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Scale to fit in rectangle with perimeter 4
    # The centers are already in [0, best_width] x [0, best_height]
    # Perimeter = 2*(best_width + best_height) = 4 by construction
    # Just return [x, y, r] for each circle
    radii = _voronoi_inradii(best_centers, best_width, best_height)
    result = np.column_stack([best_centers, radii])
    
    # Final validity check
    for i in range(N):
        x, y, r = result[i]
        if x - r < -1e-5 or x + r > best_width + 1e-5:
            result[i,2] = min(x, best_width - x) - 1e-4
        if y - r < -1e-5 or y + r > best_height + 1e-5:
            result[i,2] = min(result[i,2], min(y, best_height - y) - 1e-4)
    
    # Ensure non-negative radii
    result[:,2] = np.maximum(result[:,2], 0.0)
    
    return result

# EVOLVE-BLOCK-END
