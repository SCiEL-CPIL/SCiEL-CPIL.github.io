"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import Delaunay


def construct():
    # We want to pack N=21 circles in a rectangle of perimeter 4.
    # Let the rectangle have width W and height H.
    # Perimeter constraint: 2*(W + H) = 4  =>  W + H = 2.
    # We can choose W and H. A square is W=H=1.
    # For a square of side 1, the grid of 5x5 points has spacing 0.2.
    # Max radius in a 5x5 grid (with margins) is 0.1 - epsilon.
    # Sum of radii = 21 * 0.1 = 2.1 (approx).
    
    # However, we can do better by optimizing the rectangle aspect ratio
    # and the positions/radii.
    
    # Strategy:
    # 1. Choose an aspect ratio. Let's try a few or optimize it.
    #    Actually, for a fixed number of circles, a square often works well for grids,
    #    but for irregular packings, a different aspect ratio might allow larger radii.
    #    Let's start with a square (W=1, H=1) and see if we can improve via LP.
    
    # 2. Define initial centers.
    #    Use a grid layout as a starting point for the Delaunay triangulation.
    #    5x5 grid = 25 points. We need 21.
    #    We can remove 4 corner points or just take the first 21.
    #    Removing corners might allow the edge circles to grow larger?
    #    Actually, if we remove corners, the bounding box might shrink or we can expand radii.
    #    But the bounding box is fixed to perimeter 4.
    #    Let's use a 5x5 grid centered in the unit square.
    
    # Let's try to optimize the radii given a fixed set of centers.
    # The centers are fixed initially from a grid.
    # Variables: r_i for i=0..20.
    # Constraints:
    #   1. r_i >= 0
    #   2. r_i + r_j <= d_ij for all pairs (i,j) that are "close" (neighbors in Delaunay).
    #      Actually, for disjoint interiors, r_i + r_j <= d_ij for ALL pairs.
    #      But checking all pairs is O(N^2) = 441 constraints, which is fine.
    #   3. r_i <= x_i (distance to left wall)
    #      r_i <= W - x_i (distance to right wall)
    #      r_i <= y_i (distance to bottom wall)
    #      r_i <= H - y_i (distance to top wall)
    # Objective: Maximize sum(r_i).
    
    # Since LP maximizes sum, and constraints are linear, this will find the optimal radii
    # for the fixed centers.
    
    # But the centers themselves can be moved!
    # Moving centers is non-linear (d_ij depends on positions).
    # The directive suggests: "Represent the packing via a Delaunay triangulation of circle centers
    # and solve for radii using a linear programming formulation... decouples the combinatorial
    # structure from the continuous optimization".
    
    # This implies we should fix the combinatorial structure (Delaunay graph) and optimize
    # radii. But what about positions?
    # If we fix positions, we get a local optimum for radii.
    # To improve, we could also optimize positions.
    # However, the directive says "solve for radii using LP". It doesn't explicitly say to optimize positions.
    # But if positions are fixed to a grid, we might not be optimal.
    
    # Let's try a hybrid:
    # 1. Start with a good grid.
    # 2. Optimize radii via LP.
    # 3. Maybe iterate: update centers?
    #    If we move centers, the Delaunay triangulation might change.
    #    This becomes a complex optimization.
    
    # Alternative: Use the LP to find the best radii for a specific layout, and then
    # try a few different layouts (e.g., different aspect ratios, or perturbed grids).
    
    # Let's first implement the LP for a fixed layout.
    
    def solve_radii_lp(centers, W, H):
        n = len(centers)
        # Variables: r_0, ..., r_{n-1}
        # Maximize sum(r)  =>  Minimize -sum(r)
        c = -np.ones(n)
        
        # Constraints:
        # A_ub @ r <= b_ub
        
        A_rows = []
        b_rows = []
        
        # Pairwise distance constraints: r_i + r_j <= d_ij
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                # r_i + r_j <= d
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_rows.append(row)
                b_rows.append(d)
        
        # Wall constraints:
        # r_i <= x_i
        # r_i <= W - x_i
        # r_i <= y_i
        # r_i <= H - y_i
        
        for i in range(n):
            x, y = centers[i]
            
            # Left wall: r_i <= x
            row = np.zeros(n)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(x)
            
            # Right wall: r_i <= W - x
            row = np.zeros(n)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(W - x)
            
            # Bottom wall: r_i <= y
            row = np.zeros(n)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(y)
            
            # Top wall: r_i <= H - y
            row = np.zeros(n)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(H - y)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: r_i >= 0
        bounds = [(0, None)] * n
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x
        else:
            # If LP fails, return small radii
            return np.zeros(n)

    # Let's try a 5x5 grid in a 1x1 square.
    # Remove 4 corners to get 21 circles?
    # Or just take 21 from a 5x5 grid.
    # If we take 21 from a 5x5 grid, we have 4 empty spots.
    # The empty spots allow the neighboring circles to potentially expand?
    # No, the walls and other circles constrain them.
    # But if the corners are empty, the corner circles (which are now edge circles)
    # might be able to grow towards the corner?
    # Wait, if the corner point is removed, the circle that was at the corner is gone.
    # The circles adjacent to the corner are now closer to the corner walls.
    # They might be able to grow into the corner space.
    
    # Let's define a 5x5 grid of points in [0,1]x[0,1].
    # Points: (i/4, j/4) for i,j in 0..4.
    # This puts circles on the boundary.
    # Radii will be limited by walls.
    # Center at (0,0) has r <= 0. This is bad.
    # We should inset the grid.
    
    # Let's use centers at (i+0.5)/5, (j+0.5)/5 for i,j in 0..4.
    # This is a 5x5 grid of 25 points.
    # Spacing is 0.2.
    # Distance to wall is 0.1.
    # Initial max radius is 0.1.
    # Sum = 25 * 0.1 = 2.5.
    # We only have 21 circles.
    # If we remove 4 circles, the sum of the remaining 21 might increase
    # because the constraints from the removed circles are gone.
    # Which 4 to remove?
    # Removing corners seems good because corners have the most "wasted" space?
    # Or removing interior points?
    # If we remove interior points, the neighbors can expand.
    # If we remove corner points, the edge neighbors can expand into the corner.
    
    # Let's try removing the 4 corners of the 5x5 grid.
    # Remaining: 21 circles.
    
    centers = []
    for i in range(5):
        for j in range(5):
            # Skip corners
            if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                continue
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            centers.append((x, y))
            
    centers = np.array(centers)
    W = 1.0
    H = 1.0
    
    radii = solve_radii_lp(centers, W, H)
    
    # Check if this is valid.
    # The LP ensures r_i + r_j <= d_ij and r_i <= wall_dist.
    # So interiors are disjoint and inside the rectangle.
    
    # Construct the output array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [centers[i][0], centers[i][1], radii[i]]
        
    return result
# EVOLVE-BLOCK-END
