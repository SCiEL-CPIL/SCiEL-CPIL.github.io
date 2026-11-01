"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = 150.0  # safety margin before 120s limit? Wait, prompt says 120s limit, 150s kill.
    # The prompt says "evaluator KILLS the program after 150s wall-clock".
    # "Keep an EXPLICIT deadline guard: track elapsed time and return the best result found with a safety margin before the limit."
    # Let's target 115s to be safe.
    deadline = 115.0
    
    # Strategy: 
    # 1. Try a grid-based layout with optimization of rectangle dimensions (W, H) such that 2(W+H)=4 => H = 2-W.
    #    Maximize sum of radii.
    #    For a fixed grid layout (rows, cols), the max radius is limited by the minimum of:
    #    - spacing in x: W/cols
    #    - spacing in y: H/rows
    #    - distance to walls: W/cols, H/rows
    #    Actually, for a grid of R rows and C cols, the circles are at ( (i+0.5)*W/C, (j+0.5)*H/R ).
    #    The max radius is min( W/(2*C), H/(2*R) ).
    #    Sum of radii = R*C * min( W/(2*C), H/(2*R) ).
    #    Let r = min( W/(2*C), H/(2*R) ).
    #    We want to maximize N * r.
    #    Constraints: W <= 2*C*r, H <= 2*R*r, W+H=2.
    #    To maximize r, we should set W and H such that the constraints are tight if possible, or balanced.
    #    If W/(2*C) == H/(2*R), then W/H = C/R.
    #    W + H = 2 => H(1 + C/R) = 2 => H = 2R / (R+C).
    #    Then r = H / (2R) = (2R / (R+C)) / (2R) = 1 / (R+C).
    #    So max sum of radii = N / (R+C).
    #    We need to choose integer R, C such that R*C >= N.
    #    To maximize N/(R+C), we want to minimize R+C subject to R*C >= N.
    #    
    # Let's evaluate all valid (R, C) pairs.
    # N = 21.
    # Possible (R, C):
    # 1x21: sum=22, r=1/22, val=21/22 ~ 0.954
    # 2x11: sum=13, r=1/13, val=21/13 ~ 1.615
    # 3x7: sum=10, r=1/10, val=21/10 = 2.1
    # 4x6: sum=10, r=1/10, val=21/10 = 2.1 (since 4*6=24>=21)
    # 5x5: sum=10, r=1/10, val=21/10 = 2.1 (since 5*5=25>=21)
    # Wait, R*C must be >= N.
    # For 3x7, R*C=21. Sum R+C=10. Val = 2.1.
    # For 4x6, R*C=24. We only place 21 circles.
    #    If we use a 4x6 grid, we have 24 spots. We fill 21.
    #    The radius is still limited by the grid spacing.
    #    r = min(W/8, H/8) if W=H? No.
    #    If we optimize W, H for the 4x6 grid:
    #    r = 1/(R+C) = 1/10.
    #    Sum = 21 * 0.1 = 2.1.
    #    Is it possible to do better by not using a full grid?
    #    Maybe a hexagonal packing?
    #    Hexagonal packing has higher density.
    #    
    # Let's check the theoretical upper bound.
    # Area of rectangle = W*H. Max area when W=H=1 is 1.
    # Sum of areas of circles = pi * sum(r_i^2).
    # This doesn't directly bound sum(r_i).
    # 
    # Let's try to improve the grid layout.
    # In a 3x7 grid, we have 21 circles.
    # r = 0.1.
    # Can we pack them tighter?
    # Hexagonal packing in a rectangle.
    # Rows alternate.
    # Let's try a specific hexagonal arrangement for 21 circles.
    # 7 circles in row 1, 7 in row 2, 7 in row 3?
    # If rows are staggered, the vertical distance is sqrt(3)*r.
    # Horizontal distance is 2r.
    # Let's model this.
    # Let r be the radius.
    # Row 1 y=r. x = r + 2r*i for i=0..6. Width = 14r.
    # Row 2 y=r + sqrt(3)r. x = 2r + 2r*i for i=0..6. Width = 14r.
    # Row 3 y=r + 2*sqrt(3)r. x = r + 2r*i for i=0..6. Width = 14r.
    # Total Width W = 14r.
    # Total Height H = r + 2*sqrt(3)r + r = 2r(1+sqrt(3)).
    # Perimeter constraint: 2(W+H) = 4 => W+H = 2.
    # 14r + 2r(1+sqrt(3)) = 2
    # r(14 + 2 + 2sqrt(3)) = 2
    # r(16 + 2sqrt(3)) = 2
    # r = 2 / (16 + 2sqrt(3)) = 1 / (8 + sqrt(3))
    # sqrt(3) ~ 1.732
    # r ~ 1 / 9.732 ~ 0.10275
    # Sum of radii = 21 * 0.10275 ~ 2.158
    # This is better than 2.1.
    
    # Can we do better with different row counts?
    # 6 circles in row 1, 7 in row 2, 8 in row 3? (Total 21)
    # Or 7, 6, 8?
    # Let's generalize the hexagonal packing search.
    # We can use a local optimization to maximize sum of radii for a given topology (positions relative to each other).
    # Or, we can just use a numerical optimizer to maximize the minimum distance between circles and walls, given a fixed set of centers?
    # No, we can move the centers.
    
    # Let's try a more general approach:
    # 1. Generate a candidate layout (e.g., hexagonal grid).
    # 2. Use scipy.optimize.minimize to maximize the minimum clearance (radius) for that layout, adjusting W and H and positions?
    #    Actually, if we fix the relative positions (scaled by r), we just need to find the optimal W, H.
    #    For a fixed "shape" of the packing (defined by normalized coordinates x_i, y_i where max extent is 1),
    #    the max radius r is determined by the aspect ratio of the rectangle.
    
    # Let's define a function that takes a list of (x, y) coordinates (normalized such that they fit in [0,1]x[0,1] roughly)
    # and finds the optimal W, H and scale factor to maximize sum of radii.
    # Actually, it's easier to:
    # 1. Propose a set of relative positions (x_i, y_i).
    # 2. Let the circles have radius r.
    # 3. The bounding box of the centers is [min_x, max_x] x [min_y, max_y].
    # 4. The required rectangle width W >= max_x - min_x + 2r.
    # 5. The required rectangle height H >= max_y - min_y + 2r.
    # 6. We want to maximize N*r subject to W+H <= 2.
    #    This implies (max_x - min_x + 2r) + (max_y - min_y + 2r) <= 2.
    #    Let Dx = max_x - min_x, Dy = max_y - min_y.
    #    Dx + Dy + 4r <= 2 => r <= (2 - Dx - Dy) / 4.
    #    Also, we must ensure circles don't overlap.
    #    Distance between any two centers >= 2r.
    #    So r <= min_{i<j} (dist(i,j) / 2).
    #    
    # So for a fixed set of relative coordinates (x_i, y_i), the max r is:
    # r = min( (2 - Dx - Dy)/4, min_{i<j} dist(i,j)/2 )
    # And we want to maximize this r by scaling the coordinates?
    # Wait, if we scale coordinates by s, Dx and Dy scale by s, and distances scale by s.
    # r(s) = min( (2 - s*Dx0 - s*Dy0)/4, s * min_dist0 / 2 )
    # We want to maximize r(s).
    # The first term decreases with s, the second increases with s.
    # The optimum is where they are equal (if possible).
    # (2 - s(Dx0+Dy0))/4 = s * min_dist0 / 2
    # 2 - s(Dx0+Dy0) = 2 * s * min_dist0
    # 2 = s ( Dx0 + Dy0 + 2*min_dist0 )
    # s = 2 / ( Dx0 + Dy0 + 2*min_dist0 )
    # Then r = s * min_dist0 / 2 = min_dist0 / ( Dx0 + Dy0 + 2*min_dist0 )
    # Sum of radii = N * r.
    
    # So the problem reduces to finding a set of N points (x_i, y_i) that maximizes:
    # N * min_dist / ( Dx + Dy + 2*min_dist )
    # where min_dist is the minimum pairwise distance, Dx is x-span, Dy is y-span.
    
    # This is a "max-min" problem which can be solved with local search or simulated annealing.
    # Given the time limit, we can try a few good initial configurations (hexagonal grids) and refine them with a simple local search or just use the analytical formula for the hexagonal grid.
    
    # Let's implement the analytical calculation for a few hexagonal grid variants.
    
    def calc_score(coords):
        coords = np.array(coords)
        # Normalize coords so that min x is 0, min y is 0?
        # The formula depends on spans and min dist.
        # Shifting doesn't change spans or distances.
        # Scaling is handled by the formula.
        # So we just need the relative geometry.
        
        # Calculate min pairwise distance
        n = len(coords)
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                if d < min_dist:
                    min_dist = d
        
        Dx = np.ptp(coords[:, 0])
        Dy = np.ptp(coords[:, 1])
        
        denom = Dx + Dy + 2 * min_dist
        if denom == 0:
            return 0
        r = min_dist / denom
        return N * r

    best_score = 0
    best_coords = None

    # Generate hexagonal grids
    # Variant 1: 7-7-7
    # Row 0: 7 circles
    # Row 1: 7 circles (shifted by 1)
    # Row 2: 7 circles
    # Let's define coordinates.
    # dx = 2, dy = sqrt(3)
    # Row 0 y=0, x=0, 2, 4, 6, 8, 10, 12
    # Row 1 y=sqrt(3), x=1, 3, 5, 7, 9, 11, 13
    # Row 2 y=2*sqrt(3), x=0, 2, 4, 6, 8, 10, 12
    
    coords_777 = []
    for i in range(7):
        coords_777.append((2*i, 0))
    for i in range(7):
        coords_777.append((2*i+1, np.sqrt(3)))
    for i in range(7):
        coords_777.append((2*i, 2*np.sqrt(3)))
    
    s = calc_score(coords_777)
    if s > best_score:
        best_score = s
        best_coords = coords_777

    # Variant 2: 6-7-8
    # Row 0: 6 circles
    # Row 1: 7 circles
    # Row 2: 8 circles
    # This might be tighter.
    # Let's try to construct it.
    # It's hard to guess the exact optimal hexagonal arrangement for 21.
    # Let's try a generic local search on the coordinates.
    
    # Initialize with 7-7-7
    coords = np.array(best_coords)
    
    # Local search: move points to increase min_dist or decrease span.
    # Objective: maximize min_dist / (Dx + Dy + 2*min_dist)
    # Let f = min_dist / (Dx + Dy + 2*min_dist)
    
    # We can use a simple hill climbing.
    # Time budget: 10 seconds for this search.
    search_deadline = start_time + 10
    
    def get_score_and_grads(coords):
        n = len(coords)
        # Calculate distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                dists[i, j] = d
                dists[j, i] = d
        
        min_dist = np.min(dists[dists > 0])
        
        # Identify pairs at min_dist
        # This is a bit expensive, but N=21 is small.
        
        Dx = np.ptp(coords[:, 0])
        Dy = np.ptp(coords[:, 1])
        
        denom = Dx + Dy + 2 * min_dist
        score = min_dist / denom
        
        return score, min_dist, Dx, Dy, dists

    # Simple local search
    # Move each point slightly in random directions.
    # Accept if score improves.
    
    current_coords = best_coords.copy()
    current_score = best_score
    
    np.random.seed(42)
    
    while time.time() < search_deadline:
        improved = False
        # Try moving each point
        for i in range(N):
            # Try 4 directions + diagonal
            for dx, dy in [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01), (0.01, 0.01), (-0.01, -0.01)]:
                new_coords = current_coords.copy()
                new_coords[i, 0] += dx
                new_coords[i, 1] += dy
                
                new_score = calc_score(new_coords)
                if new_score > current_score + 1e-6:
                    current_coords = new_coords
                    current_score = new_score
                    best_score = current_score
                    best_coords = current_coords.copy()
                    improved = True
                    break
        if not improved:
            # Try larger steps or restart?
            # For now, break if no improvement found in a full pass.
            break

    # Convert best_coords to the final circle layout
    # We need to calculate the actual r, W, H.
    # From the formula:
    # s = 2 / ( Dx0 + Dy0 + 2*min_dist0 )
    # The actual coordinates are s * best_coords (shifted so min is 0? No, the formula used raw coords).
    # Wait, the formula r = min_dist / (Dx + Dy + 2*min_dist) assumes the coordinates are the "unscaled" ones.
    # The actual radius is r.
    # The actual center positions are:
    # x_i = (best_coords[i, 0] - min_x) + r
    # y_i = (best_coords[i, 1] - min_y) + r
    # W = Dx + 2r
    # H = Dy + 2r
    
    min_x = np.min(best_coords[:, 0])
    min_y = np.min(best_coords[:, 1])
    
    # Recalculate min_dist and spans for the best_coords
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_coords[i] - best_coords[j])
            if d < min_dist:
                min_dist = d
    
    Dx = np.ptp(best_coords[:, 0])
    Dy = np.ptp(best_coords[:, 1])
    
    r = min_dist / (Dx + Dy + 2 * min_dist)
    
    W = Dx + 2 * r
    H = Dy + 2 * r
    
    # Check perimeter
    # 2(W+H) should be <= 4.
    # W+H = Dx + Dy + 4r
    # r = min_dist / (Dx + Dy + 2*min_dist)
    # Let D = Dx + Dy.
    # W+H = D + 4 * min_dist / (D + 2*min_dist)
    # We want 2(W+H) <= 4 => W+H <= 2.
    # D + 4m / (D+2m) <= 2 ?
    # D(D+2m) + 4m <= 2(D+2m)
    # D^2 + 2Dm + 4m <= 2D + 4m
    # D^2 + 2Dm <= 2D
    # D(D + 2m - 2) <= 0
    # Since D>0, we need D + 2m <= 2.
    # Is this always true?
    # The derivation s = 2 / (D + 2m) came from setting W+H=2.
    # So yes, by construction W+H=2.
    
    # Construct the array
    circles = np.zeros((N, 3))
    for i in range(N):
        circles[i, 0] = best_coords[i, 0] - min_x + r
        circles[i, 1] = best_coords[i, 1] - min_y + r
        circles[i, 2] = r

    return circles

# EVOLVE-BLOCK-END
