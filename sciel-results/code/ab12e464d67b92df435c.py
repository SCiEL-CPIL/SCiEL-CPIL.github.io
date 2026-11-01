import numpy as np

N = 50

def construct():
    """
    Constructs the centers of 50 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the minimum distance between circles and to the boundary.
    """
    
    # 1. Initialize with a hexagonal lattice
    # We want to fit N points. Let's estimate rows.
    # Hexagonal packing density is pi/sqrt(12) ~ 0.9069
    # For a square of side 1, area 1. 50 circles of radius r have area 50*pi*r^2.
    # Approx r ~ sqrt(1 / (50*pi / 0.9)) ~ 0.075.
    # Diameter d ~ 0.15.
    # Rows ~ 1 / (d * sqrt(3)/2) ~ 1 / (0.15 * 0.866) ~ 7.7. So 7 or 8 rows.
    
    def generate_hex_grid(n_points, side=1.0):
        # Try to fit into square [-0.5, 0.5]
        # Hex lattice spacing: horizontal dx, vertical dy = dx * sqrt(3)/2
        # Let's iterate on the number of rows to find a good starting config
        
        best_grid = None
        best_score = -1
        
        for rows in range(5, 10):
            # Estimate columns needed
            # Points per row varies (e.g., 8, 7, 8, 7...)
            # Total points = sum of points in each row
            
            # Let's just generate a large hex grid and crop/center
            # Or generate specific rows.
            
            # Simpler: Generate a hex grid with a specific spacing and count points inside.
            # We want to scale it so it fits.
            
            # Let's try to construct a specific pattern:
            # Row lengths for 50 points:
            # 8+7+8+7+8+7+5 = 50? No, 8+7+8+7+8+7+5 is 50.
            # Or 7+8+7+8+7+8+5 = 50.
            # Let's try to fit a grid with roughly equal row lengths.
            
            # Let's use a generic hex generator and adjust spacing.
            # Generate points in a hex grid with spacing 1.
            # Then scale to fit the square.
            
            # Generate a block of hex points
            # Let's say we have R rows.
            # Points in row i: C + (i % 2) or C - 1 ...
            
            # Let's try specific row counts that sum to N
            # For 50:
            # 8, 7, 8, 7, 8, 7, 5 (last row partial) -> 50
            # 7, 8, 7, 8, 7, 8, 5 -> 50
            # 8, 8, 8, 8, 8, 8, 2 -> 50 (less optimal)
            
            # Let's try to find a configuration that fits well.
            # We will generate a hex grid with a given number of rows and points per row.
            # Then we scale it to fit the square.
            
            # Actually, a more robust initialization is to just scatter points in a hex pattern
            # and let the optimizer fix the boundaries.
            
            pass

    # Better Initialization:
    # Use a hexagonal grid that is slightly compressed or expanded to fit the square.
    # Let's try to fit 7 or 8 rows.
    
    def create_hex_initialization():
        # Try different row configurations
        configs = [
            [8, 7, 8, 7, 8, 7, 5],
            [7, 8, 7, 8, 7, 8, 5],
            [8, 8, 8, 8, 8, 8, 2], # Less likely to be good
            [6, 7, 8, 8, 8, 7, 6], # 50
            [5, 8, 8, 8, 8, 8, 5], # 50
            [7, 7, 8, 8, 8, 7, 5], # 50
            [8, 7, 7, 8, 7, 8, 5], # 50
            [6, 8, 8, 8, 8, 7, 5], # 50
            [5, 7, 8, 8, 8, 7, 7], # 50
            [7, 8, 8, 8, 8, 7, 4], # 50
            [8, 8, 8, 8, 7, 7, 4], # 50
        ]
        
        best_points = None
        best_min_dist = -1
        
        for config in configs:
            if sum(config) != N:
                continue
                
            points = []
            # Calculate positions for this config
            # Let dx be horizontal spacing, dy vertical.
            # In hex, dy = dx * sqrt(3)/2
            # We need to fit in [-0.5, 0.5]
            
            # The width of the grid depends on the max row length and the offset.
            # The height depends on the number of rows.
            
            # Let's parameterize by dx.
            # Height = (len(config) - 1) * dy = (len(config) - 1) * dx * sqrt(3)/2
            # Width = (max_row_len - 1) * dx + offset (offset is dx/2 for staggered rows)
            
            # We want to maximize the minimum distance to boundary and between points.
            # For a regular hex grid, the distance between adjacent points is dx.
            # The distance to the boundary will be determined by the scaling.
            
            # Let's find the scaling factor s such that the grid fits in the square.
            # Let the grid be centered at (0,0).
            
            # Generate unit hex grid (dx=1)
            unit_points = []
            for i, count in enumerate(config):
                y = - (len(config) - 1) / 2.0 + i
                # Center the row
                # If row is staggered, shift x by 0.5
                # Standard hex: even rows shifted?
                # Let's say even indices (0, 2, 4) are not shifted, odd are shifted by 0.5
                # Or vice versa. Let's try both or just one.
                # Usually, to minimize width, we align the longest rows in the middle.
                
                x_start = - (count - 1) / 2.0
                if i % 2 == 1:
                    x_start += 0.5 # Shift odd rows to the right by 0.5 relative to even?
                    # Wait, if we shift, the center of mass might move.
                    # Let's just generate and then center the whole set.
                
                for j in range(count):
                    x = x_start + j
                    unit_points.append([x, y])
                    
            unit_points = np.array(unit_points)
            
            # Center the unit points
            center = unit_points.mean(axis=0)
            unit_points -= center
            
            # Find bounding box
            min_x, min_y = unit_points.min(axis=0)
            max_x, max_y = unit_points.max(axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # We need to scale this to fit in 1x1 square.
            # scale = min(1 / width, 1 / height)
            # But we also need to account for the radius. The points are centers.
            # The circles have radius r.
            # The centers must be at least r away from the boundary.
            # So the effective box for centers is [ -0.5+r, 0.5-r ]
            # However, we are maximizing r.
            # If we scale the unit grid by factor s, the distance between adjacent points is s.
            # So r = s / 2.
            # The centers must fit in [ -0.5 + s/2, 0.5 - s/2 ]
            # The width of the centers is s * width.
            # We need s * width <= 1 - s  => s (width + 1) <= 1 => s <= 1 / (width + 1)
            # Similarly for height: s <= 1 / (height + 1)
            
            s = 1.0 / max(width + 1, height + 1)
            
            points = unit_points * s
            
            # Check if it fits
            # The min distance to boundary is 0.5 - max(|x|, |y|)
            # The min distance between points is s (for adjacent in hex)
            # Actually, the min distance might be smaller if the grid is distorted? No, it's regular.
            # But we need to check the actual min distance in this specific configuration.
            
            # Calculate score
            if len(points) == N:
                min_dist = 1.0
                # Point to point
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(points[i] - points[j])
                        if d < min_dist:
                            min_dist = d
                # Point to boundary
                for p in points:
                    d_boundary = 0.5 - np.max(np.abs(p))
                    if d_boundary < min_dist:
                        min_dist = d_boundary
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_points = points.copy()
                    
        return best_points

    points = create_hex_initialization()
    
    # 2. Local Optimization (Jiggling)
    # We will perform a simple local search to improve the packing.
    # This is crucial to push points into corners and equalize distances.
    
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in evaluation? 
    # Actually, the evaluator might run multiple times or expect deterministic. 
    # The task says "search score is value/seed". It implies we should be deterministic or robust.
    # Let's use a fixed seed to be safe, or just standard np.random.
    
    def calculate_min_dist(pts):
        min_d = 1.0
        # Boundary
        for p in pts:
            d_b = 0.5 - np.max(np.abs(p))
            if d_b < min_d:
                min_d = d_b
        # Pairwise
        # N=50, so 1225 pairs. Fast enough.
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_d:
                    min_d = d
        return min_d

    # Optimization Loop
    # We will try to increase the minimum distance.
    # Strategy: Find the "bottleneck" constraints (closest pair or point closest to wall).
    # Move points to relieve these constraints.
    
    current_min_dist = calculate_min_dist(points)
    max_iter = 5000
    step_size = 0.01
    
    for iter in range(max_iter):
        # Find the closest pair or closest point to boundary
        # To simplify, we can just randomly pick a point and try to move it.
        # Better: Pick the point involved in the minimum distance constraint.
        
        # Let's just do a simple random jitter with acceptance criteria.
        # This is essentially a very simple simulated annealing / hill climbing.
        
        i = rng.integers(0, N)
        original_point = points[i].copy()
        
        # Random perturbation
        perturbation = rng.normal(0, step_size, size=2)
        new_point = original_point + perturbation
        
        # Check if new point is inside the square (with margin for radius? No, radius is derived)
        # The point must be inside [-0.5, 0.5]
        if np.any(np.abs(new_point) > 0.5):
            continue
            
        points[i] = new_point
        
        new_min_dist = calculate_min_dist(points)
        
        if new_min_dist > current_min_dist:
            current_min_dist = new_min_dist
            # Optionally decrease step size
            if step_size > 0.001:
                step_size *= 0.999
        else:
            # Revert
            points[i] = original_point
            
    return points

# EVOLVE-BLOCK-END
