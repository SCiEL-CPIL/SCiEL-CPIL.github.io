import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Constants
    N = 12
    min_dist = np.sqrt(3)
    
    # Function to calculate the required side length s for a given set of points and outer hexagon angle theta
    def calc_s(points, theta):
        # points: (N, 2) array
        # theta: angle of outer hexagon in radians
        # Normals to the edges of the outer hexagon
        # Vertices are at theta + 60i. Edges are mid-way.
        # Normal angles: theta + 30 + 60i
        normal_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all points onto all normals
        # projections[i, k] = points[i] dot normals[k]
        projections = points @ normals.T
        max_proj = np.max(projections)
        
        # s * cos(30) = max_proj  => s = max_proj / cos(30)
        return max_proj / np.cos(np.pi/6)
    
    # Objective function for the optimizer
    # Variables: x1, y1, ..., x12, y12, theta
    def objective(vars):
        if time.time() > deadline:
            return 1e9
        pts = vars[:24].reshape(12, 2)
        theta = vars[24]
        
        # Check non-overlap
        # Calculate pairwise distances
        # This is O(N^2) = 144 ops. Fast.
        # Use broadcasting
        # dists[i, j] = norm(pts[i] - pts[j])
        # We only need min dist
        # To be fast, we can check a subset or use a trick.
        # For N=12, full check is fine.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        if min_d < min_dist - 1e-6:
            # Penalty for overlap
            return 1e9 + (min_dist - min_d) * 1000
        
        s = calc_s(pts, theta)
        return s

    # Generate initial guesses
    # 1. Hexagonal grid points
    # Axial coords to Cartesian
    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y

    # Generate a large grid of points
    grid_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x, y = axial_to_cart(q, r)
            grid_points.append([x, y])
    grid_points = np.array(grid_points)
    
    # We need to pick 12 points from this grid that form a compact cluster.
    # Strategy: Pick points closest to the origin, but ensure they are distinct and valid.
    # Actually, let's just use a few hand-picked topologies.
    
    # Topology 1: 7 center + 5 from next ring
    # Center (0,0)
    # Ring 1: 6 points
    # Ring 2: 12 points. Pick 5.
    
    # Let's create a list of candidate initial configurations
    candidates = []
    
    # Candidate 1: Symmetric-ish block
    # Rows of 4, 3, 3, 2?
    # Let's just take the 12 closest points to the origin from the grid, 
    # but this might form a blob that is not optimal for hex packing.
    # Better: Take a specific shape.
    
    # Shape: A 3x3 block of hexagons? No, that's 9.
    # Add 3 more.
    
    # Let's use a heuristic: 
    # Start with 7 (center + ring 1).
    # Add 5 more from ring 2, chosen to minimize the bounding box.
    # We'll just try a few random selections of 5 from ring 2.
    
    ring1 = []
    for i in range(6):
        q, r = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)][i]
        x, y = axial_to_cart(q, r)
        ring1.append([x, y])
    ring1 = np.array(ring1)
    
    ring2 = []
    # Ring 2 neighbors of Ring 1
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    ring2_coords = [
        (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0),
        (-1,-1), (0,-2), (1,-2), (2,-1)
    ]
    # Note: (-2,1) and (1,-1) etc are already in Ring 1?
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Ring 2 candidates are the neighbors of Ring 1 that are not in Ring 0 or 1.
    # Let's just generate all points in radius 2 and filter.
    
    all_ring2 = []
    for q, r in ring2_coords:
         # Check if it's actually distance 2 from center
         # Hex distance: (|q| + |r| + |q+r|)/2
         if (abs(q) + abs(r) + abs(q+r))/2 == 2:
             x, y = axial_to_cart(q, r)
             all_ring2.append([x, y])
             
    all_ring2 = np.array(all_ring2)
    
    # Base 7
    base7 = np.vstack([[0,0], ring1])
    
    # Try different subsets of 5 from all_ring2
    # To save time, we'll just try a few fixed patterns.
    # Pattern A: 5 consecutive
    # Pattern B: Spread out
    
    # Let's just run the optimizer starting from the 12 closest points to origin in the grid
    # This is a "blob".
    dists_to_origin = np.linalg.norm(grid_points, axis=1)
    sorted_indices = np.argsort(dists_to_origin)
    init_pts = grid_points[sorted_indices[:12]]
    
    # Center the points
    init_pts -= np.mean(init_pts, axis=0)
    
    initial_vars = np.append(init_pts.flatten(), 0.0)
    
    best_s = np.inf
    best_vars = None
    
    # Run Nelder-Mead
    # We might need multiple restarts.
    # Let's do 5 restarts with slightly perturbed initial points.
    
    for i in range(5):
        if time.time() > deadline - 60:
            break
            
        # Perturb initial points
        np.random.seed(i)
        perturbed_pts = init_pts + np.random.normal(0, 0.1, init_pts.shape)
        
        # Re-center
        perturbed_pts -= np.mean(perturbed_pts, axis=0)
        
        v0 = np.append(perturbed_pts.flatten(), 0.0)
        
        res = minimize(objective, v0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success or res.fun < best_s:
            if res.fun < 1e8: # Valid
                best_s = res.fun
                best_vars = res.x
    
    # If optimization failed or didn't find good, use the grid solution
    if best_vars is None or best_s > 1e8:
        # Use the initial grid points
        pts = init_pts
        # Find optimal theta for these points
        best_theta_s = np.inf
        best_theta = 0
        for t in np.linspace(0, np.pi/3, 100):
            s = calc_s(pts, t)
            if s < best_theta_s:
                best_theta_s = s
                best_theta = t
        best_s = best_theta_s
        best_vars = np.append(pts.flatten(), best_theta)
        
    # Extract results
    pts = best_vars[:24].reshape(12, 2)
    theta = best_vars[24]
    
    # Shift centers so that the outer hexagon is centered at origin?
    # The problem says: "after shifting all centers by minus the outer center"
    # The outer center is (0,0) in our calculation.
    # So we just return the pts as is.
    
    # Construct output arrays
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = pts[:, 0]
    inner_hex_data[:, 1] = pts[:, 1]
    inner_hex_data[:, 2] = 0.0  # Angle 0
    
    outer_hex_data = np.array([0.0, 0.0, np.degrees(theta)])
    
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

if __name__ == "__main__":
    inner, outer, s = hexagon_packing_12()
    print(f"Side length: {s}")
    print(f"Score: {1/s}")
