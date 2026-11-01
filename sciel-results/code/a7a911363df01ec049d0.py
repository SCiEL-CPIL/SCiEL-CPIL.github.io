"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _compute_radius(centers):
    """Compute the valid radius for a set of centers."""
    # Check finite values
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check shape
    if centers.shape != (N, 2):
        return 0.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    # Compute min distance between pairs
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0)  # pairs with distance < 1.0
    if len(pairs) == 0:
        # No pairs within 1.0, so min pair distance is large
        # But we need the actual min
        # Fall back to full computation if no pairs found (unlikely for 50 points in unit square)
        # Actually query_pairs(1.0) returns all pairs with dist < 1.0
        # For 50 points in unit square, max distance is sqrt(2) ~ 1.414, so some pairs may be > 1.0
        # Let's use a larger threshold or compute all pairs
        # Better: use query with k=2 to get nearest neighbor for each point
        dists, _ = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
        # Also need to check pairs that might be > 1.0? No, min will be in the close pairs
        # But if all pairs are > 1.0, then min_pair_dist stays inf, which is wrong
        # Let's just compute all pair distances for safety with 50 points (1225 pairs)
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_pair_dist:
                    min_pair_dist = d
    
    # Compute min distance to boundary
    # Distance to each wall
    dist_left = centers[:, 0] + 0.5
    dist_right = 0.5 - centers[:, 0]
    dist_bottom = centers[:, 1] + 0.5
    dist_top = 0.5 - centers[:, 1]
    min_boundary_dist = np.min([np.min(dist_left), np.min(dist_right), 
                                 np.min(dist_bottom), np.min(dist_top)])
    
    # Radius is half of min(pair_dist, boundary_dist)
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    return r


def _hexagonal_initialization(n, square_size=1.0):
    """Create a hexagonal-like initial placement for n circles in unit square."""
    # For 50 circles, use a hexagonal pattern
    # Approximate: rows of 7 or 8 circles
    # Hexagonal packing: rows offset by half spacing
    # Let's try different configurations
    
    # Configuration 1: 7 rows with varying counts
    # 7+7+7+7+7+7+8 = 50? No, let's think differently
    
    # Better: use a grid with hexagonal offset
    # For hexagonal packing, the ratio of horizontal to vertical spacing is 2/sqrt(3)
    
    # Let's use a simple approach: place circles in a hexagonal pattern
    # with appropriate spacing to fit in the square
    
    centers = []
    
    # Try to fit a hexagonal pattern
    # For n=50, we can use rows of approximately sqrt(n) circles
    # Hexagonal: each row has either k or k-1 circles
    
    # Let's try 7 rows: some with 7, some with 8
    # 7*7 = 49, so 6 rows of 7 and 1 row of 8? Or mix
    
    # Actually, let's use a more systematic approach
    # Start with a square grid and perturb to hexagonal
    
    m = int(np.ceil(np.sqrt(n)))
    # For n=50, m=8, so 8x8 grid with 14 empty spots
    
    # Alternative: use hexagonal coordinates
    # Let's try: 7 columns, with rows having 7 or 8 circles
    # Total: if 7 rows, need 50/7 ≈ 7.14, so some rows have 8
    
    # Simpler: create a hexagonal lattice and select n points
    # Hexagonal lattice with spacing d
    # To fit in unit square, d should be about 1/7 or so
    
    # Let's use a practical approach: 
    # Place in a hexagonal pattern with 7-8 circles per row
    
    # Row 1: 8 circles
    # Row 2: 7 circles (offset)
    # Row 3: 8 circles
    # Row 4: 7 circles
    # Row 5: 8 circles
    # Row 6: 7 circles
    # Row 7: 7 circles
    # Total: 8+7+8+7+8+7+7 = 52, too many
    
    # Let's try:
    # Row 1: 7
    # Row 2: 8
    # Row 3: 7
    # Row 4: 8
    # Row 5: 7
    # Row 6: 8
    # Row 7: 5
    # Total: 7+8+7+8+7+8+5 = 50
    
    rows_counts = [7, 8, 7, 8, 7, 8, 5]
    
    # Calculate spacing
    # Horizontal spacing for 8 circles in unit width: 1/9 (with margins)
    # Actually, for hexagonal packing, let's compute properly
    
    # Let h_spacing be horizontal distance between centers in same row
    # For row with k circles, they span (k-1)*h_spacing
    # To fit in [-0.5, 0.5], need (k-1)*h_spacing + 2*r <= 1
    # But we don't know r yet. Let's just place them and compute.
    
    # For hexagonal: vertical spacing = h_spacing * sqrt(3)/2
    # Offset alternate rows by h_spacing/2
    
    # Let's estimate h_spacing
    # Max row has 8 circles: span = 7*h_spacing
    # 7 rows: total height = 6 * v_spacing = 6 * h_spacing * sqrt(3)/2 = 3*sqrt(3)*h_spacing
    # To fit in unit square: 7*h_spacing <= 1 and 3*sqrt(3)*h_spacing <= 1
    # h_spacing <= 1/7 ≈ 0.1429 and h_spacing <= 1/(3*sqrt(3)) ≈ 0.1925
    # So h_spacing ≈ 0.14
    
    h_spacing = 1.0 / 7.5  # Slightly less than 1/7 to allow for offsets
    
    v_spacing = h_spacing * np.sqrt(3) / 2
    
    y_start = -0.5 + v_spacing * 3 + 0.05  # Center vertically
    # Actually, let's center the pattern
    
    total_height = (len(rows_counts) - 1) * v_spacing
    y_start = -total_height / 2
    
    for row_idx, count in enumerate(rows_counts):
        y = y_start + row_idx * v_spacing
        # Horizontal offset for alternating rows
        if row_idx % 2 == 1:
            x_offset = h_spacing / 2
        else:
            x_offset = 0
        
        # Center the row horizontally
        total_width = (count - 1) * h_spacing
        x_start = -total_width / 2 + x_offset
        
        for col_idx in range(count):
            x = x_start + col_idx * h_spacing
            centers.append([x, y])
    
    centers = np.array(centers[:n])
    
    # Ensure centers are within bounds (with small margin)
    centers[:, 0] = np.clip(centers[:, 0], -0.45, 0.45)
    centers[:, 1] = np.clip(centers[:, 1], -0.45, 0.45)
    
    return centers


def construct():
    """Construct centers of N equal circles in unit square to maximize radius."""
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Initial placement: hexagonal
    centers = _hexagonal_initialization(N)
    
    # Compute initial radius
    current_radius = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Simulated annealing / gradient-based optimization
    # We'll use a combination of random perturbations and local optimization
    
    np.random.seed(42)
    
    # Parameters
    T_start = 0.01
    T_end = 1e-8
    n_iterations = 100000
    
    # Adaptive step size
    step_size = 0.02
    
    for it in range(n_iterations):
        # Check time
        if time.time() > deadline:
            break
        
        # Temperature
        alpha = it / max(n_iterations - 1, 1)
        T = T_start * (T_end / T_start) ** alpha
        
        # Generate perturbation
        # Random direction
        delta = np.random.randn(N, 2) * step_size
        
        # New centers
        new_centers = centers + delta
        
        # Clip to stay within bounds (with small margin)
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.49, 0.49)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.49, 0.49)
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        
        # Accept or reject
        if new_radius >= current_radius:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
            # Reduce step size on improvement
            step_size *= 0.999
        else:
            # Metropolis criterion
            delta_r = new_radius - current_radius
            if delta_r < 0 and T > 0:
                accept_prob = np.exp(delta_r / T)
                if np.random.random() < accept_prob:
                    centers = new_centers
                    current_radius = new_radius
            # Don't update best if worse
        
        # Occasionally reset to best if stuck
        if it % 5000 == 0 and it > 0:
            if current_radius < best_radius * 0.95:
                centers = best_centers + np.random.randn(N, 2) * 0.01
                current_radius = _compute_radius(centers)
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 5:
        step_size = 0.005
        for it in range(20000):
            if time.time() > deadline:
                break
            
            delta = np.random.randn(N, 2) * step_size
            new_centers = centers + delta
            new_centers[:, 0] = np.clip(new_centers[:, 0], -0.49, 0.49)
            new_centers[:, 1] = np.clip(new_centers[:, 1], -0.49, 0.49)
            
            new_radius = _compute_radius(new_centers)
            
            if new_radius >= current_radius:
                centers = new_centers
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
                step_size *= 0.9995
            else:
                step_size *= 1.0005
                step_size = min(step_size, 0.01)
    
    # Ensure final result is valid
    final_radius = _compute_radius(best_centers)
    if final_radius < best_radius * 0.99:
        # Something went wrong, try to fix
        best_centers[:, 0] = np.clip(best_centers[:, 0], -0.49, 0.49)
        best_centers[:, 1] = np.clip(best_centers[:, 1], -0.49, 0.49)
    
    return best_centers
# EVOLVE-BLOCK-END
