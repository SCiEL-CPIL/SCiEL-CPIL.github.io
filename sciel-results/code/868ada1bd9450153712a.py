# EVOLVE-BLOCK-START
import numpy as np

def _generate_hex_grid(n):
    """
    Generate a hexagonal lattice and select n points.
    Strategy: Generate a grid, then select points to maximize the minimum
    of (distance to boundary, distance to nearest neighbor).
    """
    best_points = None
    best_score = -1
    
    # Optimal radius ~ 0.0366, diameter ~ 0.0732
    spacings = np.linspace(0.070, 0.076, 15)
    
    for d in spacings:
        row_height = d * np.sqrt(3) / 2
        col_width = d
        
        max_y = 0.5 + row_height
        max_x = 0.5 + col_width
        
        y_vals = np.arange(-max_y, max_y + row_height, row_height)
        points = []
        
        for j, y in enumerate(y_vals):
            if y < -0.5 - d or y > 0.5 + d:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = col_width / 2.0
            
            x_vals = np.arange(-max_x + offset, max_x + offset + col_width, col_width)
            for x in x_vals:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        
        if len(points) < n:
            continue
            
        points = np.array(points)
        
        # Calculate metrics for selection
        # 1. Distance to boundary
        dists_to_boundary = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
        
        # 2. Distance to nearest neighbor
        # This is O(N^2), but N is small enough (~400) for a few spacings
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_nn = np.min(dists, axis=1)
        
        # Score: min(boundary_dist, min_nn / 2)
        # We want to maximize this score for the selected set.
        # Greedy selection:
        # Sort all points by score descending.
        scores = np.minimum(dists_to_boundary, min_nn / 2)
        idx = np.argsort(scores)[::-1]
        
        # Take top n
        selected = points[idx[:n]]
        
        # Recalculate actual score of selected set
        diff_sel = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
        dists_sel = np.sqrt(np.sum(diff_sel ** 2, axis=-1))
        np.fill_diagonal(dists_sel, np.inf)
        min_pair = np.min(dists_sel)
        
        boundary_dists_sel = np.minimum(0.5 - np.abs(selected[:, 0]), 0.5 - np.abs(selected[:, 1]))
        min_boundary = np.min(boundary_dists_sel)
        
        score = min(min_pair / 2, min_boundary)
        
        if score > best_score:
            best_score = score
            best_points = selected.copy()
            
    if best_points is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
        
    return best_points

def construct():
    points = _generate_hex_grid(N)
    
    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair / 2, min_boundary)
    
    r = calc_radius(points)
    
    # Optimization parameters
    max_iter = 3000
    lr_start = 0.01
    lr_end = 0.0001
    momentum = 0.9
    v = np.zeros_like(points)
    
    for it in range(max_iter):
        lr = lr_start * (lr_end / lr_start) ** (it / max_iter)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Use Softplus for smoothness
        # Penalty: sum softplus(2r - dists) + sum softplus(r - b_i)
        # Gradient of softplus(x) is sigmoid(x)
        
        mask = dists < 2 * r
        safe_dists = np.where(dists > 1e-8, dists, 1e-8)
        
        # Violation for pairs
        viol_pair = 2 * r - dists
        # Only compute gradient for violating pairs to save time, 
        # but softplus gradient is small for large negative x.
        # We'll compute for all pairs to be safe, but mask out large positive violations?
        # No, just compute sigmoid(viol_pair) * (1/dists)
        
        # grad_pair_i = sum_j sigmoid(2r - d_ij) * (p_i - p_j) / d_ij
        # Note: d(2r - d_ij)/dp_i = -(p_i - p_j)/d_ij
        # dV/dp_i = sum_j sigmoid(2r - d_ij) * (-(p_i - p_j)/d_ij)
        # Wait, V = sum softplus(2r - d). dV/dp = sum sigmoid(2r-d) * d(2r-d)/dp.
        # d(2r-d)/dp_i = -(p_i - p_j)/d.
        # So term is - sigmoid(2r-d) * (p_i - p_j) / d.
        
        sig_pair = np.exp(-viol_pair) / (1 + np.exp(-viol_pair)) # sigmoid(viol_pair)
        # Mask out non-violating to save computation? Sigmoid is small anyway.
        weights = sig_pair / safe_dists
        np.fill_diagonal(weights, 0)
        
        row_sums = np.sum(weights, axis=1)
        weighted_points = weights @ points
        pair_grad = -row_sums[:, np.newaxis] * points + weighted_points
        
        # Boundary
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        bound_grad = np.zeros_like(points)
        
        mask_x = bx <= by
        mask_y = ~mask_x
        
        viol_x = r - bx
        viol_y = r - by
        
        # Gradient of softplus(r - b) wrt x
        # dV/dx = sigmoid(r - b) * d(r - b)/dx
        # d(r - bx)/dx = -d(bx)/dx = -(-sign(x)) = sign(x)
        
        sig_x = np.exp(-viol_x) / (1 + np.exp(-viol_x))
        sig_y = np.exp(-viol_y) / (1 + np.exp(-viol_y))
        
        grad_x = np.where(mask_x, sig_x * np.sign(points[:, 0]), 0)
        grad_y = np.where(mask_y, sig_y * np.sign(points[:, 1]), 0)
        
        bound_grad[:, 0] = grad_x
        bound_grad[:, 1] = grad_y
        
        total_grad = pair_grad + bound_grad
        
        # Momentum update
        v = momentum * v - lr * total_grad
        points += v
        
        # Project to square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Update r
        current_r = calc_radius(points)
        
        if current_r > r:
            r = current_r
        elif current_r < r:
            r = current_r * 0.99
            
    return points
# EVOLVE-BLOCK-END
