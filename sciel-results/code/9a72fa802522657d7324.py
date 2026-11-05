import numpy as np
from scipy.optimize import minimize

def construct():
    # Regular octagon
    n = 8
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Function to compute negative log product of distances
    def neg_log_prod(x):
        p = x.reshape(n, 2)
        # Compute all pairwise distances
        # dist[i, j] = ||p[i] - p[j]||
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        # Avoid log(0)
        d = np.maximum(d, 1e-10)
        return -np.sum(np.log(d))
    
    # We want to maximize product, so minimize negative log product.
    # Constraint: diameter <= 2.
    # We will use a penalty method or just normalize.
    # Let's try normalizing after each step in a custom loop.
    
    best_pts = pts
    best_score = 1.0 # Score for regular octagon is 1
    
    def compute_score(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        if np.any(d < 1e-10):
            return 0.0
        prod = np.prod(d)
        # Delta is product of ordered pairs = prod^2
        # Score = Delta / n^n = prod^2 / 8^8
        # Regular octagon prod = 8^4 = 4096? 
        # Let's check: 8^8 = 16777216.
        # prod^2 = 16777216 => prod = 4096.
        # Score = 1.
        return (prod**2) / (8**8)

    # Gradient ascent with normalization
    step_size = 0.01
    for _ in range(200):
        # Compute gradient of log product
        p = best_pts
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # grad[i] = sum_{j != i} (p[i] - p[j]) / dists_sq[i,j]
        # dists_sq is 8x8
        # Avoid division by zero
        dists_sq[~np.eye(n, dtype=bool)] = np.maximum(dists_sq[~np.eye(n, dtype=bool)], 1e-10)
        # Invert
        inv_dists_sq = 1.0 / dists_sq
        np.fill_diagonal(inv_dists_sq, 0)
        
        # grad = diff * inv_dists_sq[:, :, np.newaxis]
        # Sum over j (axis 1)
        grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
        
        # Update
        new_pts = p + step_size * grad
        
        # Normalize to diameter 2
        diff_new = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
        dists_new = np.sqrt(np.sum(diff_new**2, axis=2))
        D = np.max(dists_new)
        if D < 1e-10:
            break
        new_pts = new_pts * (2.0 / D)
        
        # Check score
        new_score = compute_score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
        else:
            # If we didn't improve, maybe we are at a local max or step is too big
            # Try smaller step? Or break?
            # For now, just continue with same step, hoping to escape noise
            pass
            
    return best_pts.tolist()
