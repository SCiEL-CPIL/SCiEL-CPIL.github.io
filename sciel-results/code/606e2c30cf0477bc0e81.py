import numpy as np
from scipy.optimize import minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 8
    
    def compute_log_score(points):
        """
        Computes the log of the score.
        Score = Delta / n^n
        Delta = product of distances between all ordered pairs (i != j) 
                after scaling the set so that its diameter is 2.
        
        Let d_ij be the distance between point i and j.
        Let D = max(d_ij).
        Scaled distance d'_ij = d_ij * (2 / D).
        Delta = prod_{i != j} d'_ij.
        log(Score) = sum_{i != j} log(d'_ij) - n * log(n).
        """
        # points: (n, 2) array
        # Compute pairwise distances
        # Using broadcasting for efficiency
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal (self-distances are 0)
        mask = ~np.eye(n, dtype=bool)
        valid_dists = dists[mask]
        
        # Check for distinct points
        if np.any(valid_dists < 1e-10):
            return -np.inf
            
        D = np.max(valid_dists)
        
        if D < 1e-10:
            return -np.inf
            
        # log(Delta) = sum(log(d'_ij)) = sum(log(d_ij * 2/D))
        #            = sum(log(d_ij)) + n(n-1) * (log(2) - log(D))
        
        # Note: "product over ordered pairs" means n*(n-1) terms.
        # valid_dists contains n*(n-1) elements (upper and lower triangle).
        
        log_prod_unscaled = np.sum(np.log(valid_dists))
        log_scaled_prod = log_prod_unscaled + n * (n - 1) * (np.log(2.0) - np.log(D))
        
        # log(Score) = log_scaled_prod - n * log(n)
        return log_scaled_prod - n * np.log(n)

    def objective(x):
        """
        Objective function for minimization (we maximize log_score, so minimize -log_score).
        x is a flat array of 16 coordinates.
        """
        pts = x.reshape((n, 2))
        
        # Soft penalty for points being too close or too far apart
        # The problem doesn't explicitly bound coordinates, but |coord| < 10^6 is required.
        # The scaling in the score makes the absolute scale irrelevant, but 
        # numerical stability is better if points are roughly in [-2, 2].
        
        # Check for distinct points
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        valid_dists = dists[mask]
        
        if np.any(valid_dists < 1e-6):
            return 1e10
            
        s = compute_log_score(pts)
        if not np.isfinite(s):
            return 1e10
            
        return -s

    def step(x, stepsize):
        """
        Perturbation function for Basin Hopping.
        """
        x = np.array(x)
        # Random Gaussian perturbation
        x += stepsize * np.random.normal(0, 1, size=x.shape)
        return x

    # Initial guesses
    # 1. Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    x_oct = pts_oct.flatten()
    
    # 2. Two Squares (D4 symmetry candidate)
    # Square 1: radius 1, angles 0, 90, 180, 270
    # Square 2: radius r, angles 45, 135, 225, 315
    # Let's try a few variations of r
    best_score = -np.inf
    best_x = x_oct
    
    # Evaluate regular octagon
    s_oct = compute_log_score(pts_oct)
    best_score = s_oct
    best_x = x_oct.copy()
    
    # Try some structured initial guesses
    structured_starts = []
    
    # Variation 1: Two concentric squares
    for r2 in np.linspace(0.5, 1.5, 5):
        a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        a2 = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)
        pts1 = r2 * np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = 1.0 * np.column_stack([np.cos(a2), np.sin(a2)])
        pts = np.vstack([pts1, pts2])
        # Normalize to diameter 2 for consistent comparison? 
        # The objective handles scaling, so we don't need to, but it helps optimizer.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        D = np.max(dists[~np.eye(n, dtype=bool)])
        if D > 0:
            pts = pts * (2.0 / D)
        structured_starts.append(pts.flatten())
        
    # Variation 2: Perturbed Octagon
    rng = np.random.default_rng(42)
    for _ in range(10):
        jitter = rng.normal(0, 0.2, size=(n, 2))
        pts = pts_oct + jitter
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        D = np.max(dists[~np.eye(n, dtype=bool)])
        if D > 0:
            pts = pts * (2.0 / D)
        structured_starts.append(pts.flatten())
    
    # Evaluate structured starts
    for x0 in structured_starts:
        if time.time() > deadline:
            break
        s = compute_log_score(x0.reshape((n, 2)))
        if s > best_score:
            best_score = s
            best_x = x0.copy()

    # Use Basin Hopping to explore the landscape
    # Basin Hopping uses a local minimizer (L-BFGS-B) and a global step function.
    # This is structurally different from just running L-BFGS-B from multiple starts
    # because it actively jumps between basins.
    
    # We will run Basin Hopping from the best starting point found so far.
    # We might also run it from a few other promising starts if time permits.
    
    candidates = [best_x]
    # Add a couple of other diverse starts
    if len(structured_starts) > 0:
        candidates.append(structured_starts[0])
    if len(structured_starts) > 5:
        candidates.append(structured_starts[5])
        
    # Deduplicate candidates roughly
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for uc in unique_candidates:
            if np.linalg.norm(c - uc) < 1e-3:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
            
    # Run Basin Hopping on the best candidate
    # stepsize: typical step size for the random walk
    # miniter: number of iterations for the local minimizer
    # maxiter: number of basin hopping steps
    
    for x0 in unique_candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            bh_res = basinhopping(
                objective, 
                x0, 
                step=step, 
                stepsize=0.5, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': [(-2.0, 2.0)] * (2 * n)},
                niter=50, 
                T=100.0, 
                accept_test=None, # Use default Metropolis acceptance
                seed=42
            )
            
            final_score = -bh_res.fun
            if final_score > best_score:
                best_score = final_score
                best_x = bh_res.x.copy()
                
        except Exception:
            pass

    # If time remains, do a fine-tuning pass with L-BFGS-B on the best found
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective, 
                best_x, 
                method='L-BFGS-B', 
                bounds=[(-2.0, 2.0)] * (2 * n),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success:
                final_score = -res.fun
                if final_score > best_score:
                    best_score = final_score
                    best_x = res.x.copy()
        except Exception:
            pass

    # Final check: Ensure points are distinct
    pts = best_x.reshape((n, 2))
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists < 1e-5):
        # Fallback to regular octagon if points collapsed
        pts = pts_oct

    return [[float(x), float(y)] for x, y in pts]
