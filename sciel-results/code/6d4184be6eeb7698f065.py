"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Grid parameters
    G = 50
    x = np.linspace(0, 1, G)
    y = np.linspace(0, 1, G)
    xx, yy = np.meshgrid(x, y)
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    num_points = len(centers)

    # Precompute distance matrix
    # Using scipy if available, otherwise numpy
    try:
        from scipy.spatial.distance import cdist
        D = cdist(centers, centers, metric='euclidean')
    except ImportError:
        # Fallback to numpy (slower but works)
        D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # Precompute boundary limits for each point
    bounds = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                        np.minimum(centers[:, 1], 1 - centers[:, 1]))

    def can_pack(R, num_restarts=100):
        threshold = 2 * R
        # Indices where the circle of radius R fits in the square
        valid_indices = np.where(bounds >= R)[0]
        
        if len(valid_indices) < N:
            return False, None

        for _ in range(num_restarts):
            # Randomize the order of valid candidates
            idx = np.random.permutation(valid_indices)
            selected = []
            
            for i in idx:
                if len(selected) == N:
                    return True, selected
                
                if len(selected) > 0:
                    # Check distance to all selected points
                    # D[i, selected] gives distances from i to each selected point
                    if np.any(D[i, selected] < threshold - 1e-9):
                        continue
                
                selected.append(i)
            
            # If loop finishes without selecting N, this restart failed
            # Try next restart
            continue
            
        return False, None

    # Binary search for maximum radius R
    lo = 0.01
    hi = 0.5
    best_packing = None
    best_R = 0.01

    # We will perform binary search steps.
    # Each step calls can_pack.
    # can_pack does 100 restarts.
    # With precomputed D, each restart is fast.
    
    for _ in range(25):  # ~25 steps for good precision
        if time.time() > deadline:
            break
            
        mid = (lo + hi) / 2
        success, packing = can_pack(mid, num_restarts=100)
        
        if success:
            best_packing = packing
            best_R = mid
            lo = mid
        else:
            hi = mid

    if best_packing is None:
        # Fallback: return a valid small packing
        # Just take the first N valid points for a small R
        R_fallback = 0.02
        valid_indices = np.where(bounds >= R_fallback)[0]
        if len(valid_indices) >= N:
            selected = valid_indices[:N]
            # Check if they actually don't overlap (they might)
            # For fallback, just return them, evaluator might reject if invalid, 
            # but 0.02 is small enough that random grid points are likely disjoint? 
            # No, grid points are close. 
            # Better fallback: seed-like grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
        else:
            return np.zeros((N, 3))

    # Construct result array
    result = np.zeros((N, 3))
    for i, idx in enumerate(best_packing):
        result[i] = np.append(centers[idx], best_R)

    return result
# EVOLVE-BLOCK-END
