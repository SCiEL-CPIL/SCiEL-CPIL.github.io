import numpy as np
import time

N = 200

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 140.0  # 10s margin for 150s limit
    
    def score(C):
        # C is (N, 2)
        # Calculate min pairwise distance
        # and min distance to boundary
        # Score is min(min_pair_dist / 2, min_boundary_dist)
        
        # Pairwise distances
        # Use scipy if available, but np is enough
        D = np.sqrt(np.sum((C[:, np.newaxis, :] - C[np.newaxis, :, :])**2, axis=2))
        # Mask diagonal
        np.fill_diagonal(D, np.inf)
        min_pair_dist = np.min(D)
        
        # Boundary distance
        # Distance to closest wall
        dx = np.minimum(0.5 - C[:, 0], C[:, 0] + 0.5)
        dy = np.minimum(0.5 - C[:, 1], C[:, 1] + 0.5)
        min_boundary_dist = np.min(np.minimum(dx, dy))
        
        # Ensure points are inside
        if np.any(np.abs(C) > 0.5 + 1e-9):
            return 0.0
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def optimize(C_init, steps=5000, lr_init=0.01, momentum=0.9):
        C = C_init.copy()
        v = np.zeros_like(C)
        best_score = score(C)
        best_C = C.copy()
        
        # Precompute indices for vectorization if needed, but N=200 is small
        # Force calculation: F_i = sum_j 2 * (C_i - C_j) / ||C_i - C_j||^4
        # This is O(N^2)
        
        for step in range(steps):
            if time.time() - start_time > timeout:
                break
            
            # Calculate distances
            D2 = np.sum((C[:, np.newaxis, :] - C[np.newaxis, :, :])**2, axis=2)
            # Avoid division by zero
            D2[np.diag_indices_from(D2)] = 1.0 
            D4 = D2**2
            
            # Force: sum over j of (C_i - C_j) / D_ij^4
            # C_i is C[:, np.newaxis, :], C_j is C[np.newaxis, :, :]
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            F = np.sum(diff / D4[:, :, np.newaxis], axis=1)
            
            # Update velocity
            v = momentum * v + lr_init * F
            
            # Update position
            C_new = C + v
            
            # Project to square [-0.5, 0.5]
            C_new = np.clip(C_new, -0.5, 0.5)
            
            # Check score every 100 steps or so to save time
            if step % 100 == 0 or step == steps - 1:
                s = score(C_new)
                if s > best_score:
                    best_score = s
                    best_C = C_new.copy()
            
            C = C_new
            # Decay learning rate
            lr_init *= 0.999
            
        return best_C, best_score

    # Initialization: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    C_init = grid[:N]
    
    best_C, best_s = optimize(C_init, steps=2000)
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for i in range(5):
        if time.time() - start_time > timeout - 20:
            break
        # Random initialization within square
        C_rand = np.random.uniform(-0.4, 0.4, (N, 2))
        # Add some structure? Just random is often bad, but with many steps might work.
        # Better: Perturb the grid
        C_pert = C_init + np.random.normal(0, 0.05, C_init.shape)
        C_pert = np.clip(C_pert, -0.45, 0.45)
        
        C_opt, s_opt = optimize(C_pert, steps=2000)
        if s_opt > best_s:
            best_C = C_opt
            best_s = s_opt
            
    return best_C

# EVOLVE-BLOCK-END
