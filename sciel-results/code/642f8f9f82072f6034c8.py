import numpy as np

def construct():
    N = 32
    # Initialize in a grid
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                centers.append([x, y])
    centers = np.array(centers)
    radii = np.zeros(N)
    
    # Parameters
    T = 500
    alpha_r = 0.5
    alpha_c = 0.05
    
    for t in range(T):
        # Compute distance matrix
        # D[i, j] = distance between i and j
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff**2, axis=2))
        
        # Compute wall distances
        W = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                       np.minimum(centers[:, 1], 1 - centers[:, 1]))
        
        # Update radii
        # r_i <= D_ij - r_j for all j != i
        # r_i <= W_i
        # We solve r_i = min(W_i, min_j (D_ij - r_j))
        # This is a system of equations. We use iterative relaxation.
        # To vectorize the min over j:
        # Create a matrix of limits L[i, j] = D[i, j] - radii[j]
        L = D - radii[np.newaxis, :]
        np.fill_diagonal(L, np.inf)
        min_neighbor_limit = np.min(L, axis=1)
        target_radii = np.minimum(W, min_neighbor_limit)
        target_radii = np.maximum(target_radii, 0)
        
        # Relax radii
        radii = (1 - alpha_r) * radii + alpha_r * target_radii
        
        # Update centers
        # We want to move each center i in the direction that increases its radius.
        # The radius is limited by the closest constraints.
        # We compute a "force" vector for each circle.
        forces = np.zeros_like(centers)
        
        # Wall forces
        # If x_i is close to 0, push right (+x). If close to 1, push left (-x).
        # We only apply force if the wall is an active constraint (i.e., W_i is the limiting factor or close to it).
        # Approximation: Always apply a small restoring force to the center if too close to wall?
        # Better: Only apply force if the wall distance is the minimum.
        
        # Let's identify active constraints for each circle.
        # Active if constraint value is close to current radius.
        tol = 1e-4
        
        # Wall active?
        # Left: x < tol + r_i? No, if x_i - r_i < tol, then left wall is active.
        # Actually, the constraint is r_i <= x_i. So if x_i - r_i < tol, active.
        
        # Let's compute the "slack" for each constraint.
        # Wall slacks:
        slack_left = centers[:, 0] - radii
        slack_right = (1 - centers[:, 0]) - radii
        slack_bottom = centers[:, 1] - radii
        slack_top = (1 - centers[:, 1]) - radii
        
        # Neighbor slacks:
        # slack_ij = D_ij - r_i - r_j
        slack_neighbors = D - radii[:, np.newaxis] - radii[np.newaxis, :]
        np.fill_diagonal(slack_neighbors, np.inf)
        
        # For each circle i, find if any constraint is active (slack < tol).
        # If active, add force in the direction away from the constraint.
        
        # Wall forces
        # Left active: slack_left < tol -> force +x
        # Right active: slack_right < tol -> force -x
        # Bottom active: slack_bottom < tol -> force +y
        # Top active: slack_top < tol -> force -y
        
        forces[:, 0] += (slack_left < tol) * 1.0
        forces[:, 0] -= (slack_right < tol) * 1.0
        forces[:, 1] += (slack_bottom < tol) * 1.0
        forces[:, 1] -= (slack_top < tol) * 1.0
        
        # Neighbor forces
        # If slack_ij < tol, force i away from j.
        # Direction is (x_i - x_j) / D_ij.
        # We can vectorize this:
        # For each i, sum of normalized diff vectors for all j where slack < tol.
        
        # Create a mask of active neighbor constraints
        active_mask = slack_neighbors < tol
        
        # Compute normalized diff vectors
        # diff[i, j, :] is x_j - x_i? No, diff = centers[:, None] - centers[None, :] so diff[i, j] = c_i - c_j.
        # We want direction from j to i, which is c_i - c_j.
        # So we use diff directly.
        
        # Normalize diff vectors
        norms = D[:, :, np.newaxis]
        # Avoid division by zero
        norms[norms == 0] = 1
        unit_diff = diff / norms
        
        # Mask out inactive constraints
        unit_diff = unit_diff * active_mask[:, :, np.newaxis]
        
        # Sum over neighbors for each i
        neighbor_forces = np.sum(unit_diff, axis=1)
        
        forces += neighbor_forces
        
        # Normalize forces to unit vectors to have consistent step size
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1
        unit_forces = forces / force_norms
        
        # Update centers
        # Use a decaying step size
        step = alpha_c * (1.0 - t / T) + 0.001
        centers += unit_forces * step
        
        # Clip centers to [0, 1] to ensure they stay in bounds
        # Actually, we should keep them slightly inside to avoid numerical issues with walls
        centers = np.clip(centers, 0.0, 1.0)
        
    # Final radius calculation
    # Iterate a few times to converge radii given final centers
    for _ in range(10):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff**2, axis=2))
        W = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                       np.minimum(centers[:, 1], 1 - centers[:, 1]))
        L = D - radii[np.newaxis, :]
        np.fill_diagonal(L, np.inf)
        min_neighbor_limit = np.min(L, axis=1)
        target_radii = np.minimum(W, min_neighbor_limit)
        target_radii = np.maximum(target_radii, 0)
        radii = target_radii
        
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [centers[i][0], centers[i][1], radii[i]]
    return result
