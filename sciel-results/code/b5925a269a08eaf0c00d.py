import numpy as np

N = 26

def _energy(pos, r):
    """
    Calculate the energy of the configuration.
    Energy is sum of squared overlaps and squared boundary violations.
    """
    x = pos[:, 0]
    y = pos[:, 1]
    n = len(x)
    
    # Overlap energy
    # d_ij is distance between i and j
    # We want d_ij >= 2r
    # Penalty is max(0, 2r - d_ij)^2
    
    # Compute pairwise distances
    # Using broadcasting
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    
    # Diagonal is 0, we ignore it
    np.fill_diagonal(d2, np.inf)
    
    d = np.sqrt(d2)
    
    # Overlap amount
    overlap = np.maximum(0, 2*r - d)
    overlap_energy = np.sum(overlap**2)
    
    # Boundary energy
    # x should be in [r, 1-r]
    # y should be in [r, 1-r]
    
    x_violation_left = np.maximum(0, r - x)
    x_violation_right = np.maximum(0, x - (1 - r))
    y_violation_left = np.maximum(0, r - y)
    y_violation_right = np.maximum(0, y - (1 - r))
    
    boundary_energy = np.sum(x_violation_left**2) + np.sum(x_violation_right**2) + \
                      np.sum(y_violation_left**2) + np.sum(y_violation_right**2)
    
    return overlap_energy + boundary_energy

def _gradient(pos, r):
    """
    Calculate the gradient of the energy function.
    """
    x = pos[:, 0]
    y = pos[:, 1]
    n = len(x)
    
    grad_x = np.zeros_like(x)
    grad_y = np.zeros_like(y)
    
    # Overlap gradient
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    
    # d(E_overlap)/d(x_i) = sum_{j!=i} -2 * (2r - d_ij) * (dx_ij / d_ij)
    # Note: overlap = max(0, 2r - d). If 2r - d > 0, gradient is -2(2r-d) * (dx/d).
    # If 2r - d <= 0, gradient is 0.
    
    # Vectorized calculation
    # overlap_term = 2 * (2r - d) where d < 2r, else 0
    # We need to be careful with d=0, but diagonal is inf so d>0 for off-diagonal.
    
    overlap_mask = d < 2*r
    # For indices where overlap exists
    # grad contribution from pair (i,j) to i is -2 * (2r - d_ij) * (x_i - x_j) / d_ij
    # grad contribution from pair (i,j) to j is -2 * (2r - d_ij) * (x_j - x_i) / d_ij = +2 * (2r - d_ij) * (x_i - x_j) / d_ij
    
    # Let's compute the force vector
    # F_ij = -2 * (2r - d_ij) * (r_vec_ij / d_ij)
    # where r_vec_ij = pos_i - pos_j
    
    # We can compute a matrix of forces
    # force_x[i,j] = -2 * (2r - d_ij) * dx_ij / d_ij if d_ij < 2r else 0
    
    factor = np.where(overlap_mask, -2 * (2*r - d) / d, 0)
    
    # grad_x[i] = sum_j factor[i,j] * dx[i,j]
    # Note: dx[i,j] = x[i] - x[j]
    # The term is -2(2r-d)/d * (x_i - x_j). This is the force on i due to j?
    # Energy E = sum (2r-d)^2. dE/dx_i = sum_j 2(2r-d) * d(2r-d)/dx_i = sum_j 2(2r-d) * (-dx_ij/d).
    # So grad_x[i] = sum_j -2(2r-d)/d * (x_i - x_j).
    # This matches factor * dx.
    
    grad_x += np.sum(factor * dx, axis=1)
    grad_y += np.sum(factor * dy, axis=1)
    
    # Boundary gradient
    # E_b = sum (max(0, r-x)^2 + max(0, x-(1-r))^2 + ...)
    # d/dx max(0, r-x)^2 = 2(r-x) * (-1) = -2(r-x) if x < r, else 0
    # d/dx max(0, x-(1-r))^2 = 2(x-(1-r)) * (1) = 2(x-(1-r)) if x > 1-r, else 0
    
    # Left wall
    mask_left_x = x < r
    grad_x[mask_left_x] += -2 * (r - x[mask_left_x])
    
    # Right wall
    mask_right_x = x > 1 - r
    grad_x[mask_right_x] += 2 * (x[mask_right_x] - (1 - r))
    
    # Bottom wall
    mask_left_y = y < r
    grad_y[mask_left_y] += -2 * (r - y[mask_left_y])
    
    # Top wall
    mask_right_y = y > 1 - r
    grad_y[mask_right_y] += 2 * (y[mask_right_y] - (1 - r))
    
    return np.column_stack((grad_x, grad_y))

def _try_pack(r, n=26, restarts=50, steps=500, lr=0.05, tol=1e-6):
    """
    Try to pack n circles of radius r.
    Returns (success, positions).
    """
    best_energy = np.inf
    best_pos = None
    
    for _ in range(restarts):
        # Initialize positions randomly
        # Ensure they start somewhat inside to avoid huge initial gradient
        # Uniform in [r, 1-r] is good
        pos = np.random.uniform(r, 1-r, size=(n, 2))
        
        # Gradient Descent
        # Adaptive learning rate or simple decay
        current_lr = lr
        for step in range(steps):
            grad = _gradient(pos, r)
            energy = _energy(pos, r)
            
            if energy < tol:
                return True, pos
            
            # Update
            # Simple gradient descent
            pos -= current_lr * grad
            
            # Project back to [0,1] roughly? 
            # The boundary penalty handles this, but explicit projection helps stability
            pos[:, 0] = np.clip(pos[:, 0], 0, 1)
            pos[:, 1] = np.clip(pos[:, 1], 0, 1)
            
            # Decay learning rate
            current_lr *= 0.99
            
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
                
    return False, best_pos

def construct():
    n = 26
    
    # Binary search for the maximum radius r
    # Lower bound: 1/sqrt(4*n) approx? 
    # Area of n circles <= 1 -> n*pi*r^2 <= 1 -> r <= 1/sqrt(n*pi)
    # 1/sqrt(26*pi) approx 0.11
    # Upper bound: 0.25 (impossible for 26, but safe upper bound for BS)
    
    r_low = 0.0
    r_high = 0.2
    
    best_r = r_low
    best_pos = None
    
    # We want to find the max r such that packing is possible.
    # To save time, we can do a coarse search first.
    
    # Coarse search
    for _ in range(20):
        r_mid = (r_low + r_high) / 2
        success, pos = _try_pack(r_mid, n=n, restarts=10, steps=300, lr=0.05)
        if success:
            best_r = r_mid
            best_pos = pos
            r_low = r_mid
        else:
            r_high = r_mid
            
    # Fine search around best_r
    # We can run more restarts for the final best_r to ensure we have a good config
    # Or just accept the one from the last successful coarse step
    
    # Let's try to polish the best_r with more restarts
    success, pos = _try_pack(best_r, n=n, restarts=50, steps=1000, lr=0.05)
    if success:
        best_pos = pos
        
    if best_pos is None:
        # Fallback to grid if something goes wrong
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:n])
        
    # Construct output array
    result = np.zeros((n, 3))
    result[:, 0] = best_pos[:, 0]
    result[:, 1] = best_pos[:, 1]
    result[:, 2] = best_r
    
    return result

