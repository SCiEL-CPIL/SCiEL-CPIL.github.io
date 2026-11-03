import numpy as np
import time

N = 50

def _compute_forces(pos, r):
    """
    Compute forces for Langevin dynamics.
    pos: (N, 2) array of positions.
    r: target radius.
    Returns: (N, 2) array of forces.
    """
    N = pos.shape[0]
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Overlap: d < 2r
    # Force magnitude k * (2r - d) for d < 2r
    # We use a soft potential V = 0.5 * k * (2r - d)^2 for d < 2r
    # F = -dV/dd * (diff/d) = k * (2r - d) * (diff/d)
    k = 10.0  # Stiffness
    
    overlap = np.maximum(0, 2*r - dists)
    
    # Force vector for each pair i,j
    # F_ij = k * overlap_ij * (diff_ij / dists_ij)
    # Sum over j for each i
    
    # diff is (N, N, 2)
    # dists is (N, N)
    # We need to broadcast dists to (N, N, 1)
    dists_b = dists[:, :, np.newaxis]
    
    # Unit vectors
    # Where dists is 0 (diagonal), we set force to 0
    # diff / dists_b
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists_b
    
    # Force magnitudes
    # F_mag = k * overlap
    F_mag = k * overlap
    
    # Total force on i from all j
    # F_i = sum_j F_mag_ij * unit_vecs_ij
    forces = np.sum(F_mag[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Boundary forces
    # Square [-0.5, 0.5]
    # If x > 0.5 - r, push left. If x < -0.5 + r, push right.
    # Soft wall: V = 0.5 * k * (penetration)^2
    # Penetration x = max(0, (0.5 - r) - |x|) ... wait
    # Inside condition: |x| <= 0.5 - r
    # Penetration px = max(0, |x| - (0.5 - r))
    # Force is towards center: -sign(x) * k * px
    
    # x coords
    x = pos[:, 0]
    y = pos[:, 1]
    
    limit = 0.5 - r
    
    px = np.maximum(0, np.abs(x) - limit)
    py = np.maximum(0, np.abs(y) - limit)
    
    Fx_wall = -np.sign(x) * k * px
    Fy_wall = -np.sign(y) * k * py
    
    forces[:, 0] += Fx_wall
    forces[:, 1] += Fy_wall
    
    return forces

def _run_langevin(r, steps=5000, seed=0):
    """
    Run Langevin dynamics to find a valid packing for radius r.
    Returns (pos, valid)
    """
    rng = np.random.default_rng(seed)
    
    # Initialize positions randomly inside the valid region
    # To avoid initial overlaps, we can use a grid + noise or just random
    # Random might have many overlaps, but MD should resolve them.
    # Let's use a grid initialization for better start
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    # Add small noise to break symmetry
    pos += rng.normal(0, 0.01, pos.shape)
    
    # Ensure inside bounds
    limit = 0.5 - r
    pos[:, 0] = np.clip(pos[:, 0], -limit, limit)
    pos[:, 1] = np.clip(pos[:, 1], -limit, limit)
    
    v = np.zeros_like(pos)
    dt = 0.001
    gamma = 0.1 # Damping
    T0 = 0.01 # Initial temperature
    T_end = 0.0001
    
    for i in range(steps):
        # Cooling schedule
        T = T0 * np.exp(-i / (steps * 0.5))
        
        F = _compute_forces(pos, r)
        
        # Langevin update
        # m dv/dt = F - gamma v + sqrt(2 * gamma * T) * noise
        # v_new = (v + dt * (F - gamma * v) / m) * (1 - gamma * dt / 2) + ... 
        # Simpler: Euler with damping
        # v += dt * (F - gamma * v)
        # v += sqrt(dt * T) * noise
        # pos += dt * v
        
        # Let's use a standard form:
        # v = v * (1 - gamma) + dt * F + sqrt(2 * T * dt * gamma) * noise
        # This is a common approximation.
        
        noise = rng.normal(0, 1, pos.shape)
        
        v = v * (1 - gamma) + dt * F + np.sqrt(2 * T * dt * gamma) * noise
        pos += dt * v
        
        # Hard constraint: Keep inside square to prevent instability
        # If they go outside, the force brings them back, but for numerical stability:
        # pos[:, 0] = np.clip(pos[:, 0], -0.5, 0.5)
        # pos[:, 1] = np.clip(pos[:, 1], -0.5, 0.5)
        # Actually, clipping can cause issues with velocity. Better to rely on forces.
        # But if they fly out, forces might be weak if we only check |x| > 0.5-r.
        # My force function handles |x| > 0.5-r.
        
    # Check validity
    # 1. Inside square
    if np.any(np.abs(pos) > 0.5 - r + 1e-9):
        return pos, False
        
    # 2. No overlap
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    
    if min_dist < 2 * r - 1e-9:
        return pos, False
        
    return pos, True

def construct():
    t_start = time.time()
    deadline = 110.0 # Safety margin
    
    # Best known solution radius is ~0.07137
    # Grid (8x8) radius is 0.0625
    # Let's binary search for the max valid radius
    
    r_low = 0.0625
    r_high = 0.072
    
    best_pos = None
    best_r = r_low
    
    # Initial grid solution
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_pos = grid[:N]
    
    attempts = 0
    while time.time() - t_start < deadline:
        if r_high - r_low < 1e-6:
            break
            
        r_mid = (r_low + r_high) / 2
        
        # Try a few seeds for this radius
        success = False
        pos_candidate = None
        
        # Number of restarts decreases as we approach the limit
        # or we can just fix it to 3-5
        for seed in range(5):
            if time.time() - t_start > deadline - 10:
                break
                
            pos, valid = _run_langevin(r_mid, steps=3000, seed=seed)
            
            if valid:
                success = True
                pos_candidate = pos
                break
        
        if success:
            best_r = r_mid
            best_pos = pos_candidate
            r_low = r_mid
        else:
            r_high = r_mid
            
        # If we are very close to the limit, break
        if time.time() - t_start > deadline - 5:
            break
            
    # Verify best_pos
    # If best_pos is None, return grid
    if best_pos is None:
        return grid[:N]
        
    return best_pos

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 50

def _compute_forces(pos, r):
    """
    Compute forces for Langevin dynamics.
    pos: (N, 2) array of positions.
    r: target radius.
    Returns: (N, 2) array of forces.
    """
    N = pos.shape[0]
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Overlap: d < 2r
    # Force magnitude k * (2r - d) for d < 2r
    # We use a soft potential V = 0.5 * k * (2r - d)^2 for d < 2r
    # F = -dV/dd * (diff/d) = k * (2r - d) * (diff/d)
    k = 10.0  # Stiffness
    
    overlap = np.maximum(0, 2*r - dists)
    
    # Force vector for each pair i,j
    # F_ij = k * overlap_ij * (diff_ij / dists_ij)
    # Sum over j for each i
    
    # diff is (N, N, 2)
    # dists is (N, N)
    # We need to broadcast dists to (N, N, 1)
    dists_b = dists[:, :, np.newaxis]
    
    # Unit vectors
    # Where dists is 0 (diagonal), we set force to 0
    # diff / dists_b
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists_b
    
    # Force magnitudes
    # F_mag = k * overlap
    F_mag = k * overlap
    
    # Total force on i from all j
    # F_i = sum_j F_mag_ij * unit_vecs_ij
    forces = np.sum(F_mag[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Boundary forces
    # Square [-0.5, 0.5]
    # If x > 0.5 - r, push left. If x < -0.5 + r, push right.
    # Soft wall: V = 0.5 * k * (penetration)^2
    # Penetration x = max(0, (0.5 - r) - |x|) ... wait
    # Inside condition: |x| <= 0.5 - r
    # Penetration px = max(0, |x| - (0.5 - r))
    # Force is towards center: -sign(x) * k * px
    
    # x coords
    x = pos[:, 0]
    y = pos[:, 1]
    
    limit = 0.5 - r
    
    px = np.maximum(0, np.abs(x) - limit)
    py = np.maximum(0, np.abs(y) - limit)
    
    Fx_wall = -np.sign(x) * k * px
    Fy_wall = -np.sign(y) * k * py
    
    forces[:, 0] += Fx_wall
    forces[:, 1] += Fy_wall
    
    return forces

def _run_langevin(r, steps=5000, seed=0):
    """
    Run Langevin dynamics to find a valid packing for radius r.
    Returns (pos, valid)
    """
    rng = np.random.default_rng(seed)
    
    # Initialize positions randomly inside the valid region
    # To avoid initial overlaps, we can use a grid + noise or just random
    # Random might have many overlaps, but MD should resolve them.
    # Let's use a grid initialization for better start
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    # Add small noise to break symmetry
    pos += rng.normal(0, 0.01, pos.shape)
    
    # Ensure inside bounds
    limit = 0.5 - r
    pos[:, 0] = np.clip(pos[:, 0], -limit, limit)
    pos[:, 1] = np.clip(pos[:, 1], -limit, limit)
    
    v = np.zeros_like(pos)
    dt = 0.001
    gamma = 0.1 # Damping
    T0 = 0.01 # Initial temperature
    T_end = 0.0001
    
    for i in range(steps):
        # Cooling schedule
        T = T0 * np.exp(-i / (steps * 0.5))
        
        F = _compute_forces(pos, r)
        
        # Langevin update
        # m dv/dt = F - gamma v + sqrt(2 * gamma * T) * noise
        # v_new = (v + dt * (F - gamma * v) / m) * (1 - gamma * dt / 2) + ... 
        # Simpler: Euler with damping
        # v += dt * (F - gamma * v)
        # v += sqrt(dt * T) * noise
        # pos += dt * v
        
        # Let's use a standard form:
        # v = v * (1 - gamma) + dt * F + sqrt(2 * T * dt * gamma) * noise
        # This is a common approximation.
        
        noise = rng.normal(0, 1, pos.shape)
        
        v = v * (1 - gamma) + dt * F + np.sqrt(2 * T * dt * gamma) * noise
        pos += dt * v
        
        # Hard constraint: Keep inside square to prevent instability
        # If they go outside, the force brings them back, but for numerical stability:
        # pos[:, 0] = np.clip(pos[:, 0], -0.5, 0.5)
        # pos[:, 1] = np.clip(pos[:, 1], -0.5, 0.5)
        # Actually, clipping can cause issues with velocity. Better to rely on forces.
        # But if they fly out, forces might be weak if we only check |x| > 0.5-r.
        # My force function handles |x| > 0.5-r.
        
    # Check validity
    # 1. Inside square
    if np.any(np.abs(pos) > 0.5 - r + 1e-9):
        return pos, False
        
    # 2. No overlap
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    
    if min_dist < 2 * r - 1e-9:
        return pos, False
        
    return pos, True

def construct():
    t_start = time.time()
    deadline = 110.0 # Safety margin
    
    # Best known solution radius is ~0.07137
    # Grid (8x8) radius is 0.0625
    # Let's binary search for the max valid radius
    
    r_low = 0.0625
    r_high = 0.072
    
    best_pos = None
    best_r = r_low
    
    # Initial grid solution
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_pos = grid[:N]
    
    attempts = 0
    while time.time() - t_start < deadline:
        if r_high - r_low < 1e-6:
            break
            
        r_mid = (r_low + r_high) / 2
        
        # Try a few seeds for this radius
        success = False
        pos_candidate = None
        
        # Number of restarts decreases as we approach the limit
        # or we can just fix it to 3-5
        for seed in range(5):
            if time.time() - t_start > deadline - 10:
                break
                
            pos, valid = _run_langevin(r_mid, steps=3000, seed=seed)
            
            if valid:
                success = True
                pos_candidate = pos
                break
        
        if success:
            best_r = r_mid
            best_pos = pos_candidate
            r_low = r_mid
        else:
            r_high = r_mid
            
        # If we are very close to the limit, break
        if time.time() - t_start > deadline - 5:
            break
            
    # Verify best_pos
    # If best_pos is None, return grid
    if best_pos is None:
        return grid[:N]
        
    return best_pos
# EVOLVE-BLOCK-END
