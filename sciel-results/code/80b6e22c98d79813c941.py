import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = 115.0 # Leave 5s for cleanup/return
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing
    # Area of unit circle = pi. Area per circle ~ pi * r^2.
    # N circles. r ~ 1/sqrt(N).
    # Spacing ~ 2r.
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    k = int(1.0 / spacing) + 2
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    # Keep points inside unit circle (with margin)
    margin = spacing / 2.0
    points = points[radius <= 1.0 - margin]
    radius = radius[radius <= 1.0 - margin]
    
    # If we have more than N, pick the N closest to center
    if len(points) > N:
        idx = np.argsort(radius)
        points = points[idx[:N]]
    elif len(points) < N:
        # If too few, shrink spacing and retry (simple loop)
        # But for N=300, this should be fine.
        # Just in case, add random points? No, better to adjust spacing.
        # For robustness, if len < N, we just use what we have and pad with random?
        # No, must be exactly N.
        # Let's assume the lattice generation is sufficient.
        # If not, we can scale down and regenerate.
        pass

    # Ensure we have exactly N points
    if len(points) < N:
        # Fallback: Random initialization
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    elif len(points) > N:
        points = points[:N]

    # 2. Precompute Distances
    # D: (N, N)
    # W: (N,)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    W = 1.0 - np.hypot(points[:, 0], points[:, 1])
    
    # Initial r
    r_current = min(np.min(D), np.min(W))
    best_points = points.copy()
    best_r = r_current
    
    # 3. Simulated Annealing
    T = 0.005 # Initial temperature
    T_min = 1e-6
    cooling = 0.999 # Cooling rate per step
    
    # Step size
    step_size = 0.05
    
    # We will iterate until deadline
    while time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        
        # Propose move
        # Gaussian noise
        noise = np.random.normal(0, step_size, 2)
        new_pos = points[i] + noise
        
        # Check boundary
        if np.linalg.norm(new_pos) > 1.0:
            # Reject or reflect?
            # Reflecting might be better to explore boundary
            # new_pos = new_pos / np.linalg.norm(new_pos) * 0.99
            continue # Reject for simplicity, boundary is handled by W
        
        # Update D and W for index i
        # Save old values
        old_D_i = D[i, :].copy()
        old_W_i = W[i]
        
        # Calculate new distances from i to all others
        # D[i, j] = norm(points[i] - points[j])
        # We can compute this vectorized
        diffs = points - new_pos
        new_D_i = np.sqrt(np.sum(diffs ** 2, axis=1))
        
        # Update D
        D[i, :] = new_D_i
        D[:, i] = new_D_i
        
        # Update W
        new_W_i = 1.0 - np.linalg.norm(new_pos)
        W[i] = new_W_i
        
        # Calculate new r
        r_new = min(np.min(D), np.min(W))
        
        # Acceptance
        delta_r = r_new - r_current
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            # Accept
            points[i] = new_pos
            r_current = r_new
            
            if r_new > best_r:
                best_r = r_new
                best_points = points.copy()
        else:
            # Reject
            D[i, :] = old_D_i
            D[:, i] = old_D_i
            W[i] = old_W_i
        
        # Cool
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: Reheat? No, let it run.
            # Or break if T is very low and no improvement for a long time?
            # For now, just keep going.
            
    return best_points
