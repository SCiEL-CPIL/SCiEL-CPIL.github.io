"""Physics-based circle packing using Position Based Dynamics (PBD)."""
import numpy as np
import time

N = 26


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # --- Initial configuration: hexagonal packing in unit square ---
    def hex_initial():
        """Generate a hexagonal grid of circles inside [0,1]^2."""
        # Try different hexagonal spacings to get ~N circles
        best_config = None
        best_count = 0
        
        for r in np.linspace(0.03, 0.15, 50):
            dx = 2 * r * 1.001  # slight gap
            dy = dx * np.sqrt(3) / 2
            
            positions = []
            row = 0
            while True:
                y = r + dy * row
                if y > 1 - r:
                    break
                col = 0
                x_offset = 0 if row % 2 == 0 else dx / 2
                while True:
                    x = r + x_offset + dx * col
                    if x > 1 - r:
                        break
                    positions.append((x, y, r))
                    col += 1
                row += 1
            
            if len(positions) > best_count:
                best_count = len(positions)
                best_config = positions[:N]
        
        if best_config is None or len(best_config) < N:
            # Fallback: grid
            side = int(np.ceil(np.sqrt(N)))
            best_config = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                          for i in range(side) for j in range(side)][:N]
        
        return np.array(best_config)

    # --- PBD solver ---
    def pbd_solve(config, max_iters=5000, dt=0.01, stiffness=1.0):
        """
        Solve circle packing using Position Based Dynamics.
        config: (N, 3) array of [x, y, r]
        """
        x = config[:, 0].copy()
        y = config[:, 1].copy()
        r = config[:, 2].copy()
        
        n = len(x)
        masses = r ** 2  # mass proportional to area
        
        # PBD parameters
        beta = 0.5  # damping
        eps = 1e-10
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Store previous positions for velocity computation
            x_prev = x.copy()
            y_prev = y.copy()
            
            # Gauss-Seidel relaxation for constraints
            for _ in range(3):  # multiple iterations per frame
                # 1. Boundary constraints: keep circles inside [0,1]^2
                for i in range(n):
                    # Left boundary
                    if x[i] < r[i]:
                        x[i] = r[i]
                    # Right boundary
                    if x[i] > 1 - r[i]:
                        x[i] = 1 - r[i]
                    # Bottom boundary
                    if y[i] < r[i]:
                        y[i] = r[i]
                    # Top boundary
                    if y[i] > 1 - r[i]:
                        y[i] = 1 - r[i]
                
                # 2. Non-overlap constraints: for each pair, enforce distance >= r_i + r_j
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        dist = np.sqrt(dx * dx + dy * dy)
                        min_dist = r[i] + r[j]
                        
                        if dist < min_dist:
                            if dist < eps:
                                # Circles are coincident; push apart randomly
                                angle = np.random.uniform(0, 2 * np.pi)
                                dx = np.cos(angle)
                                dy = np.sin(angle)
                                dist = 1.0
                            else:
                                dx /= dist
                                dy /= dist
                            
                            # Overlap amount
                            overlap = min_dist - dist
                            
                            # Distribute correction based on inverse mass
                            wi = 1.0 / masses[i]
                            wj = 1.0 / masses[j]
                            wsum = wi + wj
                            
                            # Apply correction
                            x[i] -= overlap * (wi / wsum) * dx
                            y[i] -= overlap * (wi / wsum) * dy
                            x[j] += overlap * (wj / wsum) * dx
                            y[j] += overlap * (wj / wsum) * dy
            
            # Update velocities (for next iteration's prediction)
            vx = (x - x_prev) / dt
            vy = (y - y_prev) / dt
            
            # Apply damping to velocities
            vx *= beta
            vy *= beta
            
            # Predict next position using velocity
            x = x + vx * dt
            y = y + vy * dt
            
            # Re-apply boundary constraints after prediction
            for i in range(n):
                x[i] = np.clip(x[i], r[i], 1 - r[i])
                y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        return np.column_stack([x, y, r])

    # --- Multi-start with different initial configurations ---
    def random_initial(seed=None):
        """Generate a random initial configuration."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # Random radii
        r = rng.uniform(0.03, 0.12, N)
        
        # Random positions, trying to avoid immediate overlaps
        x = rng.uniform(r, 1 - r, N)
        y = rng.uniform(r, 1 - r, N)
        
        return np.column_stack([x, y, r])

    def evaluate(config):
        """Check if configuration is valid and compute score."""
        x, y, r = config[:, 0], config[:, 1], config[:, 2]
        
        # Check boundaries
        if np.any(x < -1e-6) or np.any(x > 1 + 1e-6):
            return 0.0
        if np.any(y < -1e-6) or np.any(y > 1 + 1e-6):
            return 0.0
        
        # Check non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - 1e-6:
                    return 0.0
        
        # Valid: score is sum of radii
        return np.sum(r)

    # --- Main search loop ---
    best_score = 0.0
    best_config = None
    
    # Start with hexagonal packing
    config = hex_initial()
    config = pbd_solve(config, max_iters=3000)
    score = evaluate(config)
    if score > best_score:
        best_score = score
        best_config = config.copy()
    
    # Try multiple random starts
    n_starts = 0
    while time.time() < deadline:
        seed = n_starts
        config = random_initial(seed)
        config = pbd_solve(config, max_iters=2000)
        score = evaluate(config)
        if score > best_score:
            best_score = score
            best_config = config.copy()
        n_starts += 1
    
    # Final polish: run more PBD iterations on best config
    if best_config is not None:
        polished = pbd_solve(best_config, max_iters=5000)
        score = evaluate(polished)
        if score > best_score:
            best_score = score
            best_config = polished
    
    if best_config is None:
        # Fallback
        best_config = hex_initial()
    
    # Ensure all circles are strictly inside [0,1]^2 with tiny margin
    margin = 1e-7
    best_config[:, 0] = np.clip(best_config[:, 0], best_config[:, 2] + margin, 1 - best_config[:, 2] - margin)
    best_config[:, 1] = np.clip(best_config[:, 1], best_config[:, 2] + margin, 1 - best_config[:, 2] - margin)
    
    return best_config
