"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Force-directed physics simulation with velocity-Verlet integration.
Circles repel each other and walls via soft-potential forces, naturally settling
into a dense, collision-free configuration that maximizes the sum of radii.
"""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(200):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _sum_radii(pos, radii):
    return np.sum(radii)

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(200):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _init_grid():
    """Grid initialization with corner bias."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[k] = 0.09
                k += 1
    if k < N:
        pos[k] = [0.5, 0.5]
        radii[k] = 0.01
        k += 1
    return pos, radii

def _init_corners():
    """Corner-biased initialization."""
    rng = np.random.RandomState(42)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)]
        radii[i] = 0.17
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        if max_r < 0.01:
            for attempt in range(300):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
    return pos, radii

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_simulate(pos, radii, dt=0.002, n_steps=2000, damping=0.95, 
                             repulsion_k=500.0, wall_k=1000.0, deadline=None):
    """
    Force-directed simulation using velocity-Verlet integration.
    Circles repel each other and walls via soft-potential forces.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    vel = np.zeros((N, 2))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _sum_radii(pos, radii)
    
    # Precompute force function
    def compute_forces(pos, radii):
        forces = np.zeros((N, 2))
        
        # Wall repulsion
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                penetration = -(x - r)
                forces[i, 0] += wall_k * penetration
            # Right wall
            if 1.0 - x - r < 0:
                penetration = -(1.0 - x - r)
                forces[i, 0] -= wall_k * penetration
            # Bottom wall
            if y - r < 0:
                penetration = -(y - r)
                forces[i, 1] += wall_k * penetration
            # Top wall
            if 1.0 - y - r < 0:
                penetration = -(1.0 - y - r)
                forces[i, 1] -= wall_k * penetration
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r:
                    penetration = sum_r - dist
                    # Force magnitude proportional to penetration
                    f_mag = repulsion_k * penetration
                    # Unit vector from j to i
                    if dist > 1e-6:
                        fx = f_mag * dx / dist
                        fy = f_mag * dy / dist
                    else:
                        # Random direction if overlapping at same point
                        angle = np.random.uniform(0, 2 * np.pi)
                        fx = f_mag * np.cos(angle)
                        fy = f_mag * np.sin(angle)
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        return forces
    
    # Initial forces
    forces = compute_forces(pos, radii)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        # Velocity-Verlet integration
        # Update position
        pos = pos + vel * dt + 0.5 * forces * dt * dt
        
        # Compute new forces
        new_forces = compute_forces(pos, radii)
        
        # Update velocity
        vel = (vel + 0.5 * (forces + new_forces) * dt) * damping
        
        # Damping
        vel *= damping
        
        # Clip positions to stay within bounds (soft constraint)
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], radii[i] * 0.5, 1.0 - radii[i] * 0.5)
            pos[i, 1] = np.clip(pos[i, 1], radii[i] * 0.5, 1.0 - radii[i] * 0.5)
        
        # Check if valid and update best
        if _is_valid(pos, radii, tol=1e-6):
            score = _sum_radii(pos, radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        forces = new_forces
    
    return best_pos, best_radii

def _grow_all(pos, radii, n_passes=5):
    """Iteratively grow all circles to their maximum."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(n_passes):
        improved = False
        for i in range(N):
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(N):
                if j == i:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            new_r = max(0.005, max_r)
            if new_r > radii[i] * (1 + 1e-6):
                radii[i] = new_r
                improved = True
        if not improved:
            break
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate initializations
    inits = []
    pos_hex, rad_hex = _init_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_grid, rad_grid = _init_grid()
    inits.append((pos_grid, rad_grid))
    
    pos_cor, rad_cor = _init_corners()
    inits.append((pos_cor, rad_cor))
    
    pos_r1, rad_r1 = _init_random(100)
    inits.append((pos_r1, rad_r1))
    
    pos_r2, rad_r2 = _init_random(200)
    inits.append((pos_r2, rad_r2))
    
    best_pos = pos_hex.copy()
    best_radii = rad_hex.copy()
    best_score = _sum_radii(best_pos, best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Phase 1: Force-directed simulation
        pos_fd, rad_fd = _force_directed_simulate(
            pos_init, rad_init, 
            dt=0.002, n_steps=1500, 
            damping=0.95, 
            repulsion_k=500.0, 
            wall_k=1000.0,
            deadline=deadline
        )
        
        # Phase 2: Grow circles to maximum
        pos_grown, rad_grown = _grow_all(pos_fd, rad_fd, n_passes=10)
        
        # Phase 3: Post-process for strict validity
        pos_pp, rad_pp = _post_process(pos_grown, rad_grown)
        
        score = _sum_radii(pos_pp, rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
        
        # Phase 4: Additional force-directed refinement on the improved solution
        if time.time() < deadline - 5:
            pos_fd2, rad_fd2 = _force_directed_simulate(
                pos_pp, rad_pp,
                dt=0.001, n_steps=1000,
                damping=0.98,
                repulsion_k=300.0,
                wall_k=800.0,
                deadline=deadline
            )
            pos_grown2, rad_grown2 = _grow_all(pos_fd2, rad_fd2, n_passes=10)
            pos_pp2, rad_pp2 = _post_process(pos_grown2, rad_grown2)
            
            score2 = _sum_radii(pos_pp2, rad_pp2)
            if _is_valid(pos_pp2, rad_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = rad_pp2.copy()
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback: grid
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
