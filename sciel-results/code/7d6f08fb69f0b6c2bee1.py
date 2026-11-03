"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _is_valid(params, N, tol=1e-6):
    """Check if configuration is valid."""
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    if np.any(r < -tol):
        return False
    
    for i in range(N):
        if x[i] < -tol or x[i] > 1 + tol:
            return False
        if y[i] < -tol or y[i] > 1 + tol:
            return False
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol:
            return False
        if 1 - x[i] - r[i] < -tol:
            return False
        if y[i] - r[i] < -tol:
            return False
        if 1 - y[i] - r[i] < -tol:
            return False
    
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True

def _physics_optimize(params, N, rng, max_time=25.0):
    """
    Physics-based optimization using Hertzian contact model.
    Circles repel when overlapping, and we try to grow radii.
    """
    start = time.time()
    x = params[:, 0].copy()
    y = params[:, 1].copy()
    r = params[:, 2].copy()
    
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_score = -np.inf
    
    if _is_valid(np.column_stack([x, y, r]), N):
        best_score = np.sum(r)
    
    # Physics parameters
    dt = 0.001
    k_repel = 100.0  # Repulsion stiffness
    k_boundary = 50.0  # Boundary stiffness
    growth_rate = 0.0001  # Rate at which radii grow
    damping = 0.99  # Velocity damping
    temperature = 0.01  # Initial random perturbation
    
    vx = np.zeros(N)
    vy = np.zeros(N)
    vr = np.zeros(N)
    
    max_steps = int(max_time / (dt * 1.1))  # Estimate max steps
    
    for step in range(max_steps):
        if time.time() - start > max_time:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Inter-circle repulsion (Hertzian: F ~ penetration^(3/2))
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist and dist > 1e-10:
                    penetration = min_dist - dist
                    # Hertzian force: proportional to penetration^(3/2)
                    force = k_repel * (penetration ** 1.5)
                    # Direction: push apart
                    nx = dx / dist
                    ny = dy / dist
                    fx[i] += force * nx
                    fy[i] += force * ny
                    fx[j] -= force * nx
                    fy[j] -= force * ny
                    # Also push radii to shrink slightly to reduce overlap
                    fr[i] -= force * 0.1
                    fr[j] -= force * 0.1
                elif dist < min_dist:
                    # Very close, add random kick
                    angle = rng.uniform(0, 2*np.pi)
                    kick = k_repel * 0.01
                    fx[i] += kick * np.cos(angle)
                    fy[i] += kick * np.sin(angle)
                    fx[j] -= kick * np.cos(angle)
                    fy[j] -= kick * np.sin(angle)
        
        # Boundary forces
        for i in range(N):
            # Left wall
            if x[i] - r[i] < 0:
                penetration = r[i] - x[i]
                force = k_boundary * (penetration ** 1.5)
                fx[i] += force
                fr[i] -= force * 0.1
            # Right wall
            if 1 - x[i] - r[i] < 0:
                penetration = r[i] + x[i] - 1
                force = k_boundary * (penetration ** 1.5)
                fx[i] -= force
                fr[i] -= force * 0.1
            # Bottom wall
            if y[i] - r[i] < 0:
                penetration = r[i] - y[i]
                force = k_boundary * (penetration ** 1.5)
                fy[i] += force
                fr[i] -= force * 0.1
            # Top wall
            if 1 - y[i] - r[i] < 0:
                penetration = r[i] + y[i] - 1
                force = k_boundary * (penetration ** 1.5)
                fy[i] -= force
                fr[i] -= force * 0.1
        
        # Growth force: try to increase radii
        for i in range(N):
            fr[i] += growth_rate
        
        # Update velocities with damping
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        vr = vr * damping + fr * dt
        
        # Update positions
        x += vx * dt
        y += vy * dt
        r += vr * dt
        
        # Clamp to valid ranges
        for i in range(N):
            r[i] = max(r[i], 0.005)
            x[i] = np.clip(x[i], 0.0, 1.0)
            y[i] = np.clip(y[i], 0.0, 1.0)
        
        # Add small random perturbation to escape local minima
        if step % 50 == 0:
            temp = temperature * (1.0 - step / max_steps)
            x += rng.normal(0, temp, size=N)
            y += rng.normal(0, temp, size=N)
        
        # Check if current state is valid
        current_params = np.column_stack([x, y, r])
        if _is_valid(current_params, N):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
    
    return np.column_stack([best_x, best_y, best_r])

def _grid_init(N, rng, radius_scale=1.0):
    """Grid initialization with perturbation."""
    side = int(np.ceil(np.sqrt(N)))
    params = np.zeros((N, 3))
    for k in range(N):
        row = k // side
        col = k % side
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        r = (0.5 / side) * 0.9 * radius_scale
        params[k] = [x, y, r]
    params += rng.uniform(-0.015, 0.015, size=(N, 3))
    params[:, 2] = np.maximum(params[:, 2], 0.01)
    return params

def _hex_init(N, rng, radius_scale=1.0):
    """Hexagonal initialization."""
    params = np.zeros((N, 3))
    side = int(np.ceil(np.sqrt(N)))
    spacing = 1.0 / (side + 0.3)
    
    count = 0
    for row in range(side + 1):
        for col in range(side + 1):
            if count >= N:
                break
            x = (col + 0.5 + (0.4 if row % 2 else 0)) * spacing
            y = (row + 0.5) * spacing
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            r = spacing * 0.45 * radius_scale
            params[count] = [x, y, r]
            count += 1
        if count >= N:
            break
    
    params += rng.uniform(-0.01, 0.01, size=(N, 3))
    params[:, 2] = np.maximum(params[:, 2], 0.01)
    return params

def _random_init(N, rng, radius_scale=1.0):
    """Random initialization."""
    params = np.zeros((N, 3))
    for i in range(N):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = 0.05 * radius_scale * rng.uniform(0.5, 1.5)
        params[i] = [x, y, r]
    return params

def construct():
    start_time = time.time()
    deadline = 140.0
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    n_restarts = 12
    
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        # Choose initialization
        if i % 3 == 0:
            params = _grid_init(N, rng, radius_scale=0.8 + 0.1*(i%4))
        elif i % 3 == 1:
            params = _hex_init(N, rng, radius_scale=0.8 + 0.1*(i%4))
        else:
            params = _random_init(N, rng, radius_scale=0.8 + 0.1*(i%4))
        
        # Optimize
        time_left = deadline - (time.time() - start_time)
        if time_left < 5:
            break
        
        optimized = _physics_optimize(params, N, rng, max_time=time_left * 0.65)
        
        if _is_valid(optimized, N):
            score = np.sum(optimized[:, 2])
            if score > best_score:
                best_score = score
                best_params = optimized.copy()
    
    # Fallback
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        best_params = np.array(cells[:N])
    
    return best_params
# EVOLVE-BLOCK-END
