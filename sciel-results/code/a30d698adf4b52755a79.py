"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_packing(centers, radii):
    """
    Evaluate the packing. Returns (sum_radii, is_valid).
    """
    n = len(radii)
    total_r = np.sum(radii)
    
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < 0:
            return 0.0, False
        if x < r or x > 1.0 - r or y < r or y > 1.0 - r:
            return 0.0, False
            
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - 1e-12:
                return 0.0, False
                
    return total_r, True

def _grid_init(n_circles):
    """
    Initialize circles in a grid pattern.
    """
    side = int(np.ceil(np.sqrt(n_circles)))
    centers = []
    radii = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n_circles:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.4 / side
            centers.append([cx, cy])
            radii.append(r)
            idx += 1
            
    return np.array(centers), np.array(radii)

def _random_init(n_circles, seed=0):
    """
    Random initialization.
    """
    np.random.seed(seed)
    centers = np.random.rand(n_circles, 2) * 0.8 + 0.1
    radii = np.random.rand(n_circles) * 0.05 + 0.01
    return centers, radii

def _physics_relaxation(centers, radii, time_budget=100.0):
    """
    Physics-based relaxation:
    1. Circles are pressurized bubbles (force ~ r^2).
    2. Repulsive forces between overlapping circles.
    3. Boundary constraints (repulsive walls).
    4. Explicit integration with damping.
    """
    n = len(radii)
    centers = centers.copy()
    radii = radii.copy()
    
    # Initial velocity is zero
    velocities = np.zeros_like(centers)
    
    # Parameters
    dt = 0.005
    damping = 0.9
    pressure_coeff = 1.0
    repulsion_coeff = 50.0
    wall_coeff = 100.0
    
    start_time = time.time()
    
    # Adaptive dt: start small, increase if stable
    # We will run until time_budget is exhausted
    
    # To avoid immediate explosion, we might need a few "settling" steps with smaller dt
    # But explicit integration is fast.
    
    # We track the best valid configuration found during the process
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score = -np.inf
    
    # Check initial validity
    score, valid = _eval_packing(centers, radii)
    if valid:
        best_score = score
        best_centers = centers.copy()
        best_radii = radii.copy()
        
    step = 0
    while time.time() - start_time < time_budget:
        forces = np.zeros_like(centers)
        
        # 1. Pressure force (outward from center of mass? No, expansion is radial)
        # In a packing context, "pressure" usually means the circles want to grow.
        # But if they just grow, they hit boundaries and each other.
        # The equilibrium is where the sum of forces is zero.
        # Force from pressure: This is tricky. In gas dynamics, P * dV.
        # Here, we can model it as a force pushing the center away from neighbors?
        # No, the radius is the variable.
        # Let's treat this as a system where we optimize (x, y, r).
        # However, the prompt specifies "explicit time-stepping integrator" for "mechanical equilibrium".
        # Usually, this implies fixing the "mass" or "size" and moving centers, or moving everything.
        # If we allow r to change, we need a force on r.
        # Let's assume the "pressure" drives r to increase, balanced by repulsion.
        # Force on r_i: F_r = pressure_coeff * r_i^2 - sum(repulsions involving i)
        # This is getting complicated for a simple explicit loop.
        
        # Alternative interpretation:
        # The "bubble" metaphor often refers to the *positions* being adjusted by repulsive forces,
        # while the radii are fixed or adjusted by a separate mechanism.
        # However, the goal is to *maximize sum of radii*.
        # A common physics-based approach for packing is:
        # 1. Start with small circles.
        # 2. Apply a "growth" force that increases r.
        # 3. Apply repulsive forces that push centers apart when r increases.
        # 4. The system reaches a state where no further growth is possible without overlap.
        
        # Let's implement a simplified version:
        # Each step:
        # a. Calculate repulsive forces between circles based on current (x, y, r).
        # b. Calculate boundary forces.
        # c. Update centers using forces and velocity.
        # d. Grow radii slightly if there is "space" (approximated by low local pressure/force).
        
        # A more robust physics model for packing:
        # Treat circles as particles with mass m=1.
        # Potential energy U = sum_{i<j} (1/d_ij - 1/(r_i+r_j))^2 if d < r_i+r_j + margin
        # Plus boundary potential.
        # Plus a "negative" potential for radius to encourage growth?
        # Actually, let's stick to the prompt's "internal pressure proportional to square of radius".
        # This suggests F_pressure = k * r^2 acting to expand the circle.
        # How does this map to center movement?
        # If circle i expands, it pushes neighbors j.
        # The force on center i due to pressure is zero (symmetry), but it exerts force on neighbors.
        # The force on center j due to i is proportional to the overlap or proximity.
        
        # Let's use a standard repulsive force model that depends on the "desired" radius.
        # If we want to maximize r, we can simulate the process:
        # 1. All circles try to expand (dr/dt = growth_rate).
        # 2. If they overlap, a repulsive force pushes them apart.
        # 3. If they hit the wall, a force pushes them in.
        # 4. If the repulsive force is too high, growth stops or slows down.
        
        # Let's implement this logic.
        
        # Calculate forces
        for i in range(n):
            # Boundary forces
            # If x < r, force in +x direction
            if centers[i][0] < radii[i]:
                forces[i][0] += wall_coeff * (radii[i] - centers[i][0])
            elif centers[i][0] > 1.0 - radii[i]:
                forces[i][0] -= wall_coeff * (centers[i][0] - (1.0 - radii[i]))
                
            if centers[i][1] < radii[i]:
                forces[i][1] += wall_coeff * (radii[i] - centers[i][1])
            elif centers[i][1] > 1.0 - radii[i]:
                forces[i][1] -= wall_coeff * (centers[i][1] - (1.0 - radii[i]))
                
            # Interaction forces
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Avoid division by zero
                if dist < 1e-6:
                    # Jitter
                    forces[i][0] += np.random.rand() * 0.1
                    forces[i][1] += np.random.rand() * 0.1
                    forces[j][0] -= np.random.rand() * 0.1
                    forces[j][1] -= np.random.rand() * 0.1
                    continue
                    
                min_dist = radii[i] + radii[j]
                
                # Repulsive force if overlapping or close
                # Use a soft repulsion: F = k * (min_dist - dist) if dist < min_dist
                # To make it "pressurized", we can make the force depend on the "pressure" (r^2)
                # Let's use F = k * (r_i^2 + r_j^2) * (min_dist - dist) / dist * direction
                # This ensures larger circles push harder.
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Force magnitude
                    # Scale by pressure (r^2) to satisfy prompt
                    pressure_factor = radii[i]**2 + radii[j]**2
                    f_mag = repulsion_coeff * pressure_factor * overlap
                    f_x = f_mag * dx / dist
                    f_y = f_mag * dy / dist
                    forces[i][0] += f_x
                    forces[i][1] += f_y
                    forces[j][0] -= f_x
                    forces[j][1] -= f_y
                    # We could also allow slight separation if very close to prevent sticking
                    # But the growth will handle separation eventually if space exists.
                
        # Update velocities
        velocities += forces * dt
        velocities *= damping
        
        # Update centers
        centers += velocities * dt
        
        # Grow radii
        # Growth rate should be small.
        # If a circle is "stressed" (high force), it should grow slower.
        # Stress ~ magnitude of force on center.
        for i in range(n):
            stress = np.linalg.norm(forces[i])
            # Base growth
            growth = 0.001
            # Reduce growth if stressed
            growth *= np.exp(-stress * 0.1)
            radii[i] += growth
            
            # Cap radius to ensure it stays inside boundaries roughly
            # This is a soft cap, the forces will handle the rest
            # max_r = min(centers[i][0], 1-centers[i][0], centers[i][1], 1-centers[i][1])
            # radii[i] = min(radii[i], max_r * 0.99) # Too aggressive, prevents optimization
            
        # Periodic check for validity and best score
        if step % 10 == 0:
            score, valid = _eval_packing(centers, radii)
            if valid and score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_radii = radii.copy()
                
        step += 1
        
    return best_centers, best_radii

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin
    
    # Try multiple initializations
    best_centers = None
    best_radii = None
    best_score = -np.inf
    
    # 1. Grid Init
    centers_grid, radii_grid = _grid_init(N)
    # Run physics for a short time to see if it improves
    # We split the time budget among trials
    
    num_trials = 5
    time_per_trial = (deadline - (time.time() - start_time)) / num_trials
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
            
        if trial == 0:
            centers, radii = _grid_init(N)
        else:
            centers, radii = _random_init(N, seed=trial)
            
        # Run physics relaxation
        # Allocate remaining time minus a small buffer
        remaining_time = deadline - (time.time() - start_time) - 2.0
        if remaining_time < 1.0:
            break
            
        c, r = _physics_relaxation(centers, radii, time_budget=remaining_time)
        
        score, valid = _eval_packing(c, r)
        if valid and score > best_score:
            best_score = score
            best_centers = c
            best_radii = r
            
    # If no valid result, fall back to grid
    if best_centers is None:
        best_centers, best_radii = _grid_init(N)
        
    # Final cleanup: ensure strict validity
    # The physics simulation might leave tiny overlaps or boundary violations
    # We need to "shrink" or "move" circles to make them valid.
    
    # A simple post-processing:
    # 1. Clip centers to [r, 1-r]
    # 2. If overlap, reduce radius of the smaller circle.
    
    for i in range(N):
        x, y = best_centers[i]
        r = best_radii[i]
        
        # Boundary fix
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r:
            r = max_r
        best_radii[i] = r
        best_centers[i][0] = np.clip(x, r, 1.0 - r)
        best_centers[i][1] = np.clip(y, r, 1.0 - r)
        
    # Overlap resolution
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = best_centers[i][0] - best_centers[j][0]
                dy = best_centers[i][1] - best_centers[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = best_radii[i] + best_radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Reduce the smaller radius
                    if best_radii[i] < best_radii[j]:
                        best_radii[i] = max(0.001, best_radii[i] - overlap/2)
                    else:
                        best_radii[j] = max(0.001, best_radii[j] - overlap/2)
                    changed = True
        if not changed:
            break
            
    # Construct final array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i][0] = best_centers[i][0]
        result[i][1] = best_centers[i][1]
        result[i][2] = best_radii[i]
        
    return result
# EVOLVE-BLOCK-END
