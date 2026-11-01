import numpy as np
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_result = None
    best_score = -np.inf
    
    def is_valid(xy, r, tol=1e-9):
        n = len(r)
        # Boundary check
        if np.any(r < -tol): return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol): return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol): return False
        
        # Overlap check
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def get_score(xy, r):
        return np.sum(r)

    def simulate(xy, r, n_steps=200, dt=0.05, temp=0.01):
        """
        Lagrangian/Force-based simulation.
        We want to maximize sum(r) subject to constraints.
        This is equivalent to minimizing -sum(r) subject to constraints.
        
        Forces:
        1. Growth force on radii: dr/dt = +1 (we want to grow)
        2. Repulsive force on centers if overlapping: push apart.
        3. Repulsive force on centers if hitting boundary: push inward.
        
        We use a simple Verlet-like or Euler integration with damping.
        """
        xy = xy.copy()
        r = r.copy()
        n = len(r)
        
        # Initialize velocities to zero
        vxy = np.zeros_like(xy)
        vr = np.zeros_like(r)
        
        # Friction/Damping to settle
        damping = 0.9
        
        for step in range(n_steps):
            # Calculate forces
            fxy = np.zeros_like(xy)
            fr = np.ones_like(r) # Growth force
            
            # Overlap constraints
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        # Overlap amount
                        overlap = min_dist - dist
                        # Force is proportional to overlap, direction is along vector
                        # We want to separate them.
                        # Force on i is towards -d, on j is towards +d
                        f_vec = np.array([dx, dy]) / dist
                        # Spring constant
                        k = 10.0
                        force_mag = k * overlap
                        fxy[i] -= f_vec * force_mag
                        fxy[j] += f_vec * force_mag
                    elif dist <= 1e-10:
                        # Random kick to break symmetry
                        angle = np.random.uniform(0, 2*np.pi)
                        fxy[i] += np.array([np.cos(angle), np.sin(angle)]) * 0.1
                        fxy[j] -= np.array([np.cos(angle), np.sin(angle)]) * 0.1
            
            # Boundary constraints
            for i in range(n):
                # Left
                if xy[i, 0] < r[i]:
                    fxy[i, 0] += 10.0 * (r[i] - xy[i, 0])
                # Right
                if xy[i, 0] > 1 - r[i]:
                    fxy[i, 0] -= 10.0 * (xy[i, 0] - (1 - r[i]))
                # Bottom
                if xy[i, 1] < r[i]:
                    fxy[i, 1] += 10.0 * (r[i] - xy[i, 1])
                # Top
                if xy[i, 1] > 1 - r[i]:
                    fxy[i, 1] -= 10.0 * (xy[i, 1] - (1 - r[i]))
                
                # If radius is too big to fit anywhere, penalize growth?
                # The boundary forces on xy will effectively limit r growth 
                # because if r grows, xy gets pushed to boundary, 
                # but we also need to ensure r doesn't exceed 0.5.
                if r[i] > 0.5:
                    fr[i] -= 10.0 * (r[i] - 0.5)

            # Integrate
            # v = v * damping + force * dt
            vxy = vxy * damping + fxy * dt
            vr = vr * damping + fr * dt
            
            # Update positions
            xy += vxy * dt
            r += vr * dt
            
            # Hard clamp radius to be positive and < 0.5
            r = np.clip(r, 0, 0.5)
            
            # Hard clamp positions to be within [0,1] roughly, 
            # though forces should keep them inside. 
            # If they go out, the forces will pull them back next step.
            # But to prevent numerical explosion:
            xy = np.clip(xy, -0.1, 1.1)
            
            # Add small noise to escape local minima (annealing-like)
            if temp > 0.001:
                xy += np.random.normal(0, temp, size=xy.shape)
                r += np.random.normal(0, temp*0.1, size=r.shape)
                r = np.clip(r, 0, 0.5)
                temp *= 0.995 # Cool down
        
        return xy, r

    def get_initial_config(restart, rng):
        n = N
        if restart == 0:
            # 6x6 grid
            side = 6
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.01
        elif restart == 1:
            # 4x4 grid + 16 in gaps
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            side = 4
            for i in range(16):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.2
            for i in range(16):
                row = i // 4
                col = i % 4
                xy[16+i] = [(col + 0.5)/4 + 0.125, (row + 0.5)/4 + 0.125]
                r[16+i] = 0.08
        elif restart == 2:
            # Random
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart == 3:
            # Hexagonal
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (1.0/12 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        else:
            # Random with larger radii
            xy = rng.uniform(0.15, 0.85, size=(n, 2))
            r = rng.uniform(0.08, 0.12, size=n)
        
        return xy, r

    rng = np.random.RandomState(42)
    
    # Main search loop
    n_restarts = 60
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        xy, r = get_initial_config(restart, rng)
        
        # Run simulation
        # We run multiple short simulations with different temps or just one long one
        xy_sim, r_sim = simulate(xy, r, n_steps=300, dt=0.05, temp=0.02)
        
        # Post-process: ensure validity
        # If invalid, shrink radii slightly until valid
        if not is_valid(xy_sim, r_sim):
             # Try to repair by shrinking
             for scale in np.arange(1.0, 0.5, -0.05):
                 r_test = r_sim * scale
                 if is_valid(xy_sim, r_test):
                     r_sim = r_test
                     break
             else:
                 # If still invalid, discard
                 continue
        
        score = get_score(xy_sim, r_sim)
        if score > best_score:
            best_score = score
            best_result = (xy_sim.copy(), r_sim.copy())
            
        # Refine the best result found so far with a very low temp simulation
        if best_result is not None and restart % 10 == 0:
             if time.time() < deadline - 10:
                 xy_ref, r_ref = simulate(best_result[0], best_result[1], n_steps=100, dt=0.02, temp=0.001)
                 if not is_valid(xy_ref, r_ref):
                     for scale in np.arange(1.0, 0.5, -0.05):
                         r_test = r_ref * scale
                         if is_valid(xy_ref, r_test):
                             r_ref = r_test
                             break
                     else:
                         continue
                 ref_score = get_score(xy_ref, r_ref)
                 if ref_score > best_score:
                     best_score = ref_score
                     best_result = (xy_ref.copy(), r_ref.copy())

    if best_result is None:
        # Fallback grid
        side = 6
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            r[i] = 0.5/side - 0.01
        return np.column_stack([xy, r])

    # Final check
    xy, r = best_result
    if not is_valid(xy, r):
         # Last resort repair
         for scale in np.arange(1.0, 0.1, -0.01):
             r_test = r * scale
             if is_valid(xy, r_test):
                 r = r_test
                 break
             
    return np.column_stack([xy, r])
