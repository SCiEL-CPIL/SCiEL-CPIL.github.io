"""Circle packing in perimeter-4 rectangle: Force-Directed Dynamics + Simulated Annealing."""
import numpy as np
import time

N = 21
_IU = np.triu_indices(N, k=1)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    def evaluate(x, y, r, w):
        """Calculate sum of radii and validity."""
        h = 2.0 - w
        # Check bounds
        if np.any(r < 0) or np.any(x < 0) or np.any(x > w) or np.any(y < 0) or np.any(y > h):
            return -np.inf, False
        # Check overlap
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d = np.sqrt(dx*dx + dy*dy)
        rs = r[_IU[0]] + r[_IU[1]]
        if np.any(d < rs - 1e-9):
            return -np.inf, False
        return np.sum(r), True

    def get_forces(x, y, r, w):
        """
        Calculate forces on each circle.
        Returns Fx, Fy, Fr.
        Forces push circles apart and into the box to maximize radius.
        """
        h = 2.0 - w
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        Fr = np.zeros(N)
        
        # 1. Wall Forces (push inwards if too close to boundary)
        # We want to maximize r, so if a circle is close to a wall, 
        # the wall effectively limits r. 
        # Instead of a hard constraint, we use a soft repulsion from walls 
        # that acts as if the wall is a giant circle or just a boundary potential.
        
        # Distance to left wall: x - r
        # Distance to right wall: w - x - r
        # Distance to bottom wall: y - r
        # Distance to top wall: h - y - r
        
        d_left = x - r
        d_right = w - x - r
        d_bottom = y - r
        d_top = h - y - r
        
        # If distance < 0, it's overlapping. Apply force to push away.
        # We use a linear spring force for simplicity and stability.
        k_wall = 10.0
        
        # Left wall
        mask = d_left < 0
        Fx[mask] += k_wall * d_left[mask]  # d_left is negative, so force is negative? 
        # Wait, if x < r, we need to push x right (+) or r down (-).
        # Let's push x right. Force on x should be positive if x < r.
        # d_left = x - r. If x < r, d_left < 0.
        # We want Fx > 0. So Fx += -k * d_left.
        Fx[mask] = -k_wall * d_left[mask]
        
        # Right wall
        mask = d_right < 0
        # d_right = w - x - r. If x > w - r, d_right < 0.
        # We want Fx < 0. So Fx += -k * d_right.
        Fx[mask] -= k_wall * d_right[mask]
        
        # Bottom wall
        mask = d_bottom < 0
        Fy[mask] = -k_wall * d_bottom[mask]
        
        # Top wall
        mask = d_top < 0
        Fy[mask] -= k_wall * d_top[mask]
        
        # 2. Pairwise Repulsion
        # For each pair i, j:
        # If d_ij < r_i + r_j, they overlap.
        # Force pushes them apart along the vector connecting them.
        # Also, we want to increase r. 
        # A common heuristic: if overlapping, push centers apart and shrink radii slightly?
        # Or just push centers apart, and let a separate "growth" force expand radii.
        
        # Let's use a standard Lennard-Jones-like or simple repulsive force.
        # F = k_rep * (r_i + r_j - d_ij) if d_ij < r_i + r_j
        
        k_rep = 5.0
        k_grow = 0.1  # Constant force to grow radius
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-6:
                    # Random jitter to avoid singularity
                    dx = rng.normal(0, 0.01)
                    dy = rng.normal(0, 0.01)
                    d = np.sqrt(dx*dx + dy*dy)
                
                overlap = (r[i] + r[j]) - d
                if overlap > 0:
                    # Apply force
                    f = k_rep * overlap
                    fx = f * (dx / d)
                    fy = f * (dy / d)
                    
                    Fx[i] += fx
                    Fy[i] += fy
                    Fx[j] -= fx
                    Fy[j] -= fy
                    
                    # Optionally, shrink radii if they are too big?
                    # Fr[i] -= 0.5 * f
                    # Fr[j] -= 0.5 * f
                    # Instead, we will have a global growth force.
        
        # 3. Radius Growth Force
        # We want to increase r as much as possible.
        # Apply a constant positive force to Fr.
        Fr[:] += k_grow
        
        # 4. Damping / Velocity update is handled outside.
        return Fx, Fy, Fr

    def simulate_annealing(x0, y0, r0, w0, time_budget):
        """
        Run force-directed simulation with simulated annealing.
        """
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        w = w0
        
        vx = np.zeros(N)
        vy = np.zeros(N)
        vr = np.zeros(N)
        
        # Parameters
        dt = 0.05
        damping = 0.9
        k_temp = 1.0
        cool_rate = 0.9995
        
        end_time = time.time() + time_budget
        
        best_local_val = -np.inf
        best_local_state = (x.copy(), y.copy(), r.copy(), w)
        
        # Initial temperature
        T = k_temp
        
        while time.time() < end_time:
            # Calculate forces
            Fx, Fy, Fr = get_forces(x, y, r, w)
            
            # Update velocities
            vx = damping * vx + Fx * dt
            vy = damping * vy + Fy * dt
            vr = damping * vr + Fr * dt
            
            # Add thermal noise (annealing)
            noise = T * 0.1
            vx += rng.normal(0, noise, N)
            vy += rng.normal(0, noise, N)
            vr += rng.normal(0, noise * 0.5, N) # Less noise on radius
            
            # Update positions
            x += vx * dt
            y += vy * dt
            r += vr * dt
            
            # Enforce hard bounds to prevent divergence
            # x in [r, w-r]
            # y in [r, h-r]
            # r > 0
            # This projection ensures validity is maintained or restored quickly.
            
            h = 2.0 - w
            
            # Project r to positive
            r[r < 0.01] = 0.01
            
            # Project x, y based on current r and w
            # If x < r, set x = r
            # If x > w - r, set x = w - r
            # But if r is too big to fit (2*r > w), we must shrink r.
            max_r_possible = min(w, h) / 2.0
            r[r > max_r_possible] = max_r_possible
            
            # Now project x, y
            x = np.maximum(x, r)
            x = np.minimum(x, w - r)
            y = np.maximum(y, r)
            y = np.minimum(y, h - r)
            
            # Check validity and score
            val, is_valid = evaluate(x, y, r, w)
            
            if is_valid and val > best_local_val:
                best_local_val = val
                best_local_state = (x.copy(), y.copy(), r.copy(), w)
                
            # Cool down
            T *= cool_rate
            if T < 0.01:
                T = 0.01 # Keep some jitter to escape local minima
        
        return best_local_state, best_local_val

    def generate_initial_guesses():
        guesses = []
        
        # 1. Square-ish layouts
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2]:
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            # Simple grid initialization
            # 5x5 grid, skip corners to get 21
            idx = 0
            for row in range(5):
                for col in range(5):
                    if (row==0 and col==0) or (row==4 and col==4) or (row==0 and col==4) or (row==4 and col==0):
                        continue
                    cx = (col + 0.5) / 5.0 * w0
                    cy = (row + 0.5) / 5.0 * h0
                    r0 = min(w0/5.0, h0/5.0) * 0.4
                    x[idx] = cx
                    y[idx] = cy
                    r[idx] = r0
                    idx += 1
            
            guesses.append((x, y, r, w0))
            
        # 2. Random layouts
        for _ in range(10):
            w0 = rng.uniform(0.7, 1.3)
            h0 = 2.0 - w0
            x = rng.uniform(0.1, w0 - 0.1, N)
            y = rng.uniform(0.1, h0 - 0.1, N)
            r = np.full(N, 0.05)
            guesses.append((x, y, r, w0))
            
        return guesses

    guesses = generate_initial_guesses()
    
    # Allocate time budget
    # We have ~145s. 
    # Let's run each guess for a few seconds.
    # 15 guesses. 145/15 ~ 9s each.
    # But we should prioritize better initial guesses.
    # Let's just iterate.
    
    for gi, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        time_budget = min(8.0, deadline - (time.time() - start_time) - 5)
        if time_budget < 1.0:
            continue
            
        # Run simulation
        x_opt, y_opt, r_opt, w_opt, val = simulate_annealing(x0, y0, r0, w0, time_budget)
        
        # Final polish: try to increase radii slightly with small SLSQP or just force relax
        # Actually, the force simulation already grows radii.
        # Let's do a quick "relaxation" pass to ensure no overlaps and max radius.
        # Simple gradient ascent on sum(r) with penalty for overlap.
        
        # For now, just take the best state from annealing.
        if val > best_value:
            best_value = val
            best_result = np.column_stack((x_opt, y_opt, r_opt))

    # If we didn't find anything good, use a fallback
    if best_result is None or best_value <= 0:
        # Fallback: 5x5 grid in 1x1 square, r=0.1
        z = np.zeros(N * 3)
        for i in range(N):
            row = i // 5
            col = i % 5
            z[i*3] = (col + 0.5) * 0.2
            z[i*3+1] = (row + 0.5) * 0.2
            z[i*3+2] = 0.1
        best_result = z
        best_value = 0.1 * N

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Best value: {np.sum(res[:, 2])}")
    print(res)
