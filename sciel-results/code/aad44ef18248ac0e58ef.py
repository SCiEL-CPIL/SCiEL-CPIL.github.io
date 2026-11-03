import time
import numpy as np

N = 21
DEADLINE = 140.0  # Safety margin under 150s wall-clock

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _relax_circles(circles, w, h, iterations=1000, tol=1e-9):
    """
    Deterministic relaxation to ensure strict validity.
    Used as a final polish step after physics simulation.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]

                if dist < min_dist:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True

        if not moved:
            break

    # Final boundary clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))
    
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    return circles

def _physics_optimize(pos, rad, w, h, t0, deadline):
    """
    Physics-based optimization with Simulated Annealing.
    pos: (N, 2) array of centers
    rad: (N,) array of radii
    """
    n = len(rad)
    # State: positions and radii
    p = pos.copy()
    r = rad.copy()
    
    # Parameters
    T_start = 0.05
    T_end = 0.0001
    alpha = 0.995
    T = T_start
    
    # Forces
    f_pos = np.zeros_like(p)
    f_rad = np.zeros_like(r)
    
    # Objective function to minimize: Overlap + Boundary Violation - Sum(Radii) * weight
    # We want to maximize sum(radii) subject to constraints.
    # Penalty method: Minimize E = sum(overlap_depth^2) + sum(boundary_violation^2) - lambda * sum(r)
    # We use lambda to drive expansion.
    
    lambda_expand = 1.0
    penalty_weight = 100.0
    
    best_score = -np.inf
    best_p = p.copy()
    best_r = r.copy()
    
    max_iters = 5000
    iter_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 2.0:
            break
            
        # Reset forces
        f_pos.fill(0)
        f_rad.fill(0)
        
        # 1. Boundary Forces
        for i in range(n):
            # Left
            if p[i, 0] < r[i]:
                f_pos[i, 0] += (r[i] - p[i, 0]) * penalty_weight
            # Right
            if p[i, 0] > w - r[i]:
                f_pos[i, 0] -= (p[i, 0] - (w - r[i])) * penalty_weight
            # Bottom
            if p[i, 1] < r[i]:
                f_pos[i, 1] += (r[i] - p[i, 1]) * penalty_weight
            # Top
            if p[i, 1] > h - r[i]:
                f_pos[i, 1] -= (p[i, 1] - (h - r[i])) * penalty_weight
            
            # Expansion Force
            f_rad[i] -= lambda_expand # Negative because we minimize, so this increases r
            
        # 2. Overlap Forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = p[j, 0] - p[i, 0]
                dy = p[j, 1] - p[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist < 1e-6:
                        angle = np.random.uniform(0, 2 * np.pi)
                        ux, uy = np.cos(angle), np.sin(angle)
                    else:
                        ux, uy = dx / dist, dy / dist
                    
                    # Force proportional to overlap
                    force_mag = overlap * penalty_weight
                    
                    f_pos[i, 0] -= ux * force_mag
                    f_pos[i, 1] -= uy * force_mag
                    f_pos[j, 0] += ux * force_mag
                    f_pos[j, 1] += uy * force_mag
                    
                    # If we are overlapping, we might want to shrink radii slightly to allow separation?
                    # No, usually we push positions. But if stuck, shrinking helps.
                    # Let's add a small negative force to radii if overlap is significant
                    f_rad[i] -= overlap * 10.0
                    f_rad[j] -= overlap * 10.0
        
        # Calculate Energy (for Metropolis criterion)
        # E = sum(overlap^2) + sum(boundary^2)
        energy = 0.0
        for i in range(n):
            if p[i, 0] < r[i]: energy += (r[i] - p[i, 0])**2
            if p[i, 0] > w - r[i]: energy += (p[i, 0] - (w - r[i]))**2
            if p[i, 1] < r[i]: energy += (r[i] - p[i, 1])**2
            if p[i, 1] > h - r[i]: energy += (p[i, 1] - (h - r[i]))**2
            for j in range(i + 1, n):
                dx = p[j, 0] - p[i, 0]
                dy = p[j, 1] - p[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    energy += (min_dist - dist)**2
        
        # Propose Move
        # Step size scales with T
        step_size = 0.01 * T * 100 # Scale up for visibility
        
        # Random perturbation
        dp = np.random.normal(0, step_size, size=p.shape)
        dr = np.random.normal(0, step_size * 0.5, size=r.shape)
        
        # Apply forces (gradient descent step)
        # We mix gradient descent with random walk
        grad_step = 0.01
        p_new = p + f_pos * grad_step + dp
        r_new = r + f_rad * grad_step + dr
        
        # Clamp radii to positive
        r_new = np.maximum(r_new, 1e-6)
        
        # Calculate New Energy
        energy_new = 0.0
        for i in range(n):
            if p_new[i, 0] < r_new[i]: energy_new += (r_new[i] - p_new[i, 0])**2
            if p_new[i, 0] > w - r_new[i]: energy_new += (p_new[i, 0] - (w - r_new[i]))**2
            if p_new[i, 1] < r_new[i]: energy_new += (r_new[i] - p_new[i, 1])**2
            if p_new[i, 1] > h - r_new[i]: energy_new += (p_new[i, 1] - (h - r_new[i]))**2
            for j in range(i + 1, n):
                dx = p_new[j, 0] - p_new[i, 0]
                dy = p_new[j, 1] - p_new[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r_new[i] + r_new[j]
                if dist < min_dist:
                    energy_new += (min_dist - dist)**2
        
        # Metropolis Criterion
        delta_e = energy_new - energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            p = p_new
            r = r_new
            energy = energy_new
            
            # Track best
            score = np.sum(r)
            if score > best_score:
                # Only update best if valid or close to valid? 
                # Actually, we want the best VALID configuration.
                # But during annealing, we might be invalid.
                # We store best valid state separately.
                pass
        else:
            # Revert
            pass
            
        # Cool down
        T *= alpha
        
        # Periodic Check for Best Valid State
        if it % 50 == 0:
             # Check if current state is valid
             # This is expensive, so we do it sparsely
             # Construct array
             circ = np.column_stack((p, r))
             if _validate(circ, w, h, tol=1e-5):
                 s = np.sum(r)
                 if s > best_score:
                     best_score = s
                     best_p = p.copy()
                     best_r = r.copy()
                     
    # Final Deterministic Relaxation on the best found state
    final_circles = np.column_stack((best_p, best_r))
    final_circles = _relax_circles(final_circles, w, h, iterations=200)
    
    if _validate(final_circles, w, h):
        return final_circles
    
    # If relaxation failed, try current state
    current_circles = np.column_stack((p, r))
    current_circles = _relax_circles(current_circles, w, h, iterations=200)
    if _validate(current_circles, w, h):
        return current_circles
        
    return None

def _make_grid(n, w, h, seed=0):
    """Generate a random or grid initial state."""
    rng = np.random.RandomState(seed)
    # Grid initialization is often better for packing
    side = int(np.ceil(np.sqrt(n)))
    pos = np.zeros((n, 2))
    rad = np.zeros(n)
    
    # Simple grid layout
    for i in range(n):
        row = i // side
        col = i % side
        # Add some randomness to break symmetry
        jitter = 0.1 * min(w, h)
        pos[i, 0] = (col + 0.5) * w / side + rng.uniform(-jitter, jitter)
        pos[i, 1] = (row + 0.5) * h / side + rng.uniform(-jitter, jitter)
        # Initial radius: small enough to fit
        r_max = min(w, h) / (2 * side) * 0.8
        rad[i] = rng.uniform(r_max * 0.5, r_max)
        
    # Ensure inside bounds
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    return pos, rad

def _solve_aspect(w, h, t0, seed=0):
    pos, rad = _make_grid(N, w, h, seed)
    
    # Run physics optimization
    result = _physics_optimize(pos, rad, w, h, t0, t0 + DEADLINE)
    
    if result is not None:
        return result
        
    # Fallback: just relax the initial grid
    initial_circles = np.column_stack((pos, rad))
    relaxed = _relax_circles(initial_circles, w, h, iterations=500)
    if _validate(relaxed, w, h):
        return relaxed
        
    return None

def construct():
    t0 = time.time()
    best = None
    best_score = -np.inf
    
    # Try a range of aspect ratios
    # Perimeter 4 => 2(w+h) = 4 => w+h=2
    # w = ar * h => h(ar+1) = 2 => h = 2/(ar+1), w = 2*ar/(ar+1)
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9]
    
    # Shuffle ratios to diversify early search
    np.random.shuffle(ratios)
    
    for idx, ar in enumerate(ratios):
        if time.time() - t0 > DEADLINE:
            break
            
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        # Try both orientations
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() - t0 > DEADLINE - 5.0:
                break
                
            # Use different seeds for each aspect ratio
            seed = idx * 100
            
            try:
                x = _solve_aspect(ww, hh, t0, seed)
            except Exception:
                continue
                
            if x is not None:
                s = _total_radius(x)
                if np.isfinite(s) and s > best_score:
                    best_score = s
                    best = (x.copy(), ww, hh)
                    
    if best is None:
        # Fallback to seed-like grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, w, h = best
    
    # Final safety: if somehow invalid, shrink radii slightly
    if not _validate(x, w, h):
        for i in range(N):
            x[i, 2] *= 0.999
        if not _validate(x, w, h):
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])
            
    return x
