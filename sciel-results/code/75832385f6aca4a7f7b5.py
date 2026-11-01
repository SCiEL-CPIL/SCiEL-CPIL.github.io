import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s, safety margin

    best_result = None
    best_score = -np.inf

    def is_valid(result):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]

        if np.any(r < -1e-9):
            return False
        if np.any(xy[:, 0] - r < -1e-9) or np.any(1 - xy[:, 0] - r < -1e-9):
            return False
        if np.any(xy[:, 1] - r < -1e-9) or np.any(1 - xy[:, 1] - r < -1e-9):
            return False

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - 1e-9):
            return False

        return True

    def get_score(result):
        n = N
        r = result[2*n:]
        return np.sum(r)

    def push_apart(params, n_iters=100):
        """
        Simple relaxation to fix hard overlaps and boundary violations.
        Used to repair solutions that MD might leave slightly invalid.
        """
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()

        for _ in range(n_iters):
            moved = False

            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]

                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        wi = r[i] / (r[i] + r[j])
                        wj = r[j] / (r[i] + r[j])

                        push_x = (dx / dist) * overlap
                        push_y = (dy / dist) * overlap

                        xy[i, 0] -= push_x * wi
                        xy[i, 1] -= push_y * wi
                        xy[j, 0] += push_x * wj
                        xy[j, 1] += push_y * wj
                        moved = True
                    elif dist <= 1e-10:
                        angle = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.01 * np.cos(angle)
                        xy[i, 1] -= 0.01 * np.sin(angle)
                        xy[j, 0] += 0.01 * np.cos(angle)
                        xy[j, 1] += 0.01 * np.sin(angle)
                        moved = True

            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    moved = True
                if xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    moved = True
                if xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    moved = True

            if not moved:
                break

        return np.concatenate([xy.flatten(), r])

    def refine_solution(params, max_iter=200, tol=1e-12):
        """
        Standard L-BFGS polish using penalty function.
        Used at the end to tighten the solution.
        """
        n = N
        def penalty_function(p):
            xy = p[:2*n].reshape(n, 2)
            r = p[2*n:]
            
            d_left = xy[:, 0] - r
            d_right = 1 - xy[:, 0] - r
            d_bottom = xy[:, 1] - r
            d_top = 1 - xy[:, 1] - r
            boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                            np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)

            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            overlap = np.maximum(0, min_dist[iu] - dist[iu])
            overlap_viol = np.sum(overlap**2)

            return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

        try:
            result = minimize(penalty_function, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
            return result.x
        except Exception:
            return params

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
            # Hexagonal-ish
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart == 2:
            # Random
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        else:
            # Clustered
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                sx = (sub_i % 3) / 3.0 + 0.5/3
                sy = (sub_i // 3) / 3.0 + 0.5/3
                xy[i] = [cx + (sx - 0.5)*0.4, cy + (sy - 0.5)*0.4]
                r[i] = 0.06

        return np.concatenate([xy.flatten(), r])

    def md_simulate(params, n_steps=2000, dt=0.01, damping=0.99):
        """
        Molecular Dynamics simulation with soft repulsive potential.
        Potential: U(r) = k / r^6 for r < r_min, 0 otherwise.
        This creates a smooth force field that pushes circles apart.
        """
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        v = np.zeros_like(xy)
        
        # Constants
        k_repulse = 1.0
        r_cutoff = 0.15 # Distance at which repulsion becomes significant
        k_boundary = 10.0

        # We want to maximize sum(r). 
        # MD naturally minimizes energy. 
        # If we just simulate, circles will pack. 
        # To maximize radii, we can slowly increase radii during simulation? 
        # Or simply start with small radii, let them pack, then try to grow them?
        # Actually, a common trick is to fix radii and just pack, then scale? 
        # But we want to optimize radii too.
        
        # Alternative: Use a potential that encourages larger radii?
        # No, standard MD packs circles of fixed radius.
        # Let's do this: 
        # 1. Start with small radii.
        # 2. Run MD to pack them.
        # 3. Gradually increase radii while continuing MD to find the limit.
        
        # However, the task asks for a single construct.
        # Let's try a "growth" strategy.
        # Start with r_i = 0.01.
        # Run MD for a few steps.
        # Increase r_i by small delta.
        # Repeat until no more growth or time limit.
        
        # This is a heuristic for the whole construct, not just inside MD.
        # Let's implement a "Grow and Pack" loop in the main driver.
        
        # For now, this function just runs MD for fixed radii to relax positions.
        for step in range(n_steps):
            force = np.zeros_like(xy)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    
                    # Effective radius sum
                    r_eff = r[i] + r[j]
                    
                    # Only interact if close
                    if dist < r_cutoff + r_eff:
                        # Soft potential: F ~ 1/d^7 (derivative of 1/d^6)
                        # We want force to push apart.
                        # U = k * (r_eff / dist)^6 ? No, that's always on.
                        # Let's use U = k / (dist - r_eff)^2 if dist < r_eff + margin?
                        # Simpler: U = k * max(0, r_eff - dist)^2 ? That's Hooke's law.
                        # Let's use a smooth repulsive force when dist < r_eff + 0.1
                        margin = 0.1
                        if dist < r_eff + margin:
                            # Force magnitude
                            # dU/dd = -2k(r_eff - d)/d^2 ? 
                            # Let's just use a strong spring force when overlapping
                            if dist < r_eff:
                                f_mag = k_repulse * (r_eff - dist) * 10.0
                            else:
                                # Soft repulsion to prevent getting stuck
                                f_mag = k_repulse * (r_eff + margin - dist) * 2.0
                            
                            if dist > 1e-10:
                                fx = f_mag * dx / dist
                                fy = f_mag * dy / dist
                            else:
                                fx = f_mag
                                fy = 0
                            
                            force[i, 0] -= fx
                            force[i, 1] -= fy
                            force[j, 0] += fx
                            force[j, 1] += fy
            
            # Boundary forces
            for i in range(n):
                if xy[i, 0] < r[i]:
                    force[i, 0] += k_boundary * (r[i] - xy[i, 0])
                elif xy[i, 0] > 1 - r[i]:
                    force[i, 0] -= k_boundary * (xy[i, 0] - (1 - r[i]))
                
                if xy[i, 1] < r[i]:
                    force[i, 1] += k_boundary * (r[i] - xy[i, 1])
                elif xy[i, 1] > 1 - r[i]:
                    force[i, 1] -= k_boundary * (xy[i, 1] - (1 - r[i]))
            
            # Update velocities and positions (Velocity Verlet or simple Euler)
            v += force * dt
            v *= damping
            xy += v * dt
            
            # Hard clip to box (safety)
            xy[:, 0] = np.clip(xy[:, 0], 0, 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, 1)
            
            # Optional: Hard clip to radius to ensure no boundary violation
            # This might cause energy spikes but ensures validity
            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    v[i, 0] = 0
                if xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    v[i, 0] = 0
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    v[i, 1] = 0
                if xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    v[i, 1] = 0

        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(42)
    
    # Strategy: 
    # 1. Generate diverse initial configs.
    # 2. For each config, start with small radii.
    # 3. Iteratively increase radii and run MD to pack.
    # 4. Polish with L-BFGS.
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        params = get_initial_config(restart, rng)
        
        # Initialize with small radii
        r_init = 0.02
        xy_init = params[:2*N].reshape(N, 2)
        r_cur = np.full(N, r_init)
        cur_params = np.concatenate([xy_init.flatten(), r_cur])
        
        # Growth loop
        max_growth_steps = 100
        growth_factor = 1.05 # 5% growth per step
        
        for g in range(max_growth_steps):
            if time.time() > deadline:
                break
                
            # Try to grow radii
            r_new = r_cur * growth_factor
            # Cap radii at 0.5
            r_new = np.clip(r_new, 0, 0.5)
            
            new_params = np.concatenate([xy_init.flatten(), r_new])
            
            # Run MD to pack
            # Use fewer steps as radii get larger (tighter packing)
            steps = 500 if g < 10 else 200
            relaxed_params = md_simulate(new_params, n_steps=steps, dt=0.005, damping=0.95)
            
            # Check if we can keep this growth
            # Simple heuristic: if the "stress" (overlap penalty) is low, keep it.
            # Or just check validity and score.
            # Since MD with hard clipping might still have overlaps, we check.
            
            # Let's verify validity of the relaxed params
            # If valid, update best.
            # If not, try to repair with push_apart.
            
            repaired = push_apart(relaxed_params, n_iters=50)
            
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
                
                # If we found a valid solution, we can try to grow more
                r_cur = r_new
                xy_init = repaired[:2*N].reshape(N, 2)
            else:
                # If invalid even after repair, stop growing for this restart
                # Maybe try smaller growth? For now, break.
                break
        
        # Final polish on the best found in this restart
        if best_result is not None:
            # Use the last valid state from this restart if it's better than global?
            # Actually, we update global best inside the loop.
            # Let's polish the current best_result if it's from this restart or just polish global best?
            # Polishing global best every restart is expensive.
            # Let's polish the local best of this restart.
            
            # We need to track local best
            # For simplicity, let's just polish the global best_result at the end of all restarts?
            # No, that's too late.
            
            # Let's refine the current candidate (repaired)
            refined = refine_solution(repaired, max_iter=100)
            if is_valid(refined):
                score = get_score(refined)
                if score > best_score:
                    best_score = score
                    best_result = refined.copy()

    # Final high-precision polish on the overall best
    if best_result is not None and time.time() < deadline - 5:
        polished = refine_solution(best_result, max_iter=500, tol=1e-14)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2)
        r = best_result[2*n:]
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

if __name__ == "__main__":
    res = construct()
    print(res)
