import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the centers of 50 equal circles inside a unit circle.
    Uses Simulated Annealing with a fixed-radius feasibility check to maximize the radius.
    """
    
    def energy(c, r):
        """
        Calculate the total squared overlap for circles of radius r.
        Overlap is defined as the amount by which the distance is less than 2r (for pairs)
        or the amount by which the center is outside the boundary (1-r).
        """
        # Pairwise overlaps
        # d_ij < 2r -> overlap = 2r - d_ij
        # We sum (overlap)^2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount for pairs
        # If dist < 2r, overlap is 2r - dist, else 0
        # Using np.maximum(0, 2r - dists)
        pair_overlaps = np.maximum(0, 2*r - dists)
        # Square it
        pair_energy = np.sum(pair_overlaps**2)
        
        # Boundary overlaps
        norms = np.sqrt(np.sum(c**2, axis=1))
        # If norm > 1-r, overlap is norm - (1-r)
        boundary_overlaps = np.maximum(0, norms - (1 - r))
        boundary_energy = np.sum(boundary_overlaps**2)
        
        return pair_energy + boundary_energy

    def sa_optimize(c, r, t0, t1, steps, time_limit):
        """
        Run Simulated Annealing to minimize energy for a fixed radius r.
        Returns the best configuration found.
        """
        c = c.copy()
        e_curr = energy(c, r)
        c_best = c.copy()
        e_best = e_curr
        
        # Cooling schedule
        # T goes from t0 to t1 over 'steps'
        # T_k = t0 * (t1/t0)^(k/steps)
        
        start_time = time.time()
        
        for k in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            # Current temperature
            T = t0 * (t1 / t0) ** (k / steps)
            
            # Pick a random circle to move
            i = np.random.randint(N)
            
            # Original position
            old_pos = c[i].copy()
            
            # Random displacement
            # Scale displacement with temperature? Or fixed?
            # Fixed small displacement is often better for fine-tuning
            disp = np.random.normal(0, 0.01, 2)
            c[i] = old_pos + disp
            
            # Ensure we don't go wildly out of bounds in one step (optional, energy handles it)
            
            # Calculate new energy
            e_new = energy(c, r)
            
            # Metropolis criterion
            if e_new < e_curr:
                e_curr = e_new
                if e_new < e_best:
                    e_best = e_new
                    c_best = c.copy()
            else:
                delta_e = e_new - e_curr
                # Accept with probability exp(-delta_e / T)
                if T > 1e-10:
                    p = np.exp(-delta_e / T)
                    if np.random.random() < p:
                        e_curr = e_new
                    else:
                        # Reject move
                        c[i] = old_pos
                else:
                    # Reject move (T is effectively 0)
                    c[i] = old_pos
                    
        return c_best, e_best

    # Main Loop
    np.random.seed(42)
    
    # Initial guess for r
    # Best known is ~0.1258
    # Start lower to ensure feasibility
    r = 0.10
    
    # Initial positions: random in disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    c = np.column_stack((x, y))
    
    # Parameters for SA
    t0 = 1.0
    t1 = 1e-6
    steps_per_iter = 5000
    
    # Time management
    # We have 120s total.
    # Let's reserve some time for the final refinement.
    # We will iterate r, running SA for each.
    
    best_c = c.copy()
    best_r = 0.0
    
    # We will try to increase r in small increments
    # If we fail to find a feasible config, we keep the last best and stop increasing r,
    # but we might do more SA on the current r to improve the packing (reduce energy) 
    # which might allow a slightly larger r later? No, if it's infeasible, it's infeasible.
    # However, reducing energy for a fixed r doesn't change r.
    # So we only increase r if we find a feasible solution.
    
    # To be efficient, we can run SA for a fixed number of steps, check energy.
    # If energy < threshold, increase r.
    
    threshold = 1e-4
    
    # We will run for a total of ~100 seconds
    start_time = time.time()
    max_time = 110.0
    
    # We will perform a sequence of SA runs with increasing r.
    # Each run has a fixed number of steps.
    
    # Start with a rough initialization
    c_current = c.copy()
    
    # We can also do a few restarts if we get stuck, but SA with cooling should handle it.
    # Let's just run a long SA that gradually increases r.
    
    # Actually, a better strategy:
    # 1. Run SA to minimize energy for current r.
    # 2. If energy is low, increase r.
    # 3. Repeat.
    
    # Let's implement this loop.
    
    r = 0.10
    c_current = c.copy()
    
    # We will use a "batch" of SA steps.
    # If we find a feasible config, we lock it in and try a larger r.
    
    while time.time() - start_time < max_time:
        # Run SA for current r
        # Use a shorter cooling schedule for each increment?
        # Or keep the same T schedule but restart T if r increases?
        # Restarting T is better for exploration at new r.
        
        t0 = 0.5
        t1 = 1e-5
        steps = 2000
        
        c_best, e_best = sa_optimize(c_current, r, t0, t1, steps, max_time - (time.time() - start_time))
        
        if e_best < threshold:
            # Feasible
            best_c = c_best.copy()
            best_r = r
            c_current = c_best.copy()
            # Increase r
            r += 0.001
        else:
            # Not feasible
            # Keep the best configuration found for this r (which minimizes overlap)
            # and try to continue cooling or just break if we've tried many times?
            # Actually, if it's not feasible, increasing r won't help.
            # We should try to improve the packing for the *current* r?
            # No, if it's not feasible at r, it's not feasible at r+delta.
            # So we should stick to the best feasible r we found.
            # But maybe we got stuck in a local minimum for r.
            # Let's try a few more SA runs with different random seeds or higher T?
            # To keep it simple, if we fail, we break and return the best feasible r found so far.
            # However, we might have found a feasible r=0.12, then tried 0.121 and failed.
            # We should return the config for 0.12.
            # But maybe 0.121 is feasible with a better start?
            # Let's do a few retries with higher T if we fail.
            
            # Simple heuristic: If we fail, break. The previous best is valid.
            # But we can try to "shake" the current best config and try again for the same r?
            # Or just accept that we found the local optimum for this r.
            
            # Let's try to continue cooling for the same r for a bit longer?
            # The sa_optimize function already ran for 'steps'.
            # Let's just break to ensure we return a valid result quickly.
            # However, we might have skipped a feasible r due to bad luck.
            # Let's do a binary search style? No, linear is fine for small increments.
            
            # To be safe, let's do a few restarts for the current r before giving up.
            # But time is limited.
            
            # Let's just break and return best_c.
            break

    # Final check: ensure best_c is valid and returns the best radius
    # If no feasible r was found (unlikely with r=0.1), return random
    if best_r == 0.0:
        # Fallback: return the initial random points (score will be low but valid)
        # Actually, r=0.1 should be feasible.
        best_c = c
        # Calculate actual r for return? No, the evaluator calculates r.
        # We just return centers.
        
    return best_c

# EVOLVE-BLOCK-END
