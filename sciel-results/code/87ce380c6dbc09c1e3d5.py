import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using Basin Hopping 
    initialized from BCC lattice fragments.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    # Parameters for the Dzugutov potential as defined in the task
    # These are hardcoded based on the task description, but we can also 
    # assume the 'relax' and 'gradient' functions handle the potential parameters 
    # internally if we just pass the potential name. 
    # However, for initialization, we need a reasonable spacing.
    # Typical nearest neighbor distance for Dzugutov is around 1.8-1.9.
    
    # Time budget allocation
    # Total limit 300s. 6 instances. Let's aim for 45s per instance to be safe.
    time_limit_per_instance = 45.0
    start_time = time.time()
    
    def generate_bcc_initial(n):
        """Generate an initial cluster by cutting a BCC lattice."""
        # BCC lattice constant a. Nearest neighbor distance r = sqrt(3)/2 * a.
        # For Dzugutov, optimal r is around 1.87 (from 'a' parameter in potential definition? 
        # Actually V(r) has singularity at a=1.87. So atoms should be > 1.87.
        # Let's assume equilibrium distance is around 1.9-2.0.
        # If r_eq ~ 1.9, then a = 1.9 * 2 / sqrt(3) ~ 2.2.
        a_lattice = 2.2 
        # Generate BCC points in a large box
        size_box = int(np.ceil(n ** (1/3) * a_lattice * 2))
        points = []
        # Simple BCC generation
        for i in range(-size_box, size_box+1):
            for j in range(-size_box, size_box+1):
                for k in range(-size_box, size_box+1):
                    # Corners
                    points.append([i*a_lattice, j*a_lattice, k*a_lattice])
                    # Body center
                    points.append([(i+0.5)*a_lattice, (j+0.5)*a_lattice, (k+0.5)*a_lattice])
        
        points = np.array(points)
        
        # Center of mass
        com = np.mean(points, axis=0)
        points = points - com
        
        # Sort by distance from center to pick the most spherical cluster
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        selected = points[indices[:n]]
        
        # Re-center
        selected = selected - np.mean(selected, axis=0)
        
        return selected

    def local_minimize(positions):
        """Perform local minimization using L-BFGS-B."""
        def energy_and_grad(x):
            # Reshape to (n, 3)
            pos = x.reshape(-1, 3)
            # Check constraints? The potential might handle it, but let's be safe.
            # The task says atoms must be at least 0.5 apart.
            # L-BFGS-B can handle bounds, but not pairwise distances directly.
            # We rely on the gradient to push them apart if they get too close, 
            # or we add a penalty.
            # Let's assume the potential goes to infinity or very high energy if r < a=1.87?
            # No, V(r) is defined for r > a? 
            # "V(r) = ... [r < a] + ... [r < b]"
            # Wait, the formula is:
            # V(r) = A(r^-m - B)exp(c/(r - a)) for r < a ??
            # Usually Dzugutov is defined for r > a. Let's re-read carefully.
            # "V(r) = A(r^-m - B)exp(c/(r - a)) [r < a]" -- This usually means the term is active or defined?
            # Standard Dzugutov: V(r) = A(r^-m - B) exp(c/(r-a)) for r > a, 0 otherwise?
            # Or is it a repulsive core?
            # Actually, standard Dzugutov potential is:
            # V(r) = A(r^{-m} - B) exp(c/(r-a)) for r > a
            # V(r) = B exp(d/(r-b)) for r < b (with b > a) ?? No, usually b < a for repulsion?
            # Let's look at the parameters: a=1.87, b=1.94.
            # If r < a (1.87), we use the first term?
            # If r < b (1.94), we use the second term?
            # This implies overlapping regions?
            # Let's check the standard definition.
            # Dzugutov (1991): V(r) = A(r^{-m} - B) exp(c/(r-a)) for r > a, 0 for r <= a.
            # But here we have two terms.
            # Let's assume the `relax` function handles the exact potential calculation.
            # We just need to provide gradients.
            
            e, grad = dzugutov_energy_gradient(pos, potential)
            # Flatten gradient
            return e, grad.flatten()
            
        x0 = positions.flatten()
        res = minimize(
            fun=energy_and_grad,
            x0=x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
        )
        return res.fun, res.x.reshape(-1, 3)

    def basin_hopping_trajectory(seed):
        """Run a single basin hopping trajectory."""
        rng = np.random.default_rng(seed)
        
        # Initial structure from BCC
        x = generate_bcc_initial(n)
        # Add small random noise to break symmetry
        x += rng.normal(0, 0.05, x.shape)
        
        # Local minimize first
        e_curr, x_curr = local_minimize(x)
        
        best_e = e_curr
        best_x = x_curr.copy()
        
        # Basin Hopping parameters
        step_size = 0.5
        n_iter = 50 # Max iterations per trajectory
        
        for i in range(n_iter):
            # Check time
            if time.time() - start_time > time_limit_per_instance - 5:
                break
                
            # Generate move
            x_trial = x_curr + rng.normal(0, step_size, x_curr.shape)
            
            # Local minimize
            try:
                e_trial, x_trial_min = local_minimize(x_trial)
            except Exception:
                continue
                
            # Accept/Reject (Metropolis or just keep best? 
            # For finding global min, simple "keep if better" is okay for short runs, 
            # but Metropolis helps escape. Let's use simple acceptance for speed and reliability 
            # of finding *a* good min, as we are running multiple threads.
            # Actually, standard BH uses Metropolis. 
            # But since we have multiple independent threads, we can just do "hill climbing" 
            # (only accept if better) for each thread? No, that gets stuck.
            # Let's use a simple Metropolis with T=0.1.
            T = 0.1
            delta_e = e_trial - e_curr
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                e_curr, x_curr = e_trial, x_trial_min
                if e_curr < best_e:
                    best_e = e_curr
                    best_x = x_curr.copy()
            else:
                # Keep current best state for next step if rejected? 
                # Standard BH: x_curr stays the same? No, x_curr is the accepted state.
                # If rejected, we don't move.
                pass
                
        return best_e, best_x

    # Run multiple BH trajectories in parallel
    n_threads = 4
    best_energy = float('inf')
    best_positions = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(n_threads):
            # Different seeds for different threads
            seed = 42 + i * 1000
            futures.append(executor.submit(basin_hopping_trajectory, seed))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except Exception as e:
                pass
                
    # If no valid result was found (shouldn't happen), fallback to random relaxation
    if best_positions is None:
        # Fallback to seed-like behavior
        radius = 1.1 * n ** (1.0 / 3.0) * 0.75
        rng = np.random.default_rng(n)
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        _, best_positions = relax(np.array(points), potential)
        
    return best_positions

# EVOLVE-BLOCK-END
