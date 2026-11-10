import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def construct(instance):
    """
    Finds the global minimum of a Sutton-Chen cluster using Multi-start Simulated Annealing.
    """
    n = instance['n']
    kind = instance['kind']
    
    # Parameters for the search
    # Number of restarts depends on cluster size
    # Smaller clusters can afford more restarts, larger ones need better initialization
    n_restarts = 15 if n < 40 else 8
    
    best_energy = np.inf
    best_positions = None
    
    # Estimate a reasonable bounding box radius
    # Typical cluster radius scales as N^(1/3)
    # For SC potential, equilibrium distance is ~1.0 (in reduced units where lattice constant is 1? 
    # The task says "reduced units (lattice constant 1)". 
    # The seed uses spacing 0.72. Let's stick to a similar scale.
    # The potential is r^-12 - c sqrt(r^-6). Min is at r ~ 1.0 if c is scaled appropriately?
    # Actually, for SC, the equilibrium distance depends on the specific parameters.
    # The seed uses 0.72. Let's use a radius that ensures the cluster fits.
    radius = 0.8 * n ** (1.0 / 3.0)
    
    rng = np.random.default_rng(instance['n']) # Seed by N for reproducibility in evaluation
    
    for _ in range(n_restarts):
        # 1. Generate initial random cluster
        points = []
        attempts = 0
        while len(points) < n and attempts < 10000:
            attempts += 1
            # Sample from a sphere
            vec = rng.normal(size=3)
            norm = np.linalg.norm(vec)
            if norm == 0: continue
            # Scale to radius with a bias towards center? 
            # Uniform in sphere:
            r = radius * rng.random() ** (1.0/3.0)
            p = vec / norm * r
            
            # Check minimum distance
            if len(points) == 0:
                points.append(p)
            else:
                # Vectorized distance check
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) > 0.4: # Minimum distance to avoid singularity
                    points.append(p)
        
        if len(points) < n:
            # Fallback: just place remaining randomly, might be invalid but relax will fix if not too close
            # Better: retry with larger radius
            continue
            
        points = np.array(points)
        
        # 2. Simulated Annealing
        # We run a short SA to escape local minima
        # Initial temperature
        T = 1.0
        cooling_rate = 0.99
        steps = 1000 if n < 50 else 500
        
        current_energy, _ = sutton_chen_energy_gradient(points, kind)
        if not np.isfinite(current_energy):
            continue
            
        for step in range(steps):
            # Choose a random atom to move
            idx = rng.integers(0, n)
            # Propose move: Gaussian noise scaled by T
            move = rng.normal(scale=0.1 * T, size=3)
            new_points = points.copy()
            new_points[idx] += move
            
            new_energy, _ = sutton_chen_energy_gradient(new_points, kind)
            
            if not np.isfinite(new_energy):
                continue
                
            delta_e = new_energy - current_energy
            
            # Acceptance criterion
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                points = new_points
                current_energy = new_energy
                
            T *= cooling_rate
            if T < 0.01:
                break
                
        # 3. Final Relaxation
        try:
            energy, relaxed_points = relax(points, 'sutton_chen', kind)
        except Exception:
            continue
            
        if energy < best_energy and np.isfinite(energy):
            best_energy = energy
            best_positions = relaxed_points
            
    if best_positions is None:
        # Fallback to seed-like behavior if all failed
        # This should rarely happen
        return np.random.default_rng(n).normal(size=(n, 3)) * 0.5
        
    return best_positions
