"""
Strategy: Basin Hopping with Poly-Tetrahedral Seeding.
1. Generate initial structures by packing tetrahedra (local order motif).
2. Use Basin Hopping (random perturbation + local minimization) to escape local minima.
3. Track best energy found and return the corresponding structure.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0 # Seconds to reserve for final checks and return

def _get_tetrahedron_vertices():
    """Return vertices of a regular tetrahedron centered at origin."""
    a = 1.0
    verts = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a]
    ])
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts

def _generate_polytetrahedral_seed(n, rng):
    """
    Generate a seed structure by packing tetrahedra.
    The Dzugutov potential favors local polytetrahedral order.
    We create a central atom and fill the space with tetrahedra.
    """
    if n < 4:
        return rng.uniform(-1, 1, (n, 3))
    
    tet_verts = _get_tetrahedron_vertices()
    # Scale tetrahedron to match typical interatomic distance in Dzugutov (approx 1.8-2.0)
    # The minimum of V(r) is around r ~ 2.0. Let's aim for edge length ~ 2.0.
    # Distance between vertices in unit tet is sqrt(2) * a. If a=1, dist=sqrt(2)~1.414.
    # We want dist ~ 1.9. Scale factor = 1.9 / 1.414 ~ 1.34.
    scale = 1.34
    tet_verts = tet_verts * scale
    
    points = []
    # Start with a central atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # We will add atoms in groups of 4 (tetrahedra) or 1 (if odd remainder)
    # To ensure density, we place tetrahedra in a roughly spherical shell.
    # For simplicity and robustness, we use a random placement strategy that respects min distance.
    
    min_dist = 0.9 # Minimum distance constraint for valid structure (task says >= 0.5, but 0.9 is safer for packing)
    max_attempts = 100
    radius_limit = 5.0
    
    while len(points) < n:
        # Try to place a tetrahedron
        # Pick a random center for the tetrahedron
        # Bias towards existing atoms to create a connected cluster
        if points:
            # Pick a random existing atom and place new tet near it
            anchor_idx = rng.integers(0, len(points))
            anchor = points[anchor_idx]
            # Random offset
            offset = rng.normal(0, 1.5, 3)
            tet_center = anchor + offset
        else:
            tet_center = rng.uniform(-2, 2, 3)
        
        # Random orientation
        # Generate random rotation matrix
        q = rng.normal(0, 1, 4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        new_verts = (tet_verts @ R.T) + tet_center
        
        # Check if we can add these vertices
        # We need to check all new points against all existing points
        valid = True
        for v in new_verts:
            if np.linalg.norm(v) > radius_limit:
                valid = False
                break
            for p in points:
                if np.linalg.norm(v - p) < min_dist:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Add as many vertices as needed
            for v in new_verts:
                if len(points) < n:
                    points.append(v)
        else:
            # If tetrahedron doesn't fit, try adding a single random point near the surface
            # This handles remainders and tight spaces
            if points:
                # Find a point on the convex hull or just random near surface
                # Simple heuristic: pick a random point and push it out if too close
                base = points[rng.integers(0, len(points))]
                dir = rng.normal(0, 1, 3)
                dir /= np.linalg.norm(dir)
                new_p = base + dir * min_dist * 1.1
                # Check distance
                can_add = True
                for p in points:
                    if np.linalg.norm(new_p - p) < min_dist:
                        can_add = False
                        break
                if can_add:
                    points.append(new_p)
            
            if max_attempts > 0:
                max_attempts -= 1
            else:
                # Fallback: random point
                points.append(rng.uniform(-radius_limit, radius_limit, 3))
                max_attempts = 100 # Reset attempts

    return np.array(points[:n])

def construct(instance):
    """
    Find low-energy structures using Basin Hopping.
    """
    n = instance['n']
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Set up RNG
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in evaluation if needed, or use system time
    # Using instance n as seed for diversity across instances
    rng = np.random.default_rng(n)
    
    best_energy = float('inf')
    best_positions = None
    
    # Number of independent basin hopping runs
    # We have ~290 seconds.
    # Each relax call is fast for n<60.
    # Let's aim for 10-20 runs, each with many steps.
    
    # Strategy:
    # 1. Generate a polytetrahedral seed.
    # 2. Relax it.
    # 3. Perform Basin Hopping steps:
    #    a. Perturb positions (random Gaussian noise).
    #    b. Relax the perturbed structure.
    #    c. Accept if energy is lower (or with Metropolis criterion, but for global min search, greedy is often better if steps are small, or simple random restarts).
    #    Let's use a simple random walk in the space of local minima.
    
    num_runs = 5 # 5 independent seeds
    steps_per_run = 50 # 50 basin hopping steps per seed
    
    # Pre-calculate max time per step
    # Total budget ~ 290s.
    # 5 * 50 = 250 relax calls.
    # ~1.1s per relax call is acceptable.
    
    for run_idx in range(num_runs):
        # Check time
        if time.time() > deadline:
            break
            
        # Generate seed
        try:
            seed_pos = _generate_polytetrahedral_seed(n, rng)
        except Exception:
            seed_pos = rng.uniform(-2, 2, (n, 3))
        
        # Initial relaxation
        try:
            energy, current_pos = relax(seed_pos, 'dzugutov')
        except Exception:
            continue
            
        if energy < best_energy:
            best_energy = energy
            best_positions = current_pos.copy()
            
        # Basin Hopping Loop
        for step in range(steps_per_run):
            if time.time() > deadline:
                break
                
            # Perturb
            # Scale perturbation with current temperature or just fixed
            # Dzugutov potential has features at scale ~1.0
            # Perturb by ~0.1-0.5
            sigma = 0.2 
            perturbed = current_pos + rng.normal(0, sigma, (n, 3))
            
            # Ensure minimum distance constraint is not grossly violated before relax
            # relax should handle it, but if points overlap, gradient might be unstable
            # The provided relax function likely handles this or returns inf/error.
            
            try:
                new_energy, new_pos = relax(perturbed, 'dzugutov')
            except Exception:
                continue
            
            # Acceptance criterion:
            # For global minimum search, we want to explore.
            # If new_energy < current_energy, accept.
            # If new_energy > current_energy, accept with probability exp(-(delta)/T).
            # Since we are doing "Basin Hopping" to find *global* min, we should allow uphill moves to escape basins.
            # However, we track the *best* energy found so far.
            # A simple strategy: Always accept if better. If worse, accept with prob 0.1 (random walk).
            
            if new_energy < energy:
                energy = new_energy
                current_pos = new_pos
            else:
                # Random acceptance to explore
                if rng.random() < 0.1:
                    energy = new_energy
                    current_pos = new_pos
            
            # Update global best
            if energy < best_energy:
                best_energy = energy
                best_positions = current_pos.copy()
                
    # If no valid result, fallback to random
    if best_positions is None:
        pos = rng.uniform(-2, 2, (n, 3))
        try:
            _, best_positions = relax(pos, 'dzugutov')
        except Exception:
            best_positions = pos
            
    return best_positions
# EVOLVE-BLOCK-END
