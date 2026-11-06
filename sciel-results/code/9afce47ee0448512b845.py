import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax

# Constants for Dzugutov potential
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _get_icosahedron_vertices():
    """Returns 12 unit vectors pointing to the vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi),
        (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0),
        (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1),
        (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_seed(n, rng, r_inner, r_outer):
    """
    Generate a seed cluster with icosahedral core and random outer atoms.
    """
    if n <= 0:
        return np.array([])
    
    # 1. Core: 1 atom at origin
    points = [np.array([0.0, 0.0, 0.0])]
    
    # 2. Shell 1: 12 atoms (if n > 1)
    if n > 1:
        ico_verts = _get_icosahedron_vertices()
        # Scale by r_inner
        shell1 = ico_verts * r_inner
        # Add up to n-1 atoms
        num_shell1 = min(12, n - 1)
        points.extend(shell1[:num_shell1])
    
    # 3. Shell 2: Remaining atoms
    remaining = n - len(points)
    if remaining > 0:
        # Generate random points on a sphere of radius r_outer
        # Using uniform distribution on sphere
        # z = 1 - 2*u, theta = 2*pi*v
        u = rng.uniform(0, 1, remaining)
        v = rng.uniform(0, 1, remaining)
        z = 1 - 2 * u
        theta = 2 * np.pi * v
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        shell2 = np.column_stack((x, y, z)) * r_outer
        points.extend(shell2)
    
    return np.array(points)

def _solve_instance(instance, deadline):
    """
    Solve a single instance by trying multiple seeds and relaxing them.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # Parameters for seed generation
    # Typical nearest neighbor distance for Dzugutov is around 1.8-2.0
    # Inner shell radius ~ 1.8, Outer shell radius ~ 3.0-3.5
    r_inner_values = [1.8, 1.9, 2.0]
    r_outer_values = [3.0, 3.2, 3.5]
    
    # We will try a fixed number of seeds, or until deadline
    num_seeds = 30
    
    for i in range(num_seeds):
        if time.time() > deadline - 5: # Safety margin
            break
        
        rng = np.random.default_rng(i + n) # Seed based on instance and loop index
        
        # Vary radii
        r_in = r_inner_values[i % len(r_inner_values)]
        r_out = r_outer_values[i % len(r_outer_values)]
        
        # Add some noise to radii for diversity
        r_in *= (1.0 + 0.1 * rng.uniform(-1, 1))
        r_out *= (1.0 + 0.1 * rng.uniform(-1, 1))
        
        try:
            pos = _generate_seed(n, rng, r_in, r_out)
            # Relax
            energy, relaxed_pos = relax(pos, potential, instance.get("rho", instance.get("kind")))
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception as e:
            # If relaxation fails, skip
            pass

    # If no relaxation succeeded, return a random valid configuration
    if best_positions is None:
        rng = np.random.default_rng(12345)
        # Fallback: random points in a box, ensuring min distance
        # This is a weak fallback
        pos = np.zeros((n, 3))
        for i in range(n):
            while True:
                p = rng.uniform(-5, 5, 3)
                if all(np.linalg.norm(p - pos[j]) >= 0.5 for j in range(i)):
                    pos[i] = p
                    break
        best_positions = pos

    return best_positions

def construct(instance):
    """
    Entry point.
    """
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, leave 10s margin
    
    # The task says "called once for each n in (38, 45, 50, 55, 57, 60)"
    # But the interface is construct(instance) -> array.
    # The prompt says "all calls together within 300 s".
    # This implies the evaluator calls construct 6 times, and the *sum* of times is 300s?
    # OR does it mean the *program* runs for 300s total and handles all instances?
    # "called once for each n... all calls together within 300 s"
    # This usually means the total wall-clock time for the entire evaluation run is 300s.
    # However, the function signature is `construct(instance)`.
    # If the evaluator calls it 6 times sequentially, we need to manage the time across calls?
    # No, usually in these tasks, the "instance" contains the specific problem, and the function is called for that problem.
    # If the evaluator calls it 6 times, and we have a global 300s budget, we need to track time globally.
    
    # Let's assume a global time budget.
    # We need to store state between calls?
    # "Keep the function names and signatures... construct(instance)"
    # If we use a global variable to track time, it works if the module is imported once and called multiple times.
    
    global _global_start_time
    if not hasattr(construct, '_global_start_time'):
        construct._global_start_time = time.time()
    
    elapsed = time.time() - construct._global_start_time
    remaining = 300 - elapsed
    
    if remaining <= 0:
        # Time's up, return best effort (random)
        n = instance["n"]
        return np.random.default_rng(0).uniform(-5, 5, (n, 3))
    
    # If we are the first call, we have ~300s.
    # If we are the last call, we have less.
    # To be safe, we should divide the time among instances if possible, 
    # but we don't know which instance is being called first.
    # However, the prompt says "called once for each n... all calls together within 300s".
    # It's safer to assume we can use the remaining time for this instance.
    # But if we spend 290s on the first instance, we have 10s for the other 5.
    # This is risky.
    
    # Better strategy: Estimate time per instance.
    # If we assume 6 instances, we should aim for ~45s per instance.
    # But if we are called in parallel? No, "at most 4 threads" refers to internal threads.
    # The evaluator likely calls construct() sequentially.
    
    # Let's use a fixed time budget per call, say 45 seconds, to ensure we finish all 6 in 270s.
    # If we have more time, we can use it.
    # Let's cap the time for this specific call to 45 seconds.
    
    call_deadline = time.time() + 45
    
    # But if this is the last instance, we can use the remaining global time.
    # We don't know if it's the last.
    # Let's just use a generous but safe limit: min(remaining, 48)
    
    effective_deadline = time.time() + min(remaining - 5, 48)
    
    return _solve_instance(instance, effective_deadline)
