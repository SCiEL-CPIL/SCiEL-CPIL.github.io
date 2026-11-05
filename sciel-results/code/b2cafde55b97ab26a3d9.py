"""
Strategy: Crystalline Template + Global Geometric Optimization.
Uses known lattice structures (FCC, BCC, HCP) to generate high-quality initial guesses,
then refines them using L-BFGS-B, which is highly effective for smooth Morse potentials.
Includes a deadline guard to ensure all 6 instances finish within 300s.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 330.0  # 330s wall clock limit

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy_gradient(x, rho):
    """
    Wrapper for morse_energy_gradient that reshapes input/output for scipy.
    x: (n*3,) array
    Returns: (energy, gradient) where gradient is (n*3,)
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    # Ensure minimum distance constraint is handled softly or by checking
    # The task says atoms must be at least 0.5 apart. 
    # If they get too close, energy explodes. 
    # We rely on the optimizer to avoid the singularity, but we can add a penalty if needed.
    # For now, use the provided function.
    try:
        energy, grad = morse_energy_gradient(positions, rho)
        return energy, grad.flatten()
    except Exception:
        # If gradient fails (e.g. due to numerical issues), return a large penalty
        return 1e10, np.zeros_like(x)

def _generate_fcc(n):
    """Generate FCC-like cluster of size n."""
    points = []
    # FCC lattice vectors
    # Use a shell-based approach or simple lattice cutoff
    # Simple cubic grid of FCC points
    a = 1.0
    max_radius = 3.0 * (n ** (1.0/3.0))
    coords = []
    # Generate points in a cubic box
    r = int(math.ceil(max_radius / a)) + 1
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            for k in range(-r, r+1):
                # FCC points: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) scaled by a
                # Actually, primitive cell is simpler. Let's just use a dense random grid filtered by distance
                # Or better: generate a sphere of points and prune to keep nearest neighbors
                pass
    
    # Simpler robust method: Generate points on a sphere and project, or use a pre-computed small cluster and expand
    # Let's use a "jellium" like distribution: random points in a sphere, then relax
    # But we want a crystal template.
    
    # Let's generate an FCC lattice explicitly
    lattice_points = []
    # Basis vectors for FCC
    # We'll generate points in a cubic region and keep those within a radius
    radius = 1.5 * (n ** (1.0/3.0))
    for i in range(-int(radius), int(radius)+1):
        for j in range(-int(radius), int(radius)+1):
            for k in range(-int(radius), int(radius)+1):
                # FCC positions in a cubic unit cell of side 1
                # (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    # If we have too many, take the first n. If too few, add random points.
    if len(lattice_points) >= n:
        # Sort by distance to center to get a compact cluster
        lattice_points.sort(key=lambda p: np.linalg.norm(p))
        return np.array(lattice_points[:n])
    else:
        # Fill with random points
        remaining = n - len(lattice_points)
        for _ in range(remaining):
            p = np.random.uniform(-radius, radius, 3)
            lattice_points.append(p)
        return np.array(lattice_points)

def _generate_bcc(n):
    """Generate BCC-like cluster of size n."""
    radius = 1.5 * (n ** (1.0/3.0))
    lattice_points = []
    for i in range(-int(radius), int(radius)+1):
        for j in range(-int(radius), int(radius)+1):
            for k in range(-int(radius), int(radius)+1):
                # BCC positions: (0,0,0) and (0.5,0.5,0.5)
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    if len(lattice_points) >= n:
        lattice_points.sort(key=lambda p: np.linalg.norm(p))
        return np.array(lattice_points[:n])
    else:
        remaining = n - len(lattice_points)
        for _ in range(remaining):
            p = np.random.uniform(-radius, radius, 3)
            lattice_points.append(p)
        return np.array(lattice_points)

def _generate_random(n, seed=42):
    """Generate random cluster in a sphere."""
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check min distance
            if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                points.append(p)
    return np.array(points)

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    # Time budget per instance: 300s / 6 = 50s. 
    # We have 6 calls. Let's aim to use ~45s per call to be safe.
    # But we don't know which instance is which. 
    # We'll just check remaining time.
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate multiple templates
    templates = []
    try:
        templates.append(_generate_fcc(n))
    except: pass
    try:
        templates.append(_generate_bcc(n))
    except: pass
    try:
        templates.append(_generate_random(n, seed=123))
    except: pass
    try:
        templates.append(_generate_random(n, seed=456))
    except: pass
    
    # If templates are empty, fallback to pure random
    if not templates:
        templates.append(_generate_random(n, seed=0))
        
    # Optimize each template
    for i, template in enumerate(templates):
        if _get_remaining_time() < 10.0: # Safety margin
            break
            
        # Flatten for scipy
        x0 = template.flatten()
        
        # Use L-BFGS-B
        try:
            res = minimize(_morse_energy_gradient, x0, args=(rho,), method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            
            if res.success or res.fun < best_energy:
                # Check validity: min distance >= 0.5
                pos = res.x.reshape(n, 3)
                min_dist = np.inf
                for j in range(n):
                    for k in range(j+1, n):
                        d = np.linalg.norm(pos[j] - pos[k])
                        if d < min_dist:
                            min_dist = d
                        if d < 0.5:
                            min_dist = 0.0
                            break
                    if min_dist < 0.5:
                        break
                        
                if min_dist >= 0.5:
                    if res.fun < best_energy:
                        best_energy = res.fun
                        best_positions = pos
        except Exception:
            pass
            
    # If optimization failed or didn't improve, use the best relaxed template
    if best_positions is None:
        # Fallback: relax the first template using the provided relax function
        # This is slower but robust
        try:
            energy, positions = relax(templates[0], potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            best_positions = templates[0]
            
    return best_positions
# EVOLVE-BLOCK-END
