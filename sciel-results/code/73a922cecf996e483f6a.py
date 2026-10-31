"""LJ38 cluster global minimum search."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 38
DEADLINE = 115.0  # seconds, safety margin before 120s limit

# FCC truncated octahedron vertex positions (38 vertices)
# The truncated octahedron has 24 hexagonal faces and 8 square faces.
# Its 38 vertices are:
# - 8 vertices from the cube: (±1, ±1, ±1)
# - 12 vertices from the octahedron edges: permutations of (0, ±2, ±2)
# - 12 vertices from the octahedron face centers... wait, let me think more carefully.

# Actually, a truncated octahedron has 24 vertices, not 38.
# For LJ38, the global minimum is an FCC-structured cluster.
# The FCC truncated octahedron for N=38 has a specific arrangement.

# Let me use the known FCC stacking. In FCC:
# Layer 1: 1 atom (center)
# Layer 2: 12 atoms (first coordination shell, octahedral + tetrahedral sites)
# Layer 3: 12 atoms (second coordination shell)
# Layer 4: 13 atoms (third coordination shell)
# Total: 1 + 12 + 12 + 13 = 38

# Actually, let me use the standard FCC truncated octahedron coordinates.
# The 38-atom FCC cluster is formed by cutting an FCC lattice.
# Standard coordinates for the 38-atom FCC truncated octahedron:

def _fcc_38_positions():
    """Generate the 38-atom FCC truncated octahedron positions."""
    # FCC lattice basis vectors with lattice constant a
    # The 38-atom cluster corresponds to a specific set of FCC lattice points.
    
    # Using the known structure: it's an FCC cluster with layers
    # Let's use the explicit coordinates from literature.
    
    # FCC unit cell has atoms at:
    # (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
    
    # For a truncated octahedron, we take all FCC points within a certain radius.
    # Let's generate FCC points and select the 38 closest to origin.
    
    a = 1.0  # lattice constant
    # Generate FCC points in a cube
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC: add all combinations of half-integer offsets
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    points.append((x, y, z))
    
    points = np.array(points)
    # Remove duplicates
    points = np.unique(points, axis=0)
    
    # Sort by distance from origin
    dists = np.sum(points**2, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    # Take the 38 closest
    pts = points[:38].copy()
    
    # Center the cluster
    pts -= pts.mean(axis=0)
    
    return pts

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    r2 = np.sum(pos**2, axis=1)
    d2 = r2[:, None] + r2[None, :] - 2 * pos @ pos.T
    np.fill_diagonal(d2, 1.0)  # avoid division by zero
    r = np.sqrt(d2)
    r6 = r**(-6)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    return energy - 2.0 * np.sum(np.diag(r12 - r6))  # subtract double-counted diagonal

def _lj_energy_simple(pos):
    """Simple O(N^2) LJ energy calculation."""
    n = len(pos)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = pos[i] - pos[j]
            r2 = np.dot(d, d)
            r6 = 1.0 / (r2 * r2 * r2)
            r12 = r6 * r6
            energy += 4.0 * (r12 - r6)
    return energy

def construct():
    start_time = time.time()
    
    # Start with the FCC truncated octahedron
    fcc_pos = _fcc_38_positions()
    
    # Scale to optimal spacing (LJ minimum at r=2^(1/6)≈1.1225)
    # The FCC lattice constant should be such that nearest neighbors are at 2^(1/6)
    # In FCC, nearest neighbor distance = a/sqrt(2)
    # So a = 2^(1/6) * sqrt(2) = 2^(2/3) ≈ 1.5874
    # But our generated points are already in some scale, let's rescale
    # Find the minimum distance between pairs
    min_d = np.inf
    for i in range(len(fcc_pos)):
        for j in range(i+1, len(fcc_pos)):
            d = np.linalg.norm(fcc_pos[i] - fcc_pos[j])
            if d < min_d:
                min_d = d
    
    # Scale so that min distance = 2^(1/6)
    scale = (2.0 ** (1.0/6.0)) / min_d if min_d > 0 else 1.0
    fcc_pos = fcc_pos * scale
    
    best_pos = fcc_pos.copy()
    best_energy = _lj_energy_simple(best_pos)
    
    # Try to improve with local optimization
    def energy_fn(x):
        pos = x.reshape(-1, 3)
        # Check minimum distance constraint
        min_d = np.inf
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    min_d = d
        if min_d < 0.5:
            return 1e10
        return _lj_energy_simple(pos)
    
    # L-BFGS-B optimization from FCC start
    try:
        res = minimize(energy_fn, best_pos.flatten(), method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_energy:
            best_pos = res.x.reshape(-1, 3)
            best_energy = res.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return best_pos
    
    # Try differential evolution for global search
    try:
        # Bounds: atoms within a reasonable range
        bounds = [(-5, 5)] * (3 * N)
        
        def de_energy(x):
            pos = x.reshape(-1, 3)
            # Check minimum distance
            min_d = np.inf
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < min_d:
                        min_d = d
            if min_d < 0.5:
                return 1e10
            return _lj_energy_simple(pos)
        
        res_de = differential_evolution(de_energy, bounds, maxiter=50, 
                                        tol=1e-8, popsize=15, seed=42,
                                        polish=False, init='sobol')
        if res_de.fun < best_energy:
            best_pos = res_de.x.reshape(-1, 3)
            best_energy = res_de.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return best_pos
    
    # Final local optimization
    try:
        res_final = minimize(energy_fn, best_pos.flatten(), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        if res_final.fun < best_energy:
            best_pos = res_final.x.reshape(-1, 3)
            best_energy = res_final.fun
    except Exception:
        pass
    
    return best_pos
