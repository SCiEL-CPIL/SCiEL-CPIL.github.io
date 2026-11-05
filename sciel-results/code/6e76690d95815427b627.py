"""Dzugutov cluster global minima search with polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

# Dzugutov potential parameters
A_const = 5.82
B_const = 1.28
a_const = 1.87
b_const = 1.94
c_const = 1.10
d_const = 0.27
m_const = 16.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy exactly."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a_const:
                energy += A_const * (r ** (-m_const) - B_const) * np.exp(c_const / (r - a_const))
            elif r < b_const:
                energy += B_const * np.exp(d_const / (r - b_const))
    return energy


def _generate_icosahedron(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _build_icosahedral_shells(n):
    """
    Build a cluster using concentric icosahedral shells.
    This is the natural low-energy structure for Dzugutov potential.
    Returns a list of candidate initial structures (numpy arrays).
    """
    candidates = []
    
    # Determine how many full shells we can fit
    # 1 (center) + 12 (shell 1) = 13
    # + 12 (shell 2) = 25
    # + 12 (shell 3) = 37
    # + 12 (shell 4) = 49
    
    k = (n - 1) // 12
    remainder = (n - 1) % 12
    
    # Radii for shells: 
    # We will use a fixed set of good radii for Dzugutov.
    # From literature, the 13-atom icosahedron has R~1.
    # The 25-atom (1+12+12) has R1~1, R2~1.8.
    # The 37-atom (1+12+12+12) has R1~1, R2~1.8, R3~2.6.
    
    base_radii = [1.0, 1.8, 2.6, 3.4, 4.2]
    
    # We will create variations with slightly different radii to escape local minima.
    for r1_var in [0.95, 1.0, 1.05]:
        for r2_var in [1.7, 1.8, 1.9]:
            for r3_var in [2.5, 2.6, 2.7]:
                pts = [np.zeros(3)]
                
                # Build full shells
                current_r = 1.0
                for i in range(k):
                    if i == 0:
                        current_r = r1_var
                    elif i == 1:
                        current_r = r1_var * r2_var
                    elif i == 2:
                        current_r = r1_var * r3_var
                    else:
                        current_r *= 1.8
                    shell = _generate_icosahedron(current_r)
                    pts.extend(shell)
                    
                # Add remaining atoms to the next shell
                if remainder > 0:
                    if k == 0:
                        r_next = r1_var
                    elif k == 1:
                        r_next = r1_var * r2_var
                    elif k == 2:
                        r_next = r1_var * r3_var
                    else:
                        r_next = current_r * 1.8
                    shell = _generate_icosahedron(r_next)
                    pts.extend(shell[:remainder])
                    
                # Ensure we have exactly n points
                if len(pts) > n:
                    pts = pts[:n]
                elif len(pts) < n:
                    # Fill with random points if logic failed (shouldn't happen)
                    while len(pts) < n:
                        pts.append(np.random.randn(3))
                        
                candidates.append(np.array(pts))
                
    return candidates


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate icosahedral shell candidates
    candidates = _build_icosahedral_shells(size)
    
    # Also add a few random compact structures as fallback
    rng = np.random.default_rng(size)
    for _ in range(2):
        R = 1.0 * (size / 4.0) ** (1/3)
        pts = rng.uniform(-R, R, (size, 3))
        candidates.append(pts)
        
    # Relax each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
            
        try:
            # Use relax
            energy, positions = relax(cand, potential, None)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            pass
            
    # If we have time left, do some perturbation-based search on the best structure
    if best_positions is not None and time.time() < deadline - 30:
        # Try to improve the best structure by adding noise and re-relaxing
        for i in range(5):
            if time.time() > deadline:
                break
            noise = rng.normal(0, 0.1, best_positions.shape)
            perturbed = best_positions + noise
            try:
                energy, positions = relax(perturbed, potential, None)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except:
                pass

    if best_positions is None:
        # Fallback
        best_positions = candidates[0]
        
    return best_positions
# EVOLVE-BLOCK-END
