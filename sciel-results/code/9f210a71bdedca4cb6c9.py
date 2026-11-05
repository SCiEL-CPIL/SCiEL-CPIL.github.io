"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking
_START_TIME = [None]
_DEADLINE = 320.0  # Safety margin before 330s kill

def _remaining_time():
    if _START_TIME[0] is None:
        _START_TIME[0] = time.time()
    return _DEADLINE - (time.time() - _START_TIME[0])

def _morse_energy(positions, rho):
    """Compute Morse energy for validation or if relax fails."""
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            x = rho * (1.0 - r)
            ex = math.exp(x)
            E += ex * (ex - 2.0)
    return E

def _generate_fcc(n):
    """Generate an FCC cluster of n atoms."""
    # FCC lattice constant a=1 (equilibrium distance is 1)
    # Closest neighbors are at distance 1.
    # We generate points in an FCC lattice and take the n closest to origin.
    points = []
    # Generate a sufficient number of points in a box
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * 2
    # Actually, standard FCC unit cell has 4 atoms.
    # Let's just generate a grid of FCC points.
    # Simpler: generate points in a sphere of radius R, check if they are on FCC lattice.
    # Or just generate a list of FCC points and sort by distance.
    
    max_coord = int(math.ceil(n ** (1/3))) + 2
    candidates = []
    # FCC lattice vectors
    # To get closest neighbors at 1.0, we need to scale the lattice.
    # In standard FCC with lattice constant a, closest neighbors are a/sqrt(2).
    # So a = sqrt(2).
    a = math.sqrt(2)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC points: (i,j,k) where i+j+k is even? No.
                # Let's use the standard basis.
                # Point: (i*a, j*a, k*a) + basis
                # Basis 1: (0,0,0)
                # Basis 2: (0, a/2, a/2)
                # Basis 3: (a/2, 0, a/2)
                # Basis 4: (a/2, a/2, 0)
                
                # We can iterate over integer grid and add basis vectors.
                # To be efficient, we can just generate all 4 types.
                
                # Type 1
                p1 = np.array([i*a, j*a, k*a])
                # Type 2
                p2 = np.array([i*a, (j+0.5)*a, (k+0.5)*a])
                # Type 3
                p3 = np.array([(i+0.5)*a, j*a, (k+0.5)*a])
                # Type 4
                p4 = np.array([(i+0.5)*a, (j+0.5)*a, k*a])
                
                for p in [p1, p2, p3, p4]:
                    if np.linalg.norm(p) < max_coord * a * 1.5: # Rough bound
                        candidates.append(p)
                        
    # Remove duplicates and sort by distance to origin
    # Using a set of tuples for unique
    unique_points = set()
    for p in candidates:
        unique_points.add(tuple(np.round(p, 5)))
        
    unique_list = [np.array(p) for p in unique_points]
    unique_list.sort(key=lambda p: np.linalg.norm(p))
    
    if len(unique_list) < n:
        # Fallback: just take what we have or expand
        return np.array(unique_list[:n])
        
    return np.array(unique_list[:n])

def _generate_bcc(n):
    """Generate a BCC cluster of n atoms."""
    max_coord = int(math.ceil(n ** (1/3))) + 2
    candidates = []
    # BCC lattice constant a. Closest neighbors are a*sqrt(3)/2.
    # We want distance 1. So a*sqrt(3)/2 = 1 => a = 2/sqrt(3).
    a = 2.0 / math.sqrt(3.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # BCC basis: (0,0,0) and (0.5, 0.5, 0.5)
                p1 = np.array([i*a, j*a, k*a])
                p2 = np.array([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
                
                for p in [p1, p2]:
                    if np.linalg.norm(p) < max_coord * a * 1.5:
                        candidates.append(p)
                        
    unique_points = set()
    for p in candidates:
        unique_points.add(tuple(np.round(p, 5)))
        
    unique_list = [np.array(p) for p in unique_points]
    unique_list.sort(key=lambda p: np.linalg.norm(p))
    
    if len(unique_list) < n:
        return np.array(unique_list[:n])
        
    return np.array(unique_list[:n])

def _generate_hcp(n):
    """Generate an HCP cluster of n atoms."""
    # HCP is more complex to generate exactly.
    # Let's approximate with a distorted FCC or just use FCC/BCC for now.
    # Or generate layers.
    # For simplicity and robustness, we will rely on FCC and BCC which are very good seeds.
    # We can add a "random" seed that is relaxed to a local minimum.
    return None

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    if _START_TIME[0] is None:
        _START_TIME[0] = time.time()
        
    # 1. Generate diverse initial configurations
    inits = []
    
    # FCC
    try:
        fcc_pos = _generate_fcc(size)
        if len(fcc_pos) == size:
            inits.append(fcc_pos)
    except:
        pass
        
    # BCC
    try:
        bcc_pos = _generate_bcc(size)
        if len(bcc_pos) == size:
            inits.append(bcc_pos)
    except:
        pass
        
    # Random compact
    rng = np.random.default_rng(size)
    radius = 1.0 * size ** (1.0 / 3.0) * 0.8
    points = []
    while len(points) < size:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    if len(points) == size:
        inits.append(np.array(points))

    # Relax all initial configurations
    best_energy = np.inf
    best_pos = None
    
    for init_pos in inits:
        if _remaining_time() < 5.0:
            break
        try:
            energy, pos = relax(init_pos, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
        except:
            continue
            
    if best_pos is None:
        # Fallback if all relaxations failed
        return inits[0] if inits else np.zeros((size, 3))

    # 2. Basin Hopping
    # Start from best local minimum
    current_pos = best_pos.copy()
    current_energy = best_energy
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Parameters
    T = 1.0
    T_min = 0.01
    T_factor = 0.95
    step_size = 0.2
    max_steps = 10000
    
    rng_bh = np.random.default_rng(42)
    
    for step in range(max_steps):
        if _remaining_time() < 10.0:
            break
            
        # Propose move
        perturbation = rng_bh.normal(0, step_size, size=(size, 3))
        new_pos = current_pos + perturbation
        
        # Relax
        try:
            new_energy, new_pos_relaxed = relax(new_pos, potential, rho)
        except:
            continue
            
        # Accept/Reject
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng_bh.random() < math.exp(-delta_E / T):
            current_pos = new_pos_relaxed
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
        # Cool down
        T = max(T * T_factor, T_min)
        
        # Occasionally reheat or increase step size if stuck?
        # Simple strategy: if no improvement for 100 steps, increase step size slightly
        # (Implemented implicitly by random walk, but we can track)
        
    return best_pos
# EVOLVE-BLOCK-END
