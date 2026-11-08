"""Program for Sutton-Chen TM cluster global minima search using Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 280.0  # Safety margin before 300s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Efficient vectorized check
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    # FCC points: (i, j, k) where i+j+k is even, scaled by 1/sqrt(2)
    # This creates a lattice with NN distance 1.0.
    # We want to pack n atoms densely.
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    # BCC: (i, j, k) and (i+0.5, j+0.5, k+0.5)
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_icosahedron(n):
    """Generate an icosahedral cluster (Mackay shells)."""
    # Shell 1: 1 atom (center)
    # Shell 2: 12 atoms (vertices of icosahedron)
    # Shell 3: 42 atoms (edges and face centers)
    # Total 55.
    
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = []
    for i in range(12):
        if i < 4:
            verts.append(np.array([0, 1, phi]))
            verts.append(np.array([0, 1, -phi]))
            verts.append(np.array([0, -1, phi]))
            verts.append(np.array([0, -1, -phi]))
        elif i < 8:
            verts.append(np.array([1, phi, 0]))
            verts.append(np.array([1, -phi, 0]))
            verts.append(np.array([-1, phi, 0]))
            verts.append(np.array([-1, -phi, 0]))
        else:
            verts.append(np.array([phi, 0, 1]))
            verts.append(np.array([phi, 0, -1]))
            verts.append(np.array([-phi, 0, 1]))
            verts.append(np.array([-phi, 0, -1]))
    
    # Normalize to radius 1
    v = [p / np.linalg.norm(p) for p in verts]
    
    pts = [np.zeros(3)] # Center
    if n > 1:
        pts.extend(v[:min(12, n-1)])
    
    if n > 13:
        # Add second shell (42 atoms)
        # This is complex to generate perfectly. 
        # For n > 13, we will fall back to FCC/BCC in the main loop 
        # or just return the first n points if we can generate them.
        # Actually, for n=55, icosahedron is a strong candidate.
        # Let's generate the 42 points of the second shell.
        # These are located at the midpoints of edges and centers of faces? 
        # No, Mackay icosahedron shells are:
        # 1, 12, 42, 120...
        # The 42 points are at distance sqrt(3)/2 * R? No.
        # Let's just use FCC for n > 13 to be safe and robust, 
        # as generating exact Mackay shells is error-prone without geometry libs.
        # However, icosahedron is often the global min for n=13.
        # For n=55, it might be close.
        # Let's stick to FCC/BCC for simplicity and robustness in this implementation,
        # or add a simple "snowball" growth if needed.
        # Given time constraints, we will rely on BH to find the icosahedral 
        # minimum if we start near it, or use FCC which is also a strong candidate 
        # for larger N in SC potential.
        
        # Let's try to add random points on a sphere of radius ~1.5 for n>13
        # This is a weak start, but BH should fix it.
        rng = np.random.default_rng(42)
        for _ in range(n - len(pts)):
            # Random point on sphere
            u = rng.normal(size=3)
            u = u / np.linalg.norm(u)
            pts.append(u * 1.5)
            
    return np.array(pts[:n])

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # List of generator functions
    generators = [_generate_fcc, _generate_bcc, _generate_icosahedron]
    
    rng = np.random.default_rng(42 + n)
    
    # Basin Hopping Parameters
    # We will run multiple BH chains in parallel? No, simple sequential or 
    # a few parallel chains if time permits. 
    # Given the 300s limit and 6 instances, we have ~50s per instance.
    # relax() is likely fast (L-BFGS-B).
    
    # Strategy: 
    # 1. Try a few diverse starts with a short BH chain each.
    # 2. Use the best result as the start for a longer BH chain.
    
    # Pre-generate starts
    starts = []
    for gen in generators:
        try:
            pos = gen(n)
            if _is_valid(pos):
                starts.append(pos)
        except:
            pass
    
    # Add some random perturbations of FCC
    fcc_base = _generate_fcc(n)
    for _ in range(3):
        noise = rng.normal(0, 0.2, fcc_base.shape)
        pos = fcc_base + noise
        if _is_valid(pos):
            starts.append(pos)
    
    # If no valid starts, create one
    if not starts:
        starts.append(fcc_base)
    
    # Basin Hopping Loop
    # We will iterate through starts and perform a BH search for each.
    # To maximize diversity, we will run short BH chains for all starts,
    # then a longer chain for the best one found so far.
    
    # Time budget per start: 5s initially.
    # Total starts: ~6. 30s.
    # Remaining 20s for polishing the best.
    
    all_results = []
    
    for i, start in enumerate(starts):
        if _time_left() < 10.0:
            break
            
        # Basin Hopping
        current_pos = start
        current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
        
        if not np.isfinite(current_energy) or not _is_valid(current_pos):
            continue
            
        best_e_chain = current_energy
        best_p_chain = current_pos
        
        # Annealing parameters
        T = 0.1
        step_size = 0.5
        max_steps = 50 # Short chain for diversity
        
        for step in range(max_steps):
            if _time_left() < 10.0:
                break
                
            # Generate random perturbation
            noise = rng.normal(0, step_size, current_pos.shape)
            new_pos = current_pos + noise
            
            # Relax
            new_energy, new_pos = relax(new_pos, 'sutton_chen', kind)
            
            if not np.isfinite(new_energy) or not _is_valid(new_pos):
                continue
                
            # Metropolis criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                current_energy = new_energy
                current_pos = new_pos
                
                if current_energy < best_e_chain:
                    best_e_chain = current_energy
                    best_p_chain = current_pos
            
            # Cool down
            T *= 0.95
            step_size *= 0.99 # Slight reduction in step size
            
        all_results.append((best_e_chain, best_p_chain))
        
        if best_e_chain < best_energy:
            best_energy = best_e_chain
            best_pos = best_p_chain
            
    # Polish the best result with a longer BH chain
    if best_pos is not None and _time_left() > 15.0:
        current_pos = best_pos
        current_energy = best_energy
        
        T = 0.05
        step_size = 0.3
        max_steps = 200 # Longer chain
        
        for step in range(max_steps):
            if _time_left() < 5.0:
                break
                
            noise = rng.normal(0, step_size, current_pos.shape)
            new_pos = current_pos + noise
            
            new_energy, new_pos = relax(new_pos, 'sutton_chen', kind)
            
            if not np.isfinite(new_energy) or not _is_valid(new_pos):
                continue
                
            delta_e = new_energy - current_energy
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                current_energy = new_energy
                current_pos = new_pos
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos
            
            T *= 0.98
            step_size *= 0.995

    # Final fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            # Last resort: spread points randomly
            best_pos = rng.uniform(-2, 2, (n, 3))
            # Ensure validity
            for _ in range(100):
                if _is_valid(best_pos):
                    break
                best_pos = rng.uniform(-2, 2, (n, 3))
            # If still invalid, try FCC again
            if not _is_valid(best_pos):
                best_pos = _generate_fcc(n)

    return best_pos

# EVOLVE-BLOCK-END
