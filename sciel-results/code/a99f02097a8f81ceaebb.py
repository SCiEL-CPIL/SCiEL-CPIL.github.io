"""Morse cluster global minima: Basin hopping with robust relax and lattice seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 290.0  # 10s safety margin for 300s limit
_INSTANCE_COUNT = 6
_INSTANCE_INDEX = [0]

def _generate_fcc_lattice(n, spacing=1.0):
    """Generate n points from an FCC lattice centered at origin."""
    # Approximate radius needed for n points
    # Density of FCC is 4 atoms per conventional cell of volume a^3.
    # Volume per atom ~ a^3/4.
    # V = n * a^3/4. R ~ (3V/4pi)^(1/3).
    # Let's just generate a large grid and pick closest to origin.
    
    r_max = spacing * (n ** (1.0/3.0) + 2)
    limit = int(r_max / spacing) + 2
    
    candidates = []
    # FCC points: (i, j, k) where i,j,k integers and i+j+k even?
    # Actually, standard FCC basis in cubic coords (x,y,z) with x,y,z integers:
    # (0,0,0), (0,1,1), (1,0,1), (1,1,0) ... scaled by a/2?
    # Let's use the property that FCC is a subset of the integer lattice scaled by 0.5.
    # Points (x,y,z) such that x,y,z are integers and x+y+z is even? No, that's BCC-like in some contexts.
    # Let's stick to generating points on the grid and filtering.
    # FCC primitive vectors: a/2(0,1,1), a/2(1,0,1), a/2(1,1,0).
    # Any point is (n1+n2+n3)/2, (n1+n3+n4)/2, (n2+n3+n4)/2? No.
    # Simpler: Generate all integer triples (i,j,k) in [-limit, limit].
    # Keep those where (i+j+k) % 2 == 0? 
    # Let's check: (0,0,0) sum 0 (even). (1,1,0) sum 2 (even). (0,1,1) sum 2 (even). (1,0,1) sum 2 (even).
    # Yes, the set of points (i,j,k) * (a/2) where i,j,k are integers and i+j+k is even forms the FCC lattice.
    
    for i in range(-2*limit, 2*limit + 1):
        for j in range(-2*limit, 2*limit + 1):
            for k in range(-2*limit, 2*limit + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (spacing / 2.0)
                    candidates.append(p)
    
    if not candidates:
        return np.random.uniform(-r_max, r_max, (n, 3))
        
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_idx = np.argsort(dists)
    selected = candidates[sorted_idx[:n]]
    
    # Center
    selected -= np.mean(selected, axis=0)
    return selected

def _generate_bcc_lattice(n, spacing=1.0):
    """Generate n points from a BCC lattice centered at origin."""
    r_max = spacing * (n ** (1.0/3.0) + 2)
    limit = int(r_max / spacing) + 2
    
    candidates = []
    # BCC points: (i,j,k) * (a/2) where i,j,k have same parity.
    # (0,0,0) even. (1,1,1) odd.
    for i in range(-2*limit, 2*limit + 1):
        for j in range(-2*limit, 2*limit + 1):
            for k in range(-2*limit, 2*limit + 1):
                if (i % 2) == (j % 2) and (j % 2) == (k % 2):
                    p = np.array([i, j, k]) * (spacing / 2.0)
                    candidates.append(p)
    
    if not candidates:
        return np.random.uniform(-r_max, r_max, (n, 3))
        
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_idx = np.argsort(dists)
    selected = candidates[sorted_idx[:n]]
    
    selected -= np.mean(selected, axis=0)
    return selected

def _generate_random(n, spacing=1.0, seed=None):
    """Generate n random points."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = rng.uniform(-radius, radius, (n, 3))
    return points

def _enforce_min_distance(positions, min_dist=0.5):
    """Ensure all pairwise distances are at least min_dist."""
    n = len(positions)
    if n < 2:
        return positions
        
    # Vectorized check for violations
    for _ in range(20): # Max iterations to resolve
        # Compute all pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Find violations (dist < min_dist and i != j)
        mask = (dist < min_dist) & (~np.eye(n, dtype=bool))
        if not np.any(mask):
            break
            
        # Identify pairs
        i_idx, j_idx = np.where(mask)
        
        moved = False
        for i, j in zip(i_idx, j_idx):
            if i >= j: continue # Handle each pair once
            d = dist[i, j]
            if d < 1e-8:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = (positions[i] - positions[j]) / d
                
            push = (min_dist - d) / 2.0 + 0.01
            positions[i] += push * direction
            positions[j] -= push * direction
            moved = True
            
        if not moved:
            break
            
    return positions

def construct(instance):
    """Construct low-energy Morse cluster configuration."""
    _INSTANCE_INDEX[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Time management
    elapsed = time.time() - _START_TIME
    remaining_total = _HARD_DEADLINE - time.time()
    instances_left = _INSTANCE_COUNT - _INSTANCE_INDEX[0] + 1
    
    # Allocate time budget for this instance
    # Larger N takes longer. Heuristic: base time + per-atom time.
    # Let's reserve some time for the final instance just in case.
    time_budget = max(5.0, min(45.0, remaining_total / instances_left))
    
    start_time = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seeds
    seeds = []
    
    # 1. FCC Lattice
    try:
        seeds.append(_generate_fcc_lattice(size, spacing=1.0))
    except Exception:
        pass
        
    # 2. BCC Lattice
    try:
        seeds.append(_generate_bcc_lattice(size, spacing=1.0))
    except Exception:
        pass
        
    # 3. Random structures
    for s in [42, size, size * 7]:
        try:
            seeds.append(_generate_random(size, spacing=1.0, seed=s))
        except Exception:
            pass

    if not seeds:
        seeds.append(_generate_random(size, spacing=1.0, seed=123))

    # Phase 1: Relax all seeds
    for seed_idx, seed_pos in enumerate(seeds):
        if time.time() - start_time > time_budget * 0.5:
            break
            
        # Jitter to break symmetry
        jittered = seed_pos + np.random.randn(*seed_pos.shape) * 0.01
        jittered = _enforce_min_distance(jittered, 0.5)
        
        try:
            # relax returns (energy, positions)
            energy, positions = relax(jittered, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue

    if best_positions is None:
        # Fallback if relax failed for all
        best_positions = seeds[0].copy()
        best_energy = np.inf # We'll try to optimize it in phase 2

    # Phase 2: Basin Hopping
    # Use remaining time
    time_elapsed = time.time() - start_time
    remaining_time = time_budget - time_elapsed
    
    if remaining_time > 2.0 and best_positions is not None:
        kick_scale = 0.3
        # Estimate number of kicks
        # Each kick involves a relax which takes variable time.
        # Let's do as many as we can in the remaining time.
        
        while time.time() - start_time < time_budget:
            # Kick
            kicked = best_positions + np.random.randn(*best_positions.shape) * kick_scale
            kicked = _enforce_min_distance(kicked, 0.5)
            
            # Relax
            try:
                energy, positions = relax(kicked, potential, rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
                    # Slightly reduce kick scale if we found a good basin? 
                    # Or keep it constant. Constant is simpler.
            except Exception:
                break

    # Final check on time
    if time.time() > _HARD_DEADLINE - 5.0:
         # Emergency return
         if best_positions is None:
             best_positions = _generate_random(size, 1.0, 999)
         return _enforce_min_distance(best_positions, 0.5)

    # Final polish: One more relax on the best position to ensure it's fully converged
    if best_positions is not None:
        try:
            final_energy, final_pos = relax(best_positions, potential, rho)
            if final_energy < best_energy:
                best_positions = final_pos
        except Exception:
            pass

    best_positions = _enforce_min_distance(best_positions, 0.5)
    return best_positions
# EVOLVE-BLOCK-END
