"""Rewritten Dzugutov cluster global-minima search with Diverse Seeds and Adaptive Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 295.0  # Safety margin below 300s (evaluator kills at 330s)

def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _icosahedron_vertices(scale=1.0):
    """Generate vertices of a regular icosahedron scaled to 'scale'."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    # Normalize to unit sphere then scale
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v

def _mackay_icosahedron_55(spacing=1.0):
    """
    Generate a 55-atom Mackay Icosahedron.
    Structure: 1 center + 12 first shell + 42 second shell.
    """
    pts = [np.zeros(3)]
    
    # Shell 1: 12 atoms at radius 'spacing'
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p.copy() for p in shell1])
    
    # Shell 2: 42 atoms
    R2 = spacing * ((1.0 + np.sqrt(5.0)) / 2.0)
    shell2_verts = _icosahedron_vertices(scale=R2)
    pts.extend([p.copy() for p in shell2_verts])
    
    # Find edge midpoints
    d_edge = np.linalg.norm(shell2_verts[0] - shell2_verts[1])
    midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(shell2_verts[i] - shell2_verts[j])
            if np.isclose(d, d_edge, rtol=1e-3):
                mid = (shell2_verts[i] + shell2_verts[j]) / 2.0
                midpoints.append(mid)
                
    if len(midpoints) == 30:
        pts.extend(midpoints)
    else:
        for _ in range(42 - len(pts)):
            dir = np.random.normal(0, 1, 3)
            dir = dir / np.linalg.norm(dir)
            p = dir * R2
            pts.append(p)
        
    pts = np.array(pts, dtype=float)
    return pts

def _tetrahedral_growth_seed(n, rng, spacing=1.0):
    """
    Generate a seed using greedy tetrahedral growth.
    """
    if n < 4:
        # Handle small n
        pts = []
        for i in range(n):
            p = rng.normal(0, spacing, 3)
            pts.append(p)
        return np.array(pts)
    
    # Start with a regular tetrahedron
    # Vertices of tetrahedron centered at origin
    t_verts = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    t_verts /= np.linalg.norm(t_verts[0])
    t_verts *= spacing
    
    pts = [t_verts[i].copy() for i in range(min(4, n))]
    
    for k in range(len(pts), n):
        if _remaining_time() < 10.0:
            break
            
        # Find a good place to add the next atom
        # Strategy: Pick a random existing atom, place new atom at distance 'spacing' from it
        # in a random direction. Then relax.
        # To make it faster, we just pick a few candidates and choose the best one after relaxation?
        # No, that's too slow. Just pick one candidate.
        
        idx = rng.integers(0, len(pts))
        base = pts[idx]
        
        # Random direction
        dir = rng.normal(0, 1, 3)
        dir /= np.linalg.norm(dir)
        
        # Place new atom
        new_p = base + dir * spacing
        
        # Check distance to all other atoms to avoid overlap
        valid = True
        for p in pts:
            if np.linalg.norm(new_p - p) < 0.6 * spacing:
                valid = False
                break
        
        if not valid:
            # Try a few times
            for _ in range(10):
                dir = rng.normal(0, 1, 3)
                dir /= np.linalg.norm(dir)
                new_p = base + dir * spacing
                valid = True
                for p in pts:
                    if np.linalg.norm(new_p - p) < 0.6 * spacing:
                        valid = False
                        break
                if valid:
                    break
            
            if not valid:
                # Just add it anyway, relax will fix it
                new_p = base + dir * spacing * 1.5
        
        pts.append(new_p)
        
    return np.array(pts)

def _random_ball_seed(n, rng, spacing=1.0):
    """
    Generate a random seed in a ball.
    """
    radius = np.sqrt(n) * spacing * 0.5
    pts = []
    for _ in range(n):
        # Rejection sampling for ball
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) < radius:
                break
        pts.append(p)
    return np.array(pts)

def _make_seed(n, rng, seed_type='auto'):
    """
    Generate a seed based on type.
    """
    if seed_type == 'auto':
        # Mix of types
        types = ['mackay', 'tet', 'random']
        seed_type = types[rng.integers(0, len(types))]
        
    if seed_type == 'mackay':
        if n == 55:
            return _mackay_icosahedron_55(1.0)
        elif n < 55:
            base = _mackay_icosahedron_55(1.0)
            # Remove atoms from outer shell
            num_remove = 55 - n
            if num_remove <= 0:
                return base
            outer_indices = np.arange(13, 55)
            core_indices = np.arange(0, 13)
            
            to_remove = []
            if num_remove <= 42:
                to_remove = rng.choice(outer_indices, size=num_remove, replace=False)
            else:
                to_remove_outer = rng.choice(outer_indices, size=42, replace=False)
                remaining_remove = num_remove - 42
                to_remove_core = rng.choice(core_indices, size=remaining_remove, replace=False)
                to_remove = np.concatenate([to_remove_outer, to_remove_core])
                
            keep_mask = np.ones(55, dtype=bool)
            keep_mask[to_remove] = False
            return base[keep_mask]
        else:
            base = _mackay_icosahedron_55(1.0)
            num_add = n - 55
            new_pts = []
            R2 = 1.0 * ((1.0 + np.sqrt(5.0)) / 2.0)
            for _ in range(num_add):
                dir = rng.normal(0, 1, 3)
                dir /= np.linalg.norm(dir)
                r = R2 * (1.0 + rng.uniform(0.0, 0.2))
                p = dir * r
                new_pts.append(p)
            return np.vstack([base, np.array(new_pts)])
            
    elif seed_type == 'tet':
        return _tetrahedral_growth_seed(n, rng)
        
    else: # random
        return _random_ball_seed(n, rng)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    t0 = time.time()
    best_energy = np.inf
    best_positions = None
    
    # 1. Generate Diverse Seeds
    num_seeds = 25
    seeds = []
    rng = np.random.default_rng(42)
    
    for i in range(num_seeds):
        if _remaining_time() < 5.0:
            break
        # Ensure variety
        if i < 5:
            seed_type = 'mackay'
        elif i < 15:
            seed_type = 'tet'
        else:
            seed_type = 'random'
            
        seed = _make_seed(size, rng, seed_type)
        seeds.append(seed)
        
    # 2. Relax Seeds
    relaxed_seeds = []
    for seed in seeds:
        if _remaining_time() < 5.0:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
            if np.all(np.isfinite(energy)) and np.all(np.isfinite(positions)):
                relaxed_seeds.append((energy, positions))
        except Exception:
            continue
            
    if not relaxed_seeds:
        seed = _make_seed(size, rng, 'mackay')
        try:
            _, best_positions = relax(seed, potential, parameter)
        except Exception:
            best_positions = _make_seed(size, rng, 'mackay')
        return best_positions

    # Sort by energy
    relaxed_seeds.sort(key=lambda x: x[0])
    best_energy, best_positions = relaxed_seeds[0]
    
    # 3. Adaptive Basin Hopping
    bh_rng = np.random.default_rng(123)
    
    # Run BH from top 5 seeds
    top_seeds = relaxed_seeds[:5]
    
    for seed_energy, seed_pos in top_seeds:
        if _remaining_time() < 5.0:
            break
            
        current_energy = seed_energy
        current_pos = seed_pos
        local_best_energy = seed_energy
        local_best_pos = seed_pos
        
        # Adaptive parameters
        scale = 0.2
        T = 0.1
        accept_count = 0
        total_count = 0
        
        # Run BH for a fixed number of steps or until time runs out
        # We'll use a loop that checks time
        while _remaining_time() > 5.0:
            # Perturb
            perturbed = current_pos + bh_rng.normal(0, scale, current_pos.shape)
            
            try:
                energy, positions = relax(perturbed, potential, parameter)
            except Exception:
                continue
                
            if not np.all(np.isfinite(energy)):
                continue
                
            total_count += 1
            
            # Metropolis acceptance
            if energy < current_energy or bh_rng.random() < np.exp(-(energy - current_energy) / T):
                current_energy = energy
                current_pos = positions
                accept_count += 1
                
                if energy < local_best_energy:
                    local_best_energy = energy
                    local_best_pos = positions
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                        
            # Adapt scale and T every 10 steps
            if total_count % 10 == 0:
                acceptance_rate = accept_count / total_count
                if acceptance_rate > 0.5:
                    scale *= 1.1
                elif acceptance_rate < 0.1:
                    scale *= 0.9
                # Cool down
                T *= 0.95
                
                # Reset counters
                accept_count = 0
                total_count = 0
                
            # Occasionally restart from local best if stuck?
            # If no improvement for 50 steps, maybe reset current to local best?
            # This is a heuristic.
            
    return best_positions
# EVOLVE-BLOCK-END
