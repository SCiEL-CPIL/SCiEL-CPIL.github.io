"""
Strategy: Physics-driven self-assembly via random dense initialization and iterative relaxation.
Instead of forcing icosahedral symmetry (which may be a local trap for non-Mackay numbers like 38, 45, 50, 57, 60),
we initialize atoms in a compact random configuration and use multiple relaxation attempts.
The Dzugutov potential is designed to favor PT order, so a compact random start should
naturally evolve towards PT motifs if given enough local optimization steps.
We use a "kick-and-relax" strategy:
1. Start with a dense random packing.
2. Relax to local minimum.
3. If time permits, apply a small random perturbation (kick) and relax again.
4. Keep the best energy found.
This avoids the bias of specific geometric templates and explores the energy landscape more broadly.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from autoresearch_suite.chem_clusters import relax

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 300.0
_LOCK = threading.Lock()

def _get_remaining_time():
    # We assume the 300s is the total wall-clock time for the entire script execution.
    # Since construct is called sequentially for each n, we track global time.
    # To be safe, we aim to finish well before 300s.
    return max(0.1, 290.0 - (time.time() - _START_TIME))

def _random_dense_packing(n, rng, min_dist=0.6):
    """
    Generate n points in a compact random configuration.
    We use a rejection sampling method to ensure minimum distance.
    The bounding box is sized to be compact but not overlapping.
    """
    points = []
    # Estimate radius for compact packing
    # Volume ~ n * (min_dist/2)^3 * packing_fraction
    # Let's just try a box of side L
    # For n=60, if min_dist=0.6, volume per atom ~ 0.216. Total vol ~ 13.
    # Cube side ~ 2.3.
    # Let's scale L with n^(1/3)
    L = 2.5 * (n / 38.0) ** (1.0 / 3.0)
    
    # To ensure we can place all points, we might need to increase L if rejection fails
    attempts = 0
    max_attempts = 50000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-L, L, 3)
        # Check distance to all existing points
        if len(points) == 0:
            points.append(p)
            continue
        
        dists = np.linalg.norm(np.array(points) - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)
        attempts += 1
        
    # If we couldn't place all points, expand the box and retry
    if len(points) < n:
        # Fallback: just place remaining randomly, ensuring they are far enough
        # Or increase L and restart
        # For simplicity, let's just return what we have if it's close, or force place
        # Actually, better to increase L and retry once
        L *= 1.5
        points = []
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-L, L, 3)
            if len(points) == 0:
                points.append(p)
                continue
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) >= min_dist:
                points.append(p)
            attempts += 1
            
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    # We have ~45s per instance on average if we want to be safe with 6 instances
    # But since it's sequential, we can use the global remaining time
    # However, if we use too much time for n=38, we might run out for n=60.
    # Let's allocate a fixed budget per call, but capped by global remaining.
    # 6 calls * 45s = 270s. Safe.
    time_budget = 45.0
    
    # If this is the last call, we can use more time? No, we don't know the order.
    # Stick to 45s per call.
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy:
    # 1. Generate initial random dense packing
    # 2. Relax
    # 3. If time permits, do "kick and relax" iterations
    #    - Kick: add small random noise to positions
    #    - Relax again
    #    - Keep best
    
    # Initial setup
    initial_pos = _random_dense_packing(size, rng)
    
    # First relaxation
    try:
        energy, positions = relax(initial_pos, potential, parameter)
        best_energy = energy
        best_positions = positions
    except Exception:
        # If relax fails, return initial (should not happen)
        return initial_pos
    
    # Kick and relax loop
    # We want to escape local minima.
    # The Dzugutov potential has many local minima.
    # A small kick might help find a better basin.
    
    # Determine number of kicks based on time
    # Each relax call takes ~1-5 seconds depending on size.
    # Let's assume ~2s per relax.
    # Budget 45s -> ~20 relax calls.
    # We already did 1. So ~19 more.
    
    max_kicks = 15  # Conservative
    
    for k in range(max_kicks):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.9:
            break
        
        # Kick: perturb current best positions
        # The amount of perturbation should be small enough to stay in the "basin"
        # but large enough to cross barriers.
        # Typical barrier height is not known, but distance scale is ~1.
        # Let's try sigma = 0.1 to 0.5
        sigma = 0.1 + 0.1 * k  # Increase perturbation slightly over time
        
        perturbed_pos = best_positions + rng.normal(0, sigma, size=(size, 3))
        
        # Ensure minimum distance 0.5 is maintained after perturbation?
        # The relax function might handle this, or it might fail if atoms overlap.
        # Let's check and fix overlaps before calling relax.
        # If atoms are too close, push them apart.
        
        # Simple overlap fix
        for i in range(size):
            for j in range(i + 1, size):
                d = np.linalg.norm(perturbed_pos[i] - perturbed_pos[j])
                if d < 0.5:
                    # Push apart
                    if d > 1e-6:
                        dir_vec = (perturbed_pos[i] - perturbed_pos[j]) / d
                        push = (0.5 - d) / 2.0
                        perturbed_pos[i] += dir_vec * push
                        perturbed_pos[j] -= dir_vec * push
                    else:
                        # Random push
                        rand_dir = rng.normal(size=3)
                        rand_dir /= np.linalg.norm(rand_dir)
                        perturbed_pos[i] += rand_dir * 0.25
                        perturbed_pos[j] -= rand_dir * 0.25
        
        try:
            energy, positions = relax(perturbed_pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            # If relax fails, skip this kick
            pass
            
    return best_positions
# EVOLVE-BLOCK-END
