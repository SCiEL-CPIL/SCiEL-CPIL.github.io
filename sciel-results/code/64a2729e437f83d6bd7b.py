"""Dzugutov cluster global minima via physics-driven self-assembly with basin hopping."""
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 295.0  # safety margin before 300s kill

def _dzugutov_energy(pos):
    """Compute Dzugutov potential energy."""
    n = len(pos)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    E = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 0.5:
                return 1e10
            if r < a:
                E += A * (r**(-m) - B) * np.exp(c/(r - a))
            elif r < b:
                E += B * np.exp(d/(r - b))
    return E

def _relax_with_fallback(pos, potential, deadline):
    """Relax positions using the provided relax function, with fallback to gradient descent."""
    try:
        e, opt_pos = relax(pos, potential)
        return e, opt_pos
    except Exception:
        pass
    
    # Fallback: simple gradient descent
    try:
        pos = pos.copy()
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
            e, grad = dzugutov_energy_gradient(pos)
            pos_new = pos - lr * grad
            # Ensure minimum distance
            valid = True
            for i in range(len(pos_new)):
                for j in range(i+1, len(pos_new)):
                    if np.linalg.norm(pos_new[i] - pos_new[j]) < 0.5:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                pos = pos_new
            else:
                lr *= 0.5
        e = _dzugutov_energy(pos)
        return e, pos
    except Exception:
        e = _dzugutov_energy(pos)
        return e, pos

def _self_assemble(n, rng, deadline):
    """
    Physics-driven self-assembly: start with a small seed, add atoms one by one
    at random positions, relax after each addition.
    """
    # Start with a small icosahedron core (13 atoms)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    center = np.zeros(3)
    core = np.vstack([center, verts])
    
    # Scale core so nearest neighbor ~ 1.9
    # Current edge length is ~1.902, which is good
    pos = core[:min(n, 13)].copy()
    
    if n <= 13:
        return pos
    
    # Add remaining atoms one by one
    current_radius = np.max(np.linalg.norm(pos, axis=1))
    
    for atom_idx in range(len(pos), n):
        if time.time() > deadline - 10:
            break
        
        # Try to add atom at a random position
        added = False
        for attempt in range(100):
            # Random position on a shell
            r = rng.uniform(current_radius * 0.7, current_radius + 2.0)
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = rng.uniform(-np.pi/2, np.pi/2)
            
            p = np.array([
                r * np.cos(phi_angle) * np.cos(theta),
                r * np.cos(phi_angle) * np.sin(theta),
                r * np.sin(phi_angle)
            ])
            
            # Check minimum distance
            if np.all(np.linalg.norm(pos - p, axis=1) > 0.5):
                new_pos = np.vstack([pos, p])
                # Relax
                e, opt_pos = _relax_with_fallback(new_pos, 'dzugutov', deadline)
                if e < 1e9:  # Valid energy
                    pos = opt_pos
                    current_radius = np.max(np.linalg.norm(pos, axis=1))
                    added = True
                    break
        
        if not added:
            # Fallback: add at a fixed offset
            p = np.array([current_radius + 1.0, 0, 0])
            pos = np.vstack([pos, p])
            e, opt_pos = _relax_with_fallback(pos, 'dzugutov', deadline)
            pos = opt_pos
            current_radius = np.max(np.linalg.norm(pos, axis=1))
    
    return pos

def _basin_hopping(pos, potential, n_restarts, deadline):
    """
    Basin hopping: perturb the structure and relax to find lower energy minima.
    """
    best_energy = _dzugutov_energy(pos)
    best_pos = pos.copy()
    
    rng = np.random.default_rng(42)
    
    for k in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Perturb: random displacement
        perturbation = rng.uniform(-0.5, 0.5, size=pos.shape)
        perturbed_pos = pos + perturbation
        
        # Ensure minimum distance
        valid = True
        for i in range(len(perturbed_pos)):
            for j in range(i+1, len(perturbed_pos)):
                if np.linalg.norm(perturbed_pos[i] - perturbed_pos[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Relax
        e, opt_pos = _relax_with_fallback(perturbed_pos, potential, deadline)
        
        if e < best_energy:
            best_energy = e
            best_pos = opt_pos
            pos = opt_pos.copy()
    
    return best_pos, best_energy

def construct(instance):
    """
    Construct a low-energy Dzugutov cluster using physics-driven self-assembly.
    """
    n = instance['n']
    potential = instance['potential']
    
    # Time budget: ~45s per instance
    local_deadline = time.time() + 45.0
    
    rng = np.random.default_rng(n)
    
    # Multiple self-assembly attempts
    best_pos = None
    best_energy = np.inf
    
    n_assembly_attempts = 3
    
    for attempt in range(n_assembly_attempts):
        if time.time() > local_deadline - 10:
            break
        
        # Self-assemble
        pos = _self_assemble(n, rng, local_deadline)
        
        if len(pos) < n:
            # Pad with random positions if needed
            while len(pos) < n:
                p = rng.uniform(-3, 3, size=3)
                if np.all(np.linalg.norm(pos - p, axis=1) > 0.5):
                    pos = np.vstack([pos, p])
        
        # Basin hopping to refine
        refined_pos, energy = _basin_hopping(pos, potential, n_restarts=5, deadline=local_deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = refined_pos
    
    if best_pos is None:
        # Final fallback
        best_pos = _self_assemble(n, rng, local_deadline)
    
    return best_pos
