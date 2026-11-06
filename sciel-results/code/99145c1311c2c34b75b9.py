"""Morse cluster global minima via Simulated Annealing and Local Relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import morse_energy_gradient

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 290.0
    return _T0


def _time_left():
    global _HARD_DEADLINE
    return max(0.0, _HARD_DEADLINE - time.time())


def _enforce_min_dist(P, min_d=0.5, max_iter=50):
    """
    Vectorized-ish enforcement of minimum distance.
    Since N is small (<=80), a simple loop is fast enough and robust.
    """
    P = np.array(P, dtype=float)
    n = P.shape[0]
    for _ in range(max_iter):
        moved = False
        # Calculate all pairwise distances
        # diff[i, j] = P[i] - P[j]
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Find pairs too close
        # We only need to check i < j
        iu = np.triu_indices(n, k=1)
        close_mask = dist[iu] < min_d
        
        if not np.any(close_mask):
            break
        
        # Get indices of close pairs
        i_idx, j_idx = iu[0][close_mask], iu[1][close_mask]
        
        for i, j in zip(i_idx, j_idx):
            d = P[i] - P[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij < 1e-12:
                d = np.array([1.0, 0.0, 0.0])
            else:
                d = d / dist_ij
            
            # Push apart
            push = (min_d - dist_ij) / 2.0 + 0.001
            P[i] += push * d
            P[j] -= push * d
            moved = True
            
        if not moved:
            break
            
    return P


def _simulated_annealing(positions, rho, rng, time_budget=10.0, T_start=50.0, T_end=1e-4, cooling_rate=0.999):
    """
    Vectorized Simulated Annealing using Morse gradient.
    """
    n = len(positions)
    P = np.array(positions, dtype=float)
    
    try:
        E, grad = morse_energy_gradient(P, rho)
    except Exception:
        return P, float('inf')
    
    if not np.isfinite(E):
        return P, float('inf')
        
    best_P = P.copy()
    best_E = E
    current_E = E
    
    T = T_start
    step_size = 0.1  # Initial step size for gradient descent part
    noise_scale = 0.5 # Initial noise scale
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget and T > T_end:
        if _time_left() < 5.0:
            break
            
        # Update
        # Gradient descent step
        P_new = P - step_size * grad
        
        # Add noise
        # Scale noise with temperature
        noise = rng.normal(0, noise_scale * np.sqrt(T), P.shape)
        P_new += noise
        
        # Enforce constraints
        P_new = _enforce_min_dist(P_new, 0.5)
        
        # Compute new energy
        try:
            E_new, grad_new = morse_energy_gradient(P_new, rho)
        except Exception:
            break
            
        if not np.isfinite(E_new):
            # Revert if invalid
            P_new = P
            E_new = current_E
            grad_new = grad
            # Reduce noise to avoid invalid states
            noise_scale *= 0.9
            continue
            
        # Metropolis Acceptance
        delta_E = E_new - current_E
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            P = P_new
            current_E = E_new
            grad = grad_new
            
            if current_E < best_E:
                best_E = current_E
                best_P = P.copy()
                
        # Cool down
        T *= cooling_rate
        # Slowly reduce step size and noise as T drops
        # This helps in fine-tuning
        factor = T / T_start
        step_size = 0.1 * (0.1 + 0.9 * factor)
        noise_scale = 0.5 * (0.1 + 0.9 * factor)
        
    return best_P, best_E


def _random_gas_init(n, rho, rng):
    """
    Initialize atoms in a large sphere (ideal gas).
    Radius is chosen to be large enough to avoid initial overlaps but small enough to converge.
    """
    # For N=80, R ~ 4.5. Start with R ~ 10-15
    R = 15.0
    # Uniform distribution in sphere
    u = rng.random((n, 3))
    v = rng.random((n, 3))
    w = rng.random((n, 3))
    
    # Spherical coordinates
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    r = R * np.cbrt(w)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    P = np.column_stack((x, y, z))
    # Center
    P -= P.mean(axis=0)
    
    # Ensure no initial overlaps (though SA will fix it, better to start clean)
    P = _enforce_min_dist(P, 0.5)
    
    return P


def construct(instance):
    """Find low-energy Morse cluster configurations using Simulated Annealing."""
    _init_timer()
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(size * 1000 + int(rho))
    
    best_energy = float('inf')
    best_positions = None
    
    # Allocate time budget
    # Total ~45s per instance.
    # Phase 1: SA from random gas (20s)
    # Phase 2: Relax (1s)
    # Phase 3: SA from relaxed (15s)
    # Phase 4: Perturb & Relax (remaining)
    
    # Phase 1: Simulated Annealing from Random Gas
    t1 = time.time()
    init_pos = _random_gas_init(size, rho, rng)
    
    # We can run multiple short SA runs or one long one.
    # One long one is better for global search.
    sa_time = 20.0
    if _time_left() > sa_time + 5.0:
        pos1, E1 = _simulated_annealing(
            init_pos, rho, rng, 
            time_budget=sa_time, 
            T_start=50.0, 
            T_end=1e-4, 
            cooling_rate=0.9995 # Slower cooling for better exploration
        )
        
        if np.isfinite(E1):
            # Polish with local relaxation
            try:
                E1_relaxed, pos1_relaxed = relax(pos1, potential, rho)
                if E1_relaxed < best_energy:
                    best_energy = E1_relaxed
                    best_positions = pos1_relaxed.copy()
            except Exception:
                if E1 < best_energy:
                    best_energy = E1
                    best_positions = pos1.copy()
        else:
            # Fallback if SA failed
            try:
                E1_relaxed, pos1_relaxed = relax(init_pos, potential, rho)
                if E1_relaxed < best_energy:
                    best_energy = E1_relaxed
                    best_positions = pos1_relaxed.copy()
            except Exception:
                pass
                
    # Phase 2: Simulated Annealing from Best Found (Refinement)
    if best_positions is not None and _time_left() > 15.0:
        # Perturb slightly to escape local minimum if stuck
        # Or just run SA again with lower T_start
        pos2_start = best_positions.copy()
        # Add small noise
        pos2_start += rng.normal(0, 0.1, pos2_start.shape)
        pos2_start = _enforce_min_dist(pos2_start, 0.5)
        
        sa_time2 = min(15.0, _time_left() - 5.0)
        pos2, E2 = _simulated_annealing(
            pos2_start, rho, rng,
            time_budget=sa_time2,
            T_start=5.0,  # Lower start temp
            T_end=1e-5,
            cooling_rate=0.999
        )
        
        if np.isfinite(E2):
            try:
                E2_relaxed, pos2_relaxed = relax(pos2, potential, rho)
                if E2_relaxed < best_energy:
                    best_energy = E2_relaxed
                    best_positions = pos2_relaxed.copy()
            except Exception:
                if E2 < best_energy:
                    best_energy = E2
                    best_positions = pos2.copy()

    # Phase 3: Perturbation and Relaxation (Basin Hopping style)
    if best_positions is not None:
        # Use remaining time for perturb-relax cycles
        scales = [0.3, 0.2, 0.1, 0.05]
        for scale in scales:
            if _time_left() < 2.0:
                break
            # Try a few perturbations
            n_tries = 3
            for _ in range(n_tries):
                if _time_left() < 1.0:
                    break
                try:
                    perturbed = best_positions + rng.normal(0, scale, best_positions.shape)
                    perturbed = _enforce_min_dist(perturbed, 0.5)
                    E_pert, pos_pert = relax(perturbed, potential, rho)
                    if E_pert < best_energy:
                        best_energy = E_pert
                        best_positions = pos_pert.copy()
                except Exception:
                    continue

    # Fallback
    if best_positions is None:
        # Last resort: random compact
        P = _random_gas_init(size, rho, rng)
        P = _enforce_min_dist(P, 0.5)
        try:
            E, P = relax(P, potential, rho)
            best_positions = P
        except Exception:
            pass
            
    return best_positions
# EVOLVE-BLOCK-END
