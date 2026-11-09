"""Dzugutov cluster global minima search.
Uses physics-driven self-assembly via Langevin dynamics to exploit the
Dzugutov potential's inherent preference for polytetrahedral order,
combined with basin hopping for final refinement."""

import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
import autoresearch_suite.chem_clusters as chem_clusters
from autoresearch_suite.chem_clusters import relax

# Dzugutov Potential Parameters
A_POT = 5.82
B_POT = 1.28
A_CUT = 1.87
B_CUT = 1.94
C_EXP = 1.10
D_EXP = 0.27
M_EXP = 16

# Global deadline guard
TOTAL_TIME_LIMIT = 295.0
_start_time = None
_best_results = {}


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _remaining():
    return max(0.0, TOTAL_TIME_LIMIT - _elapsed())


def _dzugutov_energy_gradient(positions):
    """Compute Dzugutov energy and gradient."""
    try:
        energy, grad = chem_clusters.dzugutov_energy_gradient(positions)
        return energy, grad
    except Exception:
        pass

    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)

    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-8:
                continue

            if r < A_CUT:
                V = (A_POT * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
                dVdr = A_POT * (-M_EXP * r ** (-M_EXP - 1) - C_EXP / (r - A_CUT) ** 2 * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
            elif r < B_CUT:
                V = B_POT * np.exp(D_EXP / (r - B_CUT))
                dVdr = B_POT * np.exp(D_EXP / (r - B_CUT)) * (-D_EXP / (r - B_CUT) ** 2)
            else:
                continue

            energy += V
            force_factor = dVdr / r
            grad[i] -= force_factor * r_vec
            grad[j] += force_factor * r_vec

    return energy, grad


def _langevin_self_assemble(n, time_budget, seed, dt=0.001, gamma=0.5, T_init=0.3, T_final=0.01):
    """
    Physics-driven self-assembly using Langevin dynamics.
    Start with random positions and let the Dzugutov potential drive
    the system toward polytetrahedral order through thermal dynamics.
    """
    rng = np.random.default_rng(seed)
    
    # Initial random positions in a compact sphere
    radius = 1.5 * n ** (1.0 / 3.0)
    positions = rng.uniform(-radius, radius, (n, 3))
    velocities = rng.normal(0, np.sqrt(T_init), (n, 3))
    
    # Enforce minimum distance initially
    positions = _enforce_min_distance(positions)
    
    t_start = time.time()
    n_steps = 0
    max_steps = int(time_budget / dt * 0.8)  # Safety margin
    
    # Adaptive temperature schedule
    for step in range(max_steps):
        if time.time() - t_start > time_budget:
            break
        if _remaining() < 10:
            break
        
        # Linear cooling schedule
        progress = step / max(1, max_steps)
        T = T_init + (T_final - T_init) * progress
        
        # Compute forces
        _, grad = _dzugutov_energy_gradient(positions)
        forces = -grad
        
        # Langevin update: v += (F/m - gamma*v + random)*dt; x += v*dt
        # Mass = 1, so F/m = F
        random_force = rng.normal(0, np.sqrt(2 * gamma * T / dt), (n, 3))
        velocities = velocities * (1 - gamma * dt) + forces * dt + random_force
        positions = positions + velocities * dt
        
        # Enforce minimum distance every 10 steps
        if step % 10 == 0:
            positions = _enforce_min_distance(positions, min_dist=0.5)
        
        n_steps += 1
    
    # Final relaxation
    energy, final_pos = _relax_positions(positions)
    return energy, final_pos


def _make_seeded_start(n, rng=None):
    """
    Create a diverse starting configuration for self-assembly.
    Mix of random and slightly structured initial conditions.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    strategy = rng.integers(0, 4)
    
    if strategy == 0:
        # Pure random in sphere
        radius = 1.3 * n ** (1.0 / 3.0)
        positions = rng.uniform(-radius, radius, (n, 3))
    elif strategy == 1:
        # Random with some local clustering
        radius = 1.2 * n ** (1.0 / 3.0)
        positions = []
        for _ in range(n):
            if positions and rng.random() < 0.3:
                # Place near an existing atom
                idx = rng.integers(0, len(positions))
                center = np.array(positions[idx])
                p = center + rng.normal(0, 0.8, 3)
            else:
                p = rng.uniform(-radius, radius, 3)
            positions.append(p)
        positions = np.array(positions)
    elif strategy == 2:
        # Shell-based random (loose icosahedral-ish)
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts = verts / norms
        
        positions = [np.zeros(3)]
        remaining = n - 1
        r1 = 1.0
        for v in verts[:min(12, remaining)]:
            positions.append(v * r1 + rng.normal(0, 0.15, 3))
            remaining -= 1
        
        radius_outer = 1.8 * n ** (1.0 / 3.0)
        for _ in range(remaining):
            p = rng.uniform(-radius_outer, radius_outer, 3)
            if np.linalg.norm(p) <= radius_outer:
                positions.append(p)
        
        positions = np.array(positions[:n])
    else:
        # Random with minimum spacing enforced
        radius = 1.4 * n ** (1.0 / 3.0)
        positions = []
        min_dist = 0.6
        attempts = 0
        while len(positions) < n and attempts < 10000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in positions):
                    positions.append(p)
            attempts += 1
        
        while len(positions) < n:
            p = rng.uniform(-radius, radius, 3)
            positions.append(p)
        
        positions = np.array(positions[:n])
    
    return _enforce_min_distance(positions)


def _enforce_min_distance(positions, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    for _ in range(30):
        needs_fix = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm < 1e-8:
                        diff = np.array([1.0, 0.0, 0.0])
                    else:
                        diff = diff / diff_norm
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos


def _relax_positions(positions):
    """Relax positions using the relax API, with L-BFGS fallback."""
    try:
        energy, pos = relax(positions, 'dzugutov')
        if energy is not None and pos is not None:
            return energy, pos
    except Exception:
        pass

    # Fallback: L-BFGS optimization
    def objective_with_grad(x):
        pos = x.reshape(-1, 3)
        try:
            energy, grad = _dzugutov_energy_gradient(pos)
            return energy, grad.flatten()
        except Exception:
            return 1e10, np.zeros_like(x)

    x0 = positions.flatten()
    best_energy = 1e10
    best_x = x0.copy()

    try:
        result = minimize(
            objective_with_grad, x0, method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    return best_energy, best_x.reshape(-1, 3)


def _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed):
    """Basin hopping: kick, relax, accept/reject with Metropolis criterion."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_positions(positions)
    best_energy = current_energy
    best_pos = current_pos.copy()

    t_start = time.time()
    T = 0.3  # Initial temperature for Metropolis acceptance

    for kick_idx in range(n_kicks):
        if time.time() - t_start > time_budget:
            break
        if _remaining() < 5:
            break

        # Adaptive kick scale: larger kicks early, smaller later
        progress = kick_idx / max(1, n_kicks)
        current_kick_scale = kick_scale * (1.0 - 0.6 * progress)

        # Kick: random perturbation
        kick = rng.normal(0, current_kick_scale, positions.shape)
        kicked = current_pos + kick
        kicked = _enforce_min_distance(kicked)

        # Relax
        new_energy, new_pos = _relax_positions(kicked)

        if new_energy is None:
            continue

        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_e / T)

        if accept:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()

        # Cool down
        T *= 0.95

    return best_energy, best_pos


def _worker_basin_hopping(args):
    """Worker for parallel basin hopping."""
    positions, n_kicks, kick_scale, time_budget, seed = args
    try:
        best_energy, best_pos = _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed)
        return best_energy, best_pos
    except Exception:
        return None, None


def _worker_self_assemble(args):
    """Worker for parallel self-assembly via Langevin dynamics."""
    n, time_budget, seed = args
    try:
        energy, pos = _langevin_self_assemble(n, time_budget, seed)
        return energy, pos
    except Exception:
        return None, None


def construct(instance):
    """Find low-energy structures of Dzugutov clusters using physics-driven self-assembly."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = instance["n"]

    if n in _best_results:
        return _best_results[n]

    # Time budget for this instance
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    base_time_per_instance = TOTAL_TIME_LIMIT / 6.0
    time_budget = min(_remaining() * 0.9, base_time_per_instance * 1.5)
    time_budget = max(time_budget, 15.0)

    best_energy = np.inf
    best_positions = None

    # Phase 1: Parallel self-assembly via Langevin dynamics (main approach)
    # Use multiple independent runs with different random seeds
    num_sa_runs = 4
    per_run_time = time_budget * 0.55 / num_sa_runs

    sa_args = []
    for i in range(num_sa_runs):
        seed = 5000 + n * 100 + i
        sa_args.append((n, per_run_time, seed))

    try:
        with mp.Pool(processes=num_sa_runs) as pool:
            sa_results = pool.map(_worker_self_assemble, sa_args)

        for energy, pos in sa_results:
            if energy is not None and pos is not None:
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos

    except Exception:
        # Fallback: sequential self-assembly
        try:
            energy, pos = _langevin_self_assemble(n, time_budget * 0.4, seed=7777)
            if energy is not None and pos is not None:
                best_energy = energy
                best_positions = pos
        except Exception:
            pass

    # Phase 2: Basin hopping refinement from the best self-assembled structure
    if best_positions is not None and _remaining() > 10:
        bh_time_budget = min(_remaining() * 0.35, time_budget * 0.35)
        
        # Create diverse starting points for basin hopping
        rng = np.random.default_rng(12345 + n)
        bh_starts = [best_positions.copy()]
        
        # Add perturbed versions
        for i in range(3):
            perturbed = best_positions + rng.normal(0, 0.1, best_positions.shape)
            bh_starts.append(_enforce_min_distance(perturbed))
        
        # Also add a fresh self-assembly result if time allows
        if _remaining() > 15:
            try:
                extra_energy, extra_pos = _langevin_self_assemble(
                    n, min(_remaining() * 0.2, 8.0), seed=9999 + n
                )
                if extra_pos is not None:
                    bh_starts.append(extra_pos)
            except Exception:
                pass

        # Parallel basin hopping from top starts
        num_bh_workers = min(4, len(bh_starts))
        per_bh_time = bh_time_budget / num_bh_workers
        
        base_kicks = max(10, int(50 / (n / 40)))
        kick_scale = 0.10 + 0.03 * (n / 60)

        bh_args = []
        for i, start_pos in enumerate(bh_starts[:num_bh_workers]):
            seed = 8000 + n * 10 + i
            bh_args.append((start_pos, base_kicks, kick_scale, per_bh_time, seed))

        try:
            with mp.Pool(processes=num_bh_workers) as pool:
                bh_results = pool.map(_worker_basin_hopping, bh_args)

            for energy, pos in bh_results:
                if energy is not None and pos is not None:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = pos

        except Exception:
            # Fallback: sequential basin hopping
            if bh_starts:
                try:
                    energy, new_pos = _basin_hopping(
                        bh_starts[0], base_kicks, kick_scale, 
                        min(_remaining() * 0.3, 10.0), seed=4444
                    )
                    if energy is not None and energy < best_energy:
                        best_energy = energy
                        best_positions = new_pos
                except Exception:
                    pass

    # Final relaxation of the best result
    if best_positions is not None and _remaining() > 3:
        final_energy, final_pos = _relax_positions(best_positions)
        if final_energy is not None and final_pos is not None:
            best_energy = final_energy
            best_positions = final_pos

    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _enforce_min_distance(best_positions)
    else:
        # Ultimate fallback: random compact + relax
        rng = np.random.default_rng(42 + n)
        radius = 1.3 * n ** (1.0 / 3.0)
        fallback_pos = rng.uniform(-radius, radius, (n, 3))
        fallback_pos = _enforce_min_distance(fallback_pos)
        try:
            _, best_positions = _relax_positions(fallback_pos)
        except Exception:
            best_positions = fallback_pos

    _best_results[n] = best_positions
    return best_positions
