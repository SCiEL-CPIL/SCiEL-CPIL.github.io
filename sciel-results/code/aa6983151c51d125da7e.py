"""Construct the LJ38 global minimum (FCC truncated octahedron) and relax locally."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Deadline guard: return best found before 140s (safety margin under 150s limit)
_DEADLINE = 140.0
_start_time = None


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for a set of positions."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Avoid division by zero (atoms at same position)
    r2 = np.maximum(r2, 1e-12)
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    r6 = np.where(mask, r2 ** (-3), 0.0)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    
    # Gradient: dE/dri = 4 * sum_{j != i} (-12*r^-14 + 6*r^-8) * (ri - rj)
    coeff = np.where(mask, 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4)), 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _build_fcc_truncated_octahedron(scale=1.0):
    """
    Build the FCC truncated octahedron for LJ38.
    """
    # Generate FCC lattice points
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])
    
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    points.append(p)
    
    points = np.array(points)
    
    # Center the points
    center = points.mean(axis=0)
    points = points - center
    
    # Compute distances from center
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance and take the 38 closest
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Apply scale
    points = points * scale
    
    return points


def construct():
    global _start_time
    _start_time = time.time()
    
    def _elapsed():
        return time.time() - _start_time
    
    def _remaining():
        return _DEADLINE - _elapsed()
    
    # Strategy: Build the FCC truncated octahedron, optimize scale, then relax with L-BFGS
    
    # Build initial guess with optimized scale
    # LJ equilibrium distance is 2^(1/6) ≈ 1.122462
    optimal_scale = 2.0 ** (-1.0/3.0)
    pos0 = _build_fcc_truncated_octahedron(scale=optimal_scale)
    
    # Ensure minimum distance constraint
    diff = pos0[:, np.newaxis, :] - pos0[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    if min_r2 < 0.5 ** 2:
        scale_factor = 0.5 / np.sqrt(min_r2) * 1.01
        pos0 = pos0 * scale_factor
    
    # Relax using L-BFGS
    best_pos = pos0.copy()
    best_energy = _lj_energy_and_grad(pos0)[0]
    
    def energy_func(x):
        return _lj_energy_and_grad(x.reshape(-1, 3))[0]
    
    def grad_func(x):
        return _lj_energy_and_grad(x.reshape(-1, 3))[1].flatten()
    
    # Try relaxation from the FCC structure
    try:
        result = minimize(
            energy_func,
            pos0.flatten(),
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000}
        )
        
        if result.success or result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Fast-path: if we're already at the known global minimum, skip basin hopping
    # and dedicate all remaining time to ultra-tight polish
    KNOWN_GLOBAL_MIN = -173.928427
    if best_energy > KNOWN_GLOBAL_MIN - 0.01:
        # We're close to the global minimum; do ultra-tight polish
        if _remaining() > 10.0:
            try:
                result = minimize(
                    energy_func,
                    best_pos.flatten(),
                    jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 50000}
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.reshape(-1, 3)
            except Exception:
                pass
        
        # Final ultra-fine polish if time remains
        if _remaining() > 5.0:
            try:
                result = minimize(
                    energy_func,
                    best_pos.flatten(),
                    jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 20000}
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.reshape(-1, 3)
            except Exception:
                pass
    else:
        # Not at global minimum; do basin hopping with multiple restarts
        rng = np.random.default_rng(42)
        
        # Structure 2: Slightly perturbed FCC
        pos_perturbed = best_pos + rng.normal(0, 0.05, best_pos.shape)
        try:
            result = minimize(
                energy_func,
                pos_perturbed.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
        except Exception:
            pass
        
        # Structure 3: More perturbed FCC
        if _remaining() > 20.0:
            pos_perturbed2 = best_pos + rng.normal(0, 0.15, best_pos.shape)
            try:
                result = minimize(
                    energy_func,
                    pos_perturbed2.flatten(),
                    jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.reshape(-1, 3)
            except Exception:
                pass
        
        # Final polish
        if _remaining() > 10.0:
            try:
                result = minimize(
                    energy_func,
                    best_pos.flatten(),
                    jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000}
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.reshape(-1, 3)
            except Exception:
                pass
    
    # Final check: ensure minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        scale_factor = 0.5 / min_r * 1.001
        best_pos = best_pos * scale_factor
    
    return best_pos

# EVOLVE-BLOCK-END
