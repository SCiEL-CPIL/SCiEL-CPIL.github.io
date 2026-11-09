"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy (structurally different from parent):
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Binary L/H pattern search via a coarse-to-fine combinatorial grid:
     - First, find the best 20-layer binary pattern (L/H) using a fast
       simulated-annealing-style swap search that only evaluates the
       *pattern* (with quarter-wave thicknesses as a proxy).
     - This gives a good discrete structure.
  3. Then, optimize the *thicknesses* of that fixed pattern using
     scipy.optimize.minimize (L-BFGS-B) with a penalty for constraint violations.
  4. Repeat with several different patterns (best from SA, plus a few
     hand-designed gradient patterns) to diversify.
  5. Deadline guard: return best feasible result found.

The key structural difference: the parent uses a *continuous* gradient
parameterization (t0, t1) with fixed alternating L/H. Here we treat the
L/H pattern as a discrete combinatorial variable (2^20 possibilities,
searched via heuristic) and then do a full continuous thickness
optimization on the fixed pattern. This decouples the two design decisions.
"""

import time
import math
import random
import numpy as np
from scipy.optimize import minimize

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    n_ds = ns[:, None] * ds[:, None]
    delta = 2.0 * np.pi * n_ds / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _quarter_wave_thicknesses(materials, problem, wl_center):
    """Return quarter-wave thicknesses for each layer at center wavelength."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    ths = []
    for m in materials:
        n = n_mat[m]
        ths.append(wl_center / (4.0 * n))
    return ths


def _pattern_score(materials, problem, wl_center):
    """Fast proxy score: evaluate reflectance with quarter-wave thicknesses."""
    ths = _quarter_wave_thicknesses(materials, problem, wl_center)
    return _mean_reflectance(materials, ths, problem)


def _sa_pattern_search(n_layers, problem, wl_center, max_iters=500, deadline=None):
    """Simulated annealing to find a good L/H pattern."""
    # Start with alternating L/H
    pattern = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    current_score = _pattern_score(pattern, problem, wl_center)
    best_pattern = list(pattern)
    best_score = current_score
    
    temp = 1.0
    cool_rate = 0.995
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Randomly swap two layers
        if n_layers >= 2:
            i, j = random.sample(range(n_layers), 2)
            pattern[i], pattern[j] = pattern[j], pattern[i]
        else:
            pattern[0] = 'H' if pattern[0] == 'L' else 'L'
        
        new_score = _pattern_score(pattern, problem, wl_center)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_pattern = list(pattern)
        else:
            # Revert
            if n_layers >= 2:
                pattern[i], pattern[j] = pattern[j], pattern[i]
            else:
                pattern[0] = 'H' if pattern[0] == 'L' else 'L'
        
        temp *= cool_rate
        if temp < 0.01:
            break
    
    return best_pattern, best_score


def _optimize_thicknesses(materials, problem, deadline, max_layer, max_total):
    """Optimize thicknesses for a fixed pattern using L-BFGS-B."""
    n = len(materials)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    wl_center = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Initial guess: quarter-wave
    x0 = np.array([wl_center / (4.0 * n) for n in ns])
    x0 = np.clip(x0, 1.0, max_layer)
    
    # Ensure total constraint
    if np.sum(x0) > max_total:
        x0 *= max_total / np.sum(x0)
        x0 = np.clip(x0, 1.0, max_layer)
    
    def objective(x):
        # Penalty for constraint violations
        total = np.sum(x)
        if total > max_total:
            penalty = (total - max_total) * 0.1
        else:
            penalty = 0.0
        # Clamp for evaluation
        x_clamped = np.clip(x, 0.0, max_layer)
        if np.sum(x_clamped) > max_total:
            x_clamped *= max_total / np.sum(x_clamped)
        score = _mean_reflectance(materials, list(x_clamped), problem)
        return score + penalty
    
    def constraint_total(x):
        return max_total - np.sum(x)
    
    constraints = [{'type': 'ineq', 'fun': constraint_total}]
    bounds = [(1.0, max_layer) for _ in range(n)]
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x_opt = result.x
    except Exception:
        x_opt = x0
    
    # Final clamping
    x_opt = np.clip(x_opt, 1.0, max_layer)
    if np.sum(x_opt) > max_total:
        x_opt *= max_total / np.sum(x_opt)
        x_opt = np.clip(x_opt, 1.0, max_layer)
    
    final_score = _mean_reflectance(materials, list(x_opt), problem)
    return list(x_opt), final_score


def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    # --- Phase 1: Find good patterns via SA for several layer counts ---
    # Try a range of layer counts
    layer_counts = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    layer_counts = [lc for lc in layer_counts if lc <= max_layers]
    
    patterns_to_optimize = []
    
    for n_layers in layer_counts:
        if time.time() >= deadline - 30:
            break
        
        # Run SA pattern search
        sa_deadline = min(deadline - 25, start_time + (deadline - start_time) * 0.5)
        pattern, pat_score = _sa_pattern_search(
            n_layers, problem, wl_center,
            max_iters=300, deadline=sa_deadline
        )
        patterns_to_optimize.append((pattern, pat_score, n_layers))
    
    # Also add some hand-designed patterns
    # Gradient: start with L (low index) near air, end with H near substrate
    for n_layers in [6, 8, 10, 12]:
        if n_layers > max_layers:
            continue
        # Alternating starting with L
        pattern = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        patterns_to_optimize.append((pattern, None, n_layers))
        
        # Alternating starting with H
        pattern2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        patterns_to_optimize.append((pattern2, None, n_layers))
    
    # Sort by pattern score (None goes to end)
    patterns_to_optimize.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
    
    # --- Phase 2: Optimize thicknesses for top patterns ---
    max_patterns_to_opt = 12
    for pattern, pat_score, n_layers in patterns_to_optimize[:max_patterns_to_opt]:
        if time.time() >= deadline - 10:
            break
        
        ths, score = _optimize_thicknesses(pattern, problem, deadline, max_layer, max_total)
        
        # Validate
        if len(ths) == n_layers and all(0 <= t <= max_layer for t in ths) and sum(ths) <= max_total:
            if score < best_score:
                best_score = score
                best_result = (list(pattern), list(ths))
    
    # --- Phase 3: If time remains, try more random patterns with quick optimization ---
    if time.time() < deadline - 15:
        for attempt in range(20):
            if time.time() >= deadline - 10:
                break
            n_layers = random.randint(4, min(max_layers, 16))
            pattern = [random.choice(['L', 'H']) for _ in range(n_layers)]
            
            ths, score = _optimize_thicknesses(pattern, problem, deadline, max_layer, max_total)
            
            if len(ths) == n_layers and all(0 <= t <= max_layer for t in ths) and sum(ths) <= max_total:
                if score < best_score:
                    best_score = score
                    best_result = (list(pattern), list(ths))
    
    # --- Fallback ---
    if best_result is None:
        qw_L = wl_center / (4.0 * 1.38)
        return ['L'], [qw_L]
    
    mats, ths = best_result
    return list(mats), list(ths)
