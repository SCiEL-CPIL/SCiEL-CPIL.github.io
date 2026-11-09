"""
Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Fast exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Structural representation: Fix the sequence to a 20-layer alternating "H L H L..." pattern.
     This maximizes index contrast and degrees of freedom.
  3. Optimization: Nelder-Mead Simplex with multi-start initialization.
     - Initialization: Start from quarter-wave thicknesses at the center wavelength, 
       scaled to fit the total thickness constraint.
     - Multi-start: Run multiple Nelder-Mead instances from slightly perturbed initial 
       points to escape local minima.
     - Time management: Strict deadline checking to ensure completion within 150s.
"""

import time
import math
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    # Build per-layer optical path lengths
    # delta_j = 2*pi*n_j*d_j/lambda
    n_ds = np.array([n * d for n, d in zip(ns, ds)])  # (N,)
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]  # (N, K)
    c = np.cos(delta)
    s = np.sin(delta)

    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        # Update A,B,C,D
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    # Reflectance: r = (n_0 * (A + B * n_s) - (C + D * n_s)) / (n_0 * (A + B * n_s) + (C + D * n_s))
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def _nelder_mead(objective, x0, bounds, max_iter=100, tol=1e-6, deadline=None):
    """
    Nelder-Mead Simplex optimization.
    objective: function taking a numpy array, returning a float.
    x0: initial guess.
    bounds: list of (min, max).
    """
    n = len(x0)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        pt = x0.copy()
        # Step size: 10% of the range or a small absolute value
        step = 0.1 * (upper[i] - lower[i]) if (upper[i] - lower[i]) > 0 else 0.1
        pt[i] += step
        simplex.append(pt)
        
    # Ensure initial points are within bounds
    for i in range(n + 1):
        simplex[i] = np.clip(simplex[i], lower, upper)

    values = [objective(pt) for pt in simplex]
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Sort simplex by objective value
        idx = np.argsort(values)
        simplex = [simplex[i] for i in idx]
        values = [values[i] for i in idx]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
            
        x_avg = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_avg + alpha * (x_avg - simplex[-1])
        x_r = np.clip(x_r, lower, upper)
        f_r = objective(x_r)
        
        if f_r < values[0]:
            # Expansion
            x_e = x_avg + gamma * (x_r - x_avg)
            x_e = np.clip(x_e, lower, upper)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            x_c = x_avg + rho * (simplex[-1] - x_avg)
            x_c = np.clip(x_c, lower, upper)
            f_c = objective(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lower, upper)
                    values[i] = objective(simplex[i])
                    
    # Return best point
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]

def design(problem):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # Fixed structure: 20 layers, alternating H/L starting with H (High index near air? 
    # Actually, for AR on glass (n=1.52), starting with L (n=1.38) is closer to air (1.0).
    # Let's try starting with L to better match air index.
    # Sequence: L, H, L, H...
    N = 20
    materials = []
    for i in range(N):
        if i % 2 == 0:
            materials.append('L')
        else:
            materials.append('H')
            
    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)
    
    # Initial guess: quarter-wave thicknesses scaled to fit total constraint
    init_ths = np.array([qw[m] for m in materials])
    total_init = np.sum(init_ths)
    if total_init > max_total:
        init_ths = init_ths * (max_total / total_init)
    
    # Bounds
    bounds = [(0.0, max_layer)] * N
    
    def objective(ind):
        ths = np.clip(ind, 0, max_layer)
        total = np.sum(ths)
        if total > max_total:
            # Penalty for exceeding total thickness
            # Scale to valid region for evaluation to keep landscape smooth
            scale = max_total / total
            ths_valid = ths * scale
            score = _mean_reflectance(materials, ths_valid, problem)
            # Add penalty proportional to violation
            penalty = 1.0 * (total - max_total) / max_total
            return score + penalty
        else:
            return _mean_reflectance(materials, ths, problem)
            
    best_result = None
    best_score = float('inf')
    
    # Multi-start Nelder-Mead
    # We will run several restarts with different initial perturbations
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Perturb initial point
        if r == 0:
            x0 = init_ths.copy()
        else:
            # Random perturbation around the initial quarter-wave design
            # Or random within bounds
            if r % 2 == 0:
                # Small perturbation
                x0 = init_ths + np.random.normal(0, 10, size=N)
            else:
                # Larger perturbation / random
                x0 = np.random.uniform(0, max_layer, size=N)
                
        x0 = np.clip(x0, 0, max_layer)
        
        # Run Nelder-Mead
        try:
            best_ind, best_sc = _nelder_mead(
                objective, 
                x0, 
                bounds, 
                max_iter=50, 
                deadline=deadline - 2
            )
            
            # Final validation and clipping
            best_ths = np.clip(best_ind, 0, max_layer)
            total_ths = np.sum(best_ths)
            if total_ths > max_total:
                best_ths = best_ths * (max_total / total_ths)
            
            # Re-evaluate exact score for the valid, clipped thickness
            final_score = _mean_reflectance(materials, list(best_ths), problem)
            
            if final_score < best_score:
                best_score = final_score
                best_result = (materials, list(best_ths))
                
        except Exception as e:
            pass

    # If no valid result found, return a simple 1-layer AR coating
    if best_result is None:
        # 1 layer L
        return ["L"], [qw["L"]]
        
    mats, ths = best_result
    return list(mats), list(ths)
