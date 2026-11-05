"""Broadband antireflection coating design via multi-start local search."""
import time
import math
import random

import numpy as np

# Fixed deadline guard (wall-clock seconds)
_DEADLINE_SECONDS = 115.0

# Precompute wavelength grid (401 points from 400 to 800 nm inclusive)
_WLS = np.linspace(400.0, 800.0, 401)
_N_WLS = len(_WLS)


def _compute_mean_reflectance(materials, thicknesses, n_amb, n_sub, n_L, n_H):
    """
    Compute mean normal-incidence reflectance (in percent) over the precomputed
    wavelength grid using the thin-film characteristic-matrix method.
    
    Returns a float in percent.
    """
    n_layers = len(materials)
    if n_layers == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return 100.0 * r * r

    # Build the characteristic matrix for each layer
    # For a lossless dielectric layer with index n and optical thickness t (nm),
    # at wavelength lambda:
    #   M = [[cos(delta), i*sin(delta)/n],
    #        [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*t/lambda
    
    # We'll compute the product of all layer matrices, then combine with
    # ambient and substrate to get the reflection coefficient.
    
    # For efficiency, precompute for each wavelength the matrix product.
    # Total matrix M_total = M_1 * M_2 * ... * M_N
    # Then the admittance Y = (m11 + m12*n_sub) / (m21 + m22*n_sub)
    # Reflection coefficient r = (n_amb - Y) / (n_amb + Y)
    # Reflectance R = |r|^2
    
    total_reflectance = 0.0
    for wl in _WLS:
        # Start with identity matrix
        m11, m12 = 1.0 + 0.0j, 0.0 + 0.0j
        m21, m22 = 0.0 + 0.0j, 1.0 + 0.0j
        
        for i in range(n_layers):
            n_layer = n_L if materials[i] == 'L' else n_H
            t = thicknesses[i]
            delta = 2.0 * math.pi * n_layer * t / wl
            c = math.cos(delta)
            s = math.sin(delta)
            
            # Layer matrix
            l11 = c + 0.0j
            l12 = 1j * s / n_layer
            l21 = 1j * n_layer * s
            l22 = c + 0.0j
            
            # Multiply current total by layer matrix: M = M * L
            new_m11 = m11 * l11 + m12 * l21
            new_m12 = m11 * l12 + m12 * l22
            new_m21 = m21 * l11 + m22 * l21
            new_m22 = m21 * l12 + m22 * l22
            m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22
        
        # Admittance
        Y = (m11 + m12 * n_sub) / (m21 + m22 * n_sub)
        # Reflection coefficient
        r = (n_amb - Y) / (n_amb + Y)
        R = (r * r.conjugate()).real
        total_reflectance += R
    
    return 100.0 * total_reflectance / _N_WLS


def _clamp_thickness(t, max_layer_nm):
    return max(0.0, min(max_layer_nm, t))


def _is_valid(materials, thicknesses, max_layers, max_total_nm, max_layer_nm):
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    return True


def _random_materials(max_layers):
    """Generate a random stack of L/H layers."""
    n = random.randint(1, max_layers)
    return [random.choice(['L', 'H']) for _ in range(n)]


def _random_thicknesses(n, max_layer_nm, max_total_nm):
    """Generate random thicknesses that satisfy the total constraint."""
    # Generate n random values in [0, max_layer_nm]
    ts = [random.uniform(0.0, max_layer_nm) for _ in range(n)]
    total = sum(ts)
    if total > max_total_nm:
        scale = max_total_nm / total
        ts = [t * scale for t in ts]
    return ts


def _evaluate(materials, thicknesses, n_amb, n_sub, n_L, n_H, max_layers, max_total_nm, max_layer_nm):
    """Return (mean_reflectance_percent, is_valid)."""
    if not _is_valid(materials, thicknesses, max_layers, max_total_nm, max_layer_nm):
        return float('inf'), False
    return _compute_mean_reflectance(materials, thicknesses, n_amb, n_sub, n_L, n_H), True


def _local_search(materials, thicknesses, n_amb, n_sub, n_L, n_H, max_layers, max_total_nm, max_layer_nm, deadline):
    """
    Local search: perturb thicknesses and occasionally change layer count/material.
    Returns best (materials, thicknesses, score) found.
    """
    best_score = _compute_mean_reflectance(materials, thicknesses, n_amb, n_sub, n_L, n_H)
    best_mats = list(materials)
    best_ths = list(thicknesses)
    
    # Step size for thickness perturbation
    step = 30.0
    
    while time.time() < deadline:
        # Random perturbation of thicknesses
        new_ths = []
        for t in best_ths:
            new_t = t + random.gauss(0.0, step)
            new_ths.append(_clamp_thickness(new_t, max_layer_nm))
        
        # Check total constraint
        if sum(new_ths) > max_total_nm:
            scale = max_total_nm / sum(new_ths)
            new_ths = [t * scale for t in new_ths]
        
        new_score = _compute_mean_reflectance(best_mats, new_ths, n_amb, n_sub, n_L, n_H)
        
        if new_score < best_score:
            best_score = new_score
            best_ths = new_ths
            # Reduce step slightly
            step *= 0.95
        else:
            # Occasionally try changing layer count or material
            if random.random() < 0.1:
                # Try adding a layer
                if len(best_mats) < max_layers:
                    new_mats = best_mats + [random.choice(['L', 'H'])]
                    new_ths2 = list(best_ths) + [random.uniform(20.0, 150.0)]
                    if sum(new_ths2) <= max_total_nm:
                        s2 = _compute_mean_reflectance(new_mats, new_ths2, n_amb, n_sub, n_L, n_H)
                        if s2 < best_score:
                            best_score = s2
                            best_mats = new_mats
                            best_ths = new_ths2
                            step = 30.0
                # Try removing a layer
                elif len(best_mats) > 1:
                    idx = random.randint(0, len(best_mats) - 1)
                    new_mats = best_mats[:idx] + best_mats[idx+1:]
                    new_ths2 = best_ths[:idx] + best_ths[idx+1:]
                    s2 = _compute_mean_reflectance(new_mats, new_ths2, n_amb, n_sub, n_L, n_H)
                    if s2 < best_score:
                        best_score = s2
                        best_mats = new_mats
                        best_ths = new_ths2
                        step = 30.0
            # Try swapping a material
            elif len(best_mats) >= 1:
                idx = random.randint(0, len(best_mats) - 1)
                new_mats = list(best_mats)
                new_mats[idx] = 'H' if new_mats[idx] == 'L' else 'L'
                s2 = _compute_mean_reflectance(new_mats, best_ths, n_amb, n_sub, n_L, n_H)
                if s2 < best_score:
                    best_score = s2
                    best_mats = new_mats
                    step = 30.0
            
            step = max(5.0, step * 1.02)
    
    return best_mats, best_ths, best_score


def _quarter_wave_stack(n_L, n_H, n_sub, center_wl, n_layers):
    """
    Generate a symmetric quarter-wave stack.
    For broadband AR, we use a structure like: L H L H ... or similar.
    """
    # A common broadband AR approach: start with L, alternate, end with L
    # Thickness = lambda_center / (4 * n)
    mats = []
    ths = []
    for i in range(n_layers):
        if i % 2 == 0:
            mats.append('L')
            ths.append(center_wl / (4.0 * n_L))
        else:
            mats.append('H')
            ths.append(center_wl / (4.0 * n_H))
    return mats, ths


def design(problem):
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    center_wl = 0.5 * (band[0] + band[1])
    
    # Generate a set of diverse initial candidates
    candidates = []
    
    # 1. Quarter-wave stacks of various lengths
    for n in [1, 3, 5, 7, 9, 11, 13, 15]:
        if n <= max_layers:
            mats, ths = _quarter_wave_stack(n_L, n_H, n_sub, center_wl, n)
            # Check total thickness
            if sum(ths) <= max_total_nm and all(t <= max_layer_nm for t in ths):
                candidates.append((mats, ths))
    
    # 2. Single high-index layer
    candidates.append((["H"], [center_wl / (4.0 * n_H)]))
    
    # 3. Random stacks
    random.seed(42)
    for _ in range(10):
        mats = _random_materials(max_layers)
        ths = _random_thicknesses(len(mats), max_layer_nm, max_total_nm)
        if _is_valid(mats, ths, max_layers, max_total_nm, max_layer_nm):
            candidates.append((mats, ths))
    
    # 4. Known good structures: alternating starting with L, symmetric
    for n in [3, 5, 7]:
        if n <= max_layers:
            mats = []
            ths = []
            # Symmetric: L H ... H L
            for i in range(n):
                if i == 0 or i == n - 1:
                    mats.append('L')
                    ths.append(center_wl / (4.0 * n_L))
                elif i % 2 == 1:
                    mats.append('H')
                    ths.append(center_wl / (4.0 * n_H))
                else:
                    mats.append('L')
                    ths.append(center_wl / (4.0 * n_L))
            if sum(ths) <= max_total_nm and all(t <= max_layer_nm for t in ths):
                candidates.append((mats, ths))
    
    # Evaluate all candidates and pick the best few for local search
    scored = []
    for mats, ths in candidates:
        score = _compute_mean_reflectance(mats, ths, n_amb, n_sub, n_L, n_H)
        scored.append((score, mats, ths))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top candidates for local search
    best_overall_score = float('inf')
    best_overall_mats = candidates[0][0] if candidates else ['L']
    best_overall_ths = candidates[0][1] if candidates else [center_wl / (4.0 * n_L)]
    
    # Local search on top candidates
    n_search = min(5, len(scored))
    for i in range(n_search):
        if time.time() > deadline - 10:
            break
        _, mats, ths = scored[i]
        s_mats, s_ths, s_score = _local_search(
            mats, ths, n_amb, n_sub, n_L, n_H,
            max_layers, max_total_nm, max_layer_nm, deadline
        )
        if s_score < best_overall_score:
            best_overall_score = s_score
            best_overall_mats = s_mats
            best_overall_ths = s_ths
    
    # Final validation
    if not _is_valid(best_overall_mats, best_overall_ths, max_layers, max_total_nm, max_layer_nm):
        # Fallback to simple quarter-wave
        best_overall_mats = ['L']
        best_overall_ths = [center_wl / (4.0 * n_L)]
    
    return best_overall_mats, best_overall_ths
