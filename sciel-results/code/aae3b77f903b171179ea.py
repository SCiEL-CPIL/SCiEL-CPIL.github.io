"""
Broadband antireflection coating design: Parametric Continuous Profile Discretization.

Strategy:
1. Define a continuous effective refractive index profile n(z) using a low-order polynomial
   in normalized position z in [0, 1]. The profile is constrained to stay within [n_L, n_H].
2. Optimize the polynomial coefficients (and possibly a scale/offset) to minimize the
   mean reflectance of the *continuous* approximation (or a high-resolution discrete
   approximation of it).
3. Discretize the optimal continuous profile into exactly 20 layers by mapping the
   optical path length into 20 equal segments. For each segment, determine the dominant
   material (L or H) and the corresponding thickness.
4. Polish the resulting 20-layer discrete stack using Nelder-Mead on the thicknesses
   to recover any discretization loss.
"""
import time
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
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # delta_j = 2*pi*n_j*d_j/lambda -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
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
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def _continuous_to_discrete(profile_params, problem, n_layers=20):
    """
    Convert a continuous profile parameter set to a discrete 20-layer stack.
    profile_params: [a0, a1, a2, a3] for n(z) = a0 + a1*z + a2*z^2 + a3*z^3
    z is normalized position in [0, 1].
    """
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    
    # Ensure profile stays within bounds by clipping during evaluation
    # But for discretization, we need to map the continuous index to L or H.
    
    # We discretize by optical path length (OPL).
    # Total OPL of continuous profile: integral n(z) dz from 0 to 1 (scaled by total physical thickness? No, z is normalized).
    # Actually, z is just a coordinate. Let's assume the continuous profile spans a virtual thickness T_virt.
    # The OPL contribution is n(z) * T_virt * dz.
    # We want to map this to 20 layers.
    
    # Let's define the continuous profile in terms of OPL fraction.
    # Let N(z) = a0 + a1*z + a2*z^2 + a3*z^3.
    # Clip N(z) to [n_L, n_H].
    
    # We want to find 20 layers (n_i, d_i) such that they approximate the continuous profile.
    # Strategy: Divide the OPL integral into 20 equal parts.
    # OPL(z) = integral_0^z N(t) dt.
    # Total OPL = OPL(1).
    # For layer k (0-indexed), the OPL range is [k/20 * TotalOPL, (k+1)/20 * TotalOPL].
    # Find z_k such that OPL(z_k) = k/20 * TotalOPL.
    # The physical thickness of layer k is d_k = z_{k+1} - z_k (assuming T_virt=1 for now, scale later).
    # The effective index of layer k is the average index over that segment? Or the index at the midpoint?
    # Since we must choose L or H, we choose the one closer to the average index of the segment.
    
    # 1. Sample N(z) finely
    z_samples = np.linspace(0, 1, 1001)
    N_vals = profile_params[0] + profile_params[1]*z_samples + profile_params[2]*z_samples**2 + profile_params[3]*z_samples**3
    N_vals = np.clip(N_vals, n_L, n_H)
    
    # 2. Compute cumulative OPL
    # Assume virtual thickness T_virt = 1.0.
    opl_cum = np.cumsum(N_vals) * (z_samples[1] - z_samples[0])
    total_opl = opl_cum[-1]
    
    # 3. Find boundaries z_k for equal OPL segments
    target_opls = np.linspace(0, total_opl, n_layers + 1)
    z_bounds = np.interp(target_opls, opl_cum, z_samples)
    
    # 4. Determine material and thickness for each layer
    materials = []
    thicknesses = []
    
    for k in range(n_layers):
        z_start = z_bounds[k]
        z_end = z_bounds[k+1]
        
        # Physical thickness (in virtual units)
        d_virtual = z_end - z_start
        
        # Average index in this segment
        mask = (z_samples >= z_start) & (z_samples <= z_end)
        if np.sum(mask) > 0:
            n_avg = np.mean(N_vals[mask])
        else:
            n_avg = N_vals[np.argmin(np.abs(z_samples - z_start))]
            
        # Assign material
        if abs(n_avg - n_L) < abs(n_avg - n_H):
            materials.append('L')
            n_assigned = n_L
        else:
            materials.append('H')
            n_assigned = n_H
            
        # Adjust thickness to match the OPL of the segment
        # OPL_segment = n_assigned * d_physical
        # OPL_target = integral N(z) dz over segment
        # We approximated OPL_target as (z_end - z_start) * n_avg? No, we used the exact integral from samples.
        # Let's compute the exact OPL for this segment from samples
        opl_seg = np.sum(N_vals[mask]) * (z_samples[1] - z_samples[0]) if np.sum(mask) > 0 else 0
        d_physical = opl_seg / n_assigned
        
        thicknesses.append(d_physical)
        
    # Scale thicknesses to fit max_total_nm
    # The virtual thickness was 1.0. The actual total thickness will be sum(thicknesses).
    # We need to scale them to be <= max_total_nm.
    # Also, the profile shape might need to be scaled? No, the relative OPL distribution is preserved.
    # But we must ensure the absolute thicknesses are reasonable.
    
    total_thick = sum(thicknesses)
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    if total_thick > 0:
        scale = max_total / total_thick
        thicknesses = [t * scale for t in thicknesses]
        
    # Clip individual layers
    thicknesses = [max(0.0, min(max_layer, t)) for t in thicknesses]
    
    return materials, thicknesses

def _evaluate_profile(params, problem):
    """
    Evaluate the reflectance of the discrete stack derived from the continuous profile.
    """
    materials, thicknesses = _continuous_to_discrete(params, problem)
    score = _mean_reflectance(materials, thicknesses, problem)
    return score, materials, thicknesses

def design(problem):
    start_time = time.time()
    deadline = start_time + 140
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    # Parameter space for the cubic polynomial n(z) = a0 + a1*z + a2*z^2 + a3*z^3
    # We want n(0) close to n_amb (1.0) ideally, but min is n_L (1.38).
    # We want n(1) close to n_sub (1.52) ideally, but max is n_H (2.35).
    # Actually, for AR coating, we often want a graded index from n_amb to n_sub.
    # Since n_amb < n_L and n_sub < n_H, the profile will be clipped at the boundaries.
    # Let's parameterize the profile to encourage a monotonic increase from ~n_L to ~n_H.
    # Or just optimize the 4 coefficients directly.
    
    # Bounds for coefficients:
    # n(z) must be in [n_L, n_H] for z in [0,1] (after clipping).
    # Let's search in a reasonable box.
    # a0: [n_L, n_H]
    # a1: [-2, 2]
    # a2: [-2, 2]
    # a3: [-2, 2]
    
    # Differential Evolution-like search
    # Population size
    pop_size = 50
    # Number of generations
    max_gens = 200
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        a0 = random.uniform(n_L, n_H)
        a1 = random.uniform(-2.0, 2.0)
        a2 = random.uniform(-2.0, 2.0)
        a3 = random.uniform(-2.0, 2.0)
        population.append([a0, a1, a2, a3])
    
    best_params = population[0]
    best_score, best_mats, best_ths = _evaluate_profile(best_params, problem)
    
    for gen in range(max_gens):
        if time.time() > deadline - 30:
            break
            
        new_population = [best_params[:]] # Elitism
        
        for i in range(pop_size - 1):
            # Select 3 distinct random individuals
            indices = random.sample(range(pop_size), 3)
            r1, r2, r3 = population[indices[0]], population[indices[1]], population[indices[2]]
            
            # Differential mutation
            F = 0.8
            C = 0.9
            mutant = []
            for j in range(4):
                if random.random() < C:
                    mutant.append(r1[j] + F * (r2[j] - r3[j]))
                else:
                    mutant.append(r1[j])
            
            # Clip to reasonable bounds to prevent explosion
            mutant[0] = np.clip(mutant[0], n_L - 1, n_H + 1)
            for j in range(1, 4):
                mutant[j] = np.clip(mutant[j], -5, 5)
            
            new_population.append(mutant)
        
        # Evaluate new population
        population = new_population
        for params in population:
            score, mats, ths = _evaluate_profile(params, problem)
            if score < best_score:
                best_score = score
                best_params = params[:]
                best_mats = mats
                best_ths = ths
        
        # Check convergence
        if gen % 10 == 0 and gen > 0:
             # Optional: log or check time
             pass

    # Final Polish: Nelder-Mead on the best discrete stack
    # We fix the materials and optimize thicknesses
    
    def nelder_mead(materials_pattern, init_ths, deadline, max_iters=2000):
        ths = list(init_ths)
        ths = [max(0.0, min(problem["max_layer_nm"], t)) for t in ths]
        total = sum(ths)
        max_total = problem["max_total_nm"]
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        
        best_score = _mean_reflectance(materials_pattern, ths, problem)
        best_ths = list(ths)
        n = len(ths)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        step = 5.0
        simplex = [list(ths)]
        for i in range(n):
            pt = list(ths)
            pt[i] = max(0.0, min(problem["max_layer_nm"], pt[i] + step))
            simplex.append(pt)
            
        scores = [_mean_reflectance(materials_pattern, simplex[i], problem) for i in range(n + 1)]
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            if scores[0] < best_score:
                best_score = scores[0]
                best_ths = list(simplex[0])
                
            if max(scores) - min(scores) < 1e-8:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = [max(0.0, min(problem["max_layer_nm"], x)) for x in xr]
            total = sum(xr)
            if total > max_total:
                scale = max_total / total
                xr = [x * scale for x in xr]
            fr = _mean_reflectance(materials_pattern, xr, problem)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = [max(0.0, min(problem["max_layer_nm"], x)) for x in xe]
                total = sum(xe)
                if total > max_total:
                    scale = max_total / total
                    xe = [x * scale for x in xe]
                fe = _mean_reflectance(materials_pattern, xe, problem)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                simplex[-1] = xr
                scores[-1] = fr
            else:
                if fr < scores[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = [max(0.0, min(problem["max_layer_nm"], x)) for x in xc]
                    total = sum(xc)
                    if total > max_total:
                        scale = max_total / total
                        xc = [x * scale for x in xc]
                    fc = _mean_reflectance(materials_pattern, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = [max(0.0, min(problem["max_layer_nm"], x)) for x in simplex[i]]
                        total = sum(simplex[i])
                        if total > max_total:
                            scale = max_total / total
                            simplex[i] = [x * scale for x in simplex[i]]
                        scores[i] = _mean_reflectance(materials_pattern, simplex[i], problem)
                        
        return best_ths, best_score

    final_ths, final_score = nelder_mead(best_mats, best_ths, deadline, max_iters=2000)
    
    if final_score < best_score:
        best_ths = final_ths
        best_score = final_score
        
    return list(best_mats), list(best_ths)
