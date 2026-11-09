"""
Broadband antireflection coating design: Fixed symmetric graded structure with Stochastic Global Search.
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


def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    # Strategy: Fixed symmetric pattern of length 20 (or max_layers)
    # Pattern: L H H L L H H L ... or similar graded structure?
    # Actually, a common high-performance structure is a "graded" index stack.
    # Let's fix a specific symmetric pattern that approximates a linear index gradient.
    # n goes from ~1.38 to ~2.35.
    # We have 20 layers. 10 unique layers mirrored.
    # Let's define the first 10 layers to go from Low to High.
    # Example: L L H H L L H H L H ?
    # Better: Just use a fixed pattern that alternates or steps.
    # A very effective known structure is the "Woodpile" or similar, but with binary materials,
    # a symmetric stack with increasing "H" content towards the center (if L is air-side)
    # or decreasing?
    # Standard AR on glass: Air -> L -> H -> L -> H ... -> Glass?
    # Actually, for broadband, a linear gradient from n_air to n_glass is ideal.
    # Since we only have L (1.38) and H (2.35), we can approximate a gradient.
    # Glass is 1.52.
    # Ideally: Air(1.0) -> L(1.38) -> ... -> Glass(1.52).
    # Wait, H is 2.35, which is much higher than Glass.
    # So a simple L-H-L-H... might be good for narrow band, but for broadband,
    # we want to avoid large index jumps if possible, or use multiple layers to smooth it.
    
    # Let's fix the pattern to be symmetric and use 20 layers.
    # Pattern: L H L H ... L H (10 pairs) -> Mirror?
    # If we mirror, the center is ... L H H L ...
    # Let's try a pattern that starts with L (closest to air) and ends with L (closest to glass? No, glass is 1.52, L is 1.38. H is 2.35).
    # Actually, putting H next to glass is bad for AR if the band is wide?
    # Let's look at the "graded" approach.
    # We will fix the *sequence* of materials to a specific symmetric 20-layer pattern
    # and only optimize the *thicknesses*.
    
    # Let's choose a pattern that is known to be robust:
    # Symmetric 20-layer stack.
    # First 10 layers: L H L H L H L H L H
    # Last 10 layers: H L H L H L H L H L
    # This creates a symmetric stack.
    
    # Alternative: L L H H L L H H L L H H ...
    # Let's stick to the alternating pattern as it provides good impedance matching steps.
    
    half_len = max_layers // 2
    # Generate first half: L H L H ...
    first_half = []
    for i in range(half_len):
        if i % 2 == 0:
            first_half.append('L')
        else:
            first_half.append('H')
    
    # Mirror for second half (reverse order)
    second_half = first_half[::-1]
    
    fixed_pattern = tuple(first_half + second_half)
    
    # Initial thicknesses: Quarter-wave at center wavelength
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    def get_qw(mat):
        return qw_L if mat == 'L' else qw_H
    
    init_ths = [get_qw(m) for m in fixed_pattern]
    
    # Constraint handling
    def constrain(ths):
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        for i in range(len(ths)):
            ths[i] = max(0.0, min(max_layer, ths[i]))
        # Re-check total after clamping (rarely needed if scale < 1, but safe)
        total2 = sum(ths)
        if total2 > max_total:
            scale = max_total / total2
            ths = [t * scale for t in ths]
        return ths

    current_ths = constrain(init_ths)
    current_score = _mean_reflectance(fixed_pattern, current_ths, problem)
    
    best_ths = list(current_ths)
    best_score = current_score
    
    # Stochastic Search (Simulated Annealing style)
    # We will perturb the thicknesses randomly and accept if better or with probability exp(-dE/T)
    
    # Parameters
    T_start = 5.0  # Initial temperature (in percent reflectance units)
    T_end = 0.01
    cooling_rate = 0.995
    
    # Number of iterations per temperature step
    iters_per_temp = 5
    
    T = T_start
    iterations = 0
    
    # Estimate max iterations based on time
    # Each eval is ~1ms? 401 points, 20 layers. Vectorized numpy is fast.
    # Let's assume 1000 evals/sec. 140s -> 140,000 evals.
    # We can run many iterations.
    
    while time.time() < deadline and T > T_end:
        for _ in range(iters_per_temp):
            if time.time() >= deadline:
                break
                
            # Generate perturbation
            # Randomly select a few layers to change
            n_perturb = random.randint(1, 3)
            indices = random.sample(range(len(current_ths)), n_perturb)
            
            new_ths = list(current_ths)
            for idx in indices:
                # Perturb by a random amount
                # Scale of perturbation: start large, shrink later?
                # Or just uniform noise.
                noise = random.gauss(0, 20.0 * (T / T_start + 0.1))
                new_ths[idx] += noise
                
            new_ths = constrain(new_ths)
            
            new_score = _mean_reflectance(fixed_pattern, new_ths, problem)
            
            delta_e = new_score - current_score
            
            # Acceptance criterion
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                current_ths = new_ths
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_ths = list(current_ths)
            else:
                # Rejected, keep current
                pass
                
            iterations += 1
            
        T *= cooling_rate
        
        # Optional: If no improvement for a long time, reset to best and restart?
        # Or just continue. SA should handle this.
        
    # Final Polish: Nelder-Mead on the best found so far
    # We use NM to refine the global search result.
    
    def nelder_mead(materials_pattern, init_ths, problem, deadline, max_iters=200):
        ths = list(init_ths)
        ths = constrain(ths)
        best_score = _mean_reflectance(materials_pattern, ths, problem)
        best_ths = list(ths)
        n = len(ths)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Smaller initial step for polish
        step = 5.0
        simplex = [list(ths)]
        for i in range(n):
            pt = list(ths)
            pt[i] = max(0.0, min(max_layer, pt[i] + step))
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
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = constrain(list(xr))
            fr = _mean_reflectance(materials_pattern, xr, problem)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = constrain(list(xe))
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
                    xc = constrain(list(xc))
                    fc = _mean_reflectance(materials_pattern, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = constrain(simplex[i])
                        scores[i] = _mean_reflectance(materials_pattern, simplex[i], problem)
                    if np.max(scores) - np.min(scores) < 1e-7:
                        break
                        
        return materials_pattern, best_ths, best_score

    # Run NM polish
    _, final_ths, final_score = nelder_mead(fixed_pattern, best_ths, problem, deadline, max_iters=300)
    
    # Final check
    if final_score < best_score:
        best_ths = final_ths
        best_score = final_score
        
    return list(fixed_pattern), list(best_ths)
