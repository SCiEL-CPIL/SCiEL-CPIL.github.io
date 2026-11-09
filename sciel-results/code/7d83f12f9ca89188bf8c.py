"""Broadband antireflection coating design: Parametric Graded Index Profile."""
import time
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
    
    # delta_j = 2*pi*n_j*d_j/lambda  -> (N, K)
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
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    band = problem["band_nm"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    N = max_layers  # Use maximum allowed layers for best approximation
    
    # 1. Define a parametric continuous index profile n(z)
    # We model n(z) as a cubic polynomial: n(z) = c0 + c1*z + c2*z^2 + c3*z^3
    # z is normalized depth [0, 1]
    # Constraints:
    # n(0) = n_ambient (match air)
    # n(1) = n_substrate (match glass)
    # n'(0) = 0 (smooth start, reduces reflection at air interface)
    # n'(1) = 0 (smooth end, reduces reflection at glass interface)
    
    # Let's solve for coefficients based on boundary conditions
    # n(0) = c0 = n_ambient
    # n'(0) = c1 = 0
    # n(1) = c0 + c2 + c3 = n_substrate
    # n'(1) = 3*c2 + 3*c3 = 0 => c2 = -c3
    # Substitute c2 = -c3 into n(1):
    # c0 + c3 - c3 = c0 = n_ambient. Wait.
    # n(1) = c0 + c2 + c3. If c2=-c3, then n(1)=c0. This implies n_substrate must equal n_ambient.
    # My boundary conditions for a simple cubic are too restrictive or I need a higher order polynomial
    # or different conditions.
    
    # Let's use a 4th order polynomial or just optimize the coefficients directly with bounds.
    # Better: Use a polynomial that naturally fits the boundaries.
    # n(z) = n_ambient + (n_substrate - n_ambient) * P(z)
    # where P(0)=0, P(1)=1.
    # Let's use P(z) = z^k. Simple power law.
    # Or better, let's optimize the coefficients of a cubic polynomial n(z) = c0 + c1*z + c2*z^2 + c3*z^3
    # subject to n(0)=n_ambient, n(1)=n_substrate.
    # We have 2 degrees of freedom (c1, c2, c3 with one constraint? No, 4 coeffs, 2 constraints -> 2 free params).
    # Let's parameterize:
    # n(z) = n_ambient + (n_substrate - n_ambient) * (3*z^2 - 2*z^3)  <-- Smoothstep
    # This satisfies n(0)=n_amb, n(1)=n_sub, n'(0)=0, n'(1)=0.
    # This is a very good starting point for a graded index.
    
    # We will optimize the "shape" of the transition.
    # Let's use a polynomial: n(z) = n_amb + (n_sub - n_amb) * (a*z + b*z^2 + c*z^3)
    # Constraints:
    # n(1) = n_sub => a + b + c = 1
    # n'(0) = 0 => a = 0 ? No, n'(z) = (n_sub-n_amb)*(a + 2b z + 3c z^2).
    # If we want smooth entry/exit, n'(0)=0 and n'(1)=0.
    # a=0, a+2b+3c=0 => 2b+3c=0 => b = -1.5c.
    # a+b+c=1 => -0.5c = 1 => c = -2, b = 3, a = 0.
    # So n(z) = n_amb + (n_sub-n_amb)*(3z^2 - 2z^3). This is the standard smoothstep.
    
    # To allow more flexibility, let's optimize a "sharpness" parameter or a deviation.
    # Let's define the target profile as:
    # n_target(z) = n_amb + (n_sub - n_amb) * (z^p)  where p is optimized.
    # p=1 is linear, p=2 is quadratic, etc.
    # Actually, let's just optimize the coefficients of a cubic polynomial directly,
    # but enforce n(0)=n_amb and n(1)=n_sub.
    # n(z) = n_amb + z * (n_sub - n_amb) + z(1-z) * (u*z + v*(1-z)) ?
    # Let's stick to the simplest robust method:
    # Define target index at the center of each layer.
    # z_i = (i + 0.5) / N for i in 0..N-1.
    # We want to find a function n(z).
    # Let's use n(z) = n_amb + (n_sub - n_amb) * (z^alpha)
    # Optimize alpha.
    
    # However, the "graded index" concept in thin films often works best when the effective index
    # of the stack follows a specific curve.
    # With only 2 materials L and H, we can't make an arbitrary index.
    # We can only make a binary sequence.
    # The "effective index" of a thick layer is just its index.
    # But if we have a stack, the reflection is determined by the sequence.
    # The "graded index" strategy here means:
    # 1. Define a target index profile n_target(z) from n_amb to n_sub.
    # 2. For each layer i, determine if n_target(z_i) is closer to n_L or n_H.
    # 3. Assign material accordingly.
    # 4. Optimize thicknesses to minimize reflectance.
    
    # This is a hybrid: Global structure determined by profile, local thicknesses by NM.
    
    # Let's define the target profile function.
    # We will optimize the parameters of the profile to minimize the resulting reflectance.
    # Profile: n_target(z) = n_amb + (n_sub - n_amb) * f(z; params)
    # Let's use f(z) = z^k. k is the parameter.
    # k=1: Linear.
    # k=2: Quadratic (slow start, fast end).
    # k=0.5: Square root (fast start, slow end).
    
    # We will scan k in [0.2, 5.0] and for each k, generate the pattern,
    # then run a quick Nelder-Mead on thicknesses.
    
    # To be faster, we can also try a sinusoidal profile or a polynomial.
    # Let's try a few profile shapes.
    
    def get_pattern_and_init(k, profile_type='power'):
        z_centers = (np.arange(N) + 0.5) / N
        if profile_type == 'power':
            # n_target = n_amb + (n_sub - n_amb) * z^k
            n_targets = n_amb + (n_sub - n_amb) * (z_centers ** k)
        elif profile_type == 'smoothstep':
            # n_target = n_amb + (n_sub - n_amb) * (3z^2 - 2z^3)
            n_targets = n_amb + (n_sub - n_amb) * (3 * z_centers**2 - 2 * z_centers**3)
        elif profile_type == 'log':
            # Geometric mean profile
            log_n_amb = np.log(n_amb)
            log_n_sub = np.log(n_sub)
            log_targets = log_n_amb + (log_n_sub - log_n_amb) * (z_centers ** k)
            n_targets = np.exp(log_targets)
        else:
            n_targets = n_amb + (n_sub - n_amb) * z_centers
            
        # Assign materials
        materials = []
        # Quarter wave thicknesses at center wavelength for initialization
        wl_center = 0.5 * (band[0] + band[1])
        qw_L = wl_center / (4.0 * n_L)
        qw_H = wl_center / (4.0 * n_H)
        
        init_ths = []
        
        for n_t in n_targets:
            # Distance to L and H
            d_L = abs(n_t - n_L)
            d_H = abs(n_t - n_H)
            
            if d_L < d_H:
                materials.append('L')
                init_ths.append(qw_L)
            else:
                materials.append('H')
                init_ths.append(qw_H)
                
        # Constrain initial thicknesses
        total = sum(init_ths)
        if total > max_total:
            scale = max_total / total
            init_ths = [t * scale for t in init_ths]
        for i in range(len(init_ths)):
            init_ths[i] = max(0.0, min(max_layer, init_ths[i]))
            
        return materials, init_ths

    def constrain_thicknesses(ths):
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        for i in range(len(ths)):
            ths[i] = max(0.0, min(max_layer, ths[i]))
        total2 = sum(ths)
        if total2 > max_total:
            scale = max_total / total2
            ths = [t * scale for t in ths]
        return ths

    def nelder_mead_fast(materials, init_ths, max_iters=50):
        """Fast Nelder-Mead for thicknesses."""
        ths = list(init_ths)
        ths = constrain_thicknesses(ths)
        best_score = _mean_reflectance(materials, ths, problem)
        best_ths = list(ths)
        n = len(ths)
        
        if n == 0:
            return best_ths, best_score

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial step size
        step = 5.0
        simplex = [list(ths)]
        for i in range(n):
            pt = list(ths)
            pt[i] = max(0.0, min(max_layer, pt[i] + step))
            simplex.append(pt)
        
        scores = [_mean_reflectance(materials, simplex[i], problem) for i in range(n + 1)]
        
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
            xr = constrain_thicknesses(list(xr))
            fr = _mean_reflectance(materials, xr, problem)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = constrain_thicknesses(list(xe))
                fe = _mean_reflectance(materials, xe, problem)
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
                    xc = constrain_thicknesses(list(xc))
                    fc = _mean_reflectance(materials, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = constrain_thicknesses(simplex[i])
                        scores[i] = _mean_reflectance(materials, simplex[i], problem)
                    if np.max(scores) - np.min(scores) < 1e-7:
                        break
                        
        return best_ths, best_score

    best_ths = None
    best_score = float('inf')
    best_materials = None
    
    # Strategy: Scan a few profile shapes and optimize thicknesses for each.
    # 1. Power law with k=0.5, 1.0, 2.0
    # 2. Smoothstep
    # 3. Logarithmic with k=1.0
    
    profiles = [
        ('power', 0.5),
        ('power', 1.0),
        ('power', 2.0),
        ('smoothstep', None),
        ('log', 1.0),
        ('log', 0.5),
        ('log', 2.0)
    ]
    
    for p_type, p_val in profiles:
        if time.time() > deadline - 20:
            break
            
        mats, init_ths = get_pattern_and_init(p_val if p_val else 1.0, p_type)
        
        # Quick polish
        polished_ths, polished_score = nelder_mead_fast(mats, init_ths, max_iters=50)
        
        if polished_score < best_score:
            best_score = polished_score
            best_ths = polished_ths
            best_materials = mats

    # If we have time, do a more thorough search on the best profile found
    if best_materials is not None and time.time() < deadline - 30:
        # Try random perturbations of the best thicknesses
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = []
            for t in best_ths:
                factor = random.uniform(0.8, 1.2)
                perturbed.append(t * factor)
            perturbed = constrain_thicknesses(perturbed)
            
            polished_ths, polished_score = nelder_mead_fast(best_materials, perturbed, max_iters=50)
            if polished_score < best_score:
                best_score = polished_score
                best_ths = polished_ths

    # Final polish on the absolute best
    if best_ths and time.time() < deadline - 5:
        final_ths, final_score = nelder_mead_fast(best_materials, best_ths, max_iters=100)
        if final_score < best_score:
            best_ths = final_ths
            best_score = final_score

    if best_ths is None:
        # Fallback
        return ["L"], [band[0] / (4.0 * n_L)]
        
    return list(best_materials), list(best_ths)
