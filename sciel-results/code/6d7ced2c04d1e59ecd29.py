"""
Broadband antireflection coating design: Physics-Informed Initialization with Multi-Start Nelder-Mead.

Strategy:
1.  Acknowledge that for broadband AR coatings with fixed index contrast (1.38/2.35 on 1.52),
    the optimal structure is almost always an alternating sequence of High and Low index layers.
2.  Instead of randomly searching for the sequence (which wastes time and can get stuck in
    local minima of the discrete space), we fix the sequence to the most likely optimal candidates:
    -   Alternating H-L-H-L... starting with H (High index first, close to air? No, usually Low first for AR? 
        Actually, for AR on glass (n=1.52), we want to transition from 1.0 to 1.52. 
        n_L=1.38 is closer to 1.0. n_H=2.35 is closer to 1.52? No, 2.35 is further.
        Standard AR is n=sqrt(n0*ns). sqrt(1.52)=1.23. We have 1.38.
        Graded index usually goes 1.0 -> 1.38 -> ... -> 1.52.
        However, with only 1.38 and 2.35, we can create an effective graded index by alternating layers.
        The sequence H-L-H-L... or L-H-L-H... are the two primary candidates.
        Let's test both.
    -   We will also consider the number of layers. The problem allows up to 20.
        More layers generally allow better broadband performance.
        We will fix N=20 layers for the optimization.
3.  Initialize thicknesses using the quarter-wave condition at the center wavelength (600 nm).
    d_j = lambda_c / (4 * n_j).
4.  Run Nelder-Mead on the thicknesses for each candidate sequence (L-start and H-start).
5.  To increase robustness, we will perform a "multi-start" Nelder-Mead:
    -   Start 1: Quarter-wave at 600nm.
    -   Start 2: Quarter-wave at 500nm.
    -   Start 3: Quarter-wave at 700nm.
    -   Start 4: Random perturbations of the best quarter-wave solution.
6.  Return the best result found.
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

def _constrain_thicknesses(ths, max_total, max_layer):
    """Constrain thicknesses to be within bounds and total sum."""
    ths = np.asarray(ths, dtype=float)
    ths = np.clip(ths, 0.0, max_layer)
    total = np.sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = ths * scale
        ths = np.clip(ths, 0.0, max_layer)
    return ths

def _nelder_mead(f, x0, bounds, max_iter=200, timeout=None):
    """
    Simple Nelder-Mead implementation.
    f: objective function
    x0: initial point
    bounds: tuple (min_val, max_val) for each dimension
    """
    n = len(x0)
    # Initial simplex
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        # Perturb in direction i
        # Scale perturbation based on bound size
        scale = (bounds[i][1] - bounds[i][0]) * 0.1 + 1.0
        pt[i] += scale
        pt[i] = np.clip(pt[i], bounds[i][0], bounds[i][1])
        simplex.append(pt)
    
    # Evaluate initial simplex
    scores = [f(pt) for pt in simplex]
    
    best_score = min(scores)
    best_pt = simplex[scores.index(best_score)]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    it = 0
    while it < max_iter:
        if timeout and time.time() > timeout:
            break
            
        # Sort simplex
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_pt = list(simplex[0])
            
        # Check convergence
        if max(scores) - min(scores) < 1e-8:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - np.array(simplex[-1]))
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = f(xr)
        
        if fr < scores[0]:
            # Expansion
            xe = centroid + gamma * (centroid - np.array(simplex[-1]))
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                scores[-1] = fe
            else:
                simplex[-1] = xr
                scores[-1] = fr
        elif fr < scores[-2]:
            # Accept reflection
            simplex[-1] = xr
            scores[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (np.array(simplex[-1]) - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = f(xc)
            if fc <= fr:
                simplex[-1] = xc
                scores[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    scores[i] = f(simplex[i])
        
        it += 1
        
    return best_pt, best_score

def design(problem):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    # We will optimize for 20 layers, as it provides the most flexibility for broadband performance.
    N = max_layers
    
    # Candidate sequences:
    # 1. Start with L (Low index) -> L H L H ...
    # 2. Start with H (High index) -> H L H L ...
    
    seq_L_start = ['L' if i % 2 == 0 else 'H' for i in range(N)]
    seq_H_start = ['H' if i % 2 == 0 else 'L' for i in range(N)]
    
    # Bounds for thicknesses
    # Each layer must be in [0, max_layer]
    # Sum must be <= max_total
    # Nelder-Mead doesn't handle sum constraints natively, so we use a penalty or projection.
    # The _constrain_thicknesses function projects to the feasible region.
    # We will wrap the objective function to apply this projection.
    
    def make_objective(materials):
        def obj(ths):
            ths_constrained = _constrain_thicknesses(ths, max_total, max_layer)
            # Check if projection changed the point significantly (indicating we hit a boundary)
            # For now, just evaluate the constrained point.
            return _mean_reflectance(materials, ths_constrained, problem)
        return obj
    
    obj_L = make_objective(seq_L_start)
    obj_H = make_objective(seq_H_start)
    
    best_score = float('inf')
    best_mats = seq_L_start
    best_ths = np.zeros(N)
    
    # Initial guesses based on quarter-wave at different wavelengths
    # d = lambda / (4 * n)
    # We need to generate initial thickness vectors for each sequence.
    
    def get_init_ths(seq, lambda_c):
        ths = []
        for m in seq:
            n = n_L if m == 'L' else n_H
            d = lambda_c / (4.0 * n)
            ths.append(d)
        return np.array(ths)
    
    # Multi-start Nelder-Mead
    # Start points:
    # 1. lambda_c = 600 nm (center)
    # 2. lambda_c = 500 nm
    # 3. lambda_c = 700 nm
    # 4. Random perturbations of the center start
    
    start_lambdas = [600.0, 500.0, 700.0, 450.0, 750.0]
    
    for seq, obj in [(seq_L_start, obj_L), (seq_H_start, obj_H)]:
        for lam in start_lambdas:
            if time.time() > deadline - 5:
                break
                
            x0 = get_init_ths(seq, lam)
            
            # Add some random noise to avoid getting stuck in exactly the same local min
            # But keep it small to stay near the physics-based guess
            noise = np.random.normal(0, 10.0, N)
            x0_noisy = x0 + noise
            
            # We will run Nelder-Mead on both the clean and noisy start
            for x_init in [x0, x0_noisy]:
                if time.time() > deadline - 5:
                    break
                
                # Bounds for Nelder-Mead
                # Individual bounds: [0, max_layer]
                # We ignore the sum constraint in bounds and rely on projection in the objective.
                # This is safer for NM convergence.
                bounds = [(0.0, max_layer) for _ in range(N)]
                
                # Run NM
                # Limit iterations to ensure we finish in time
                # 20 layers -> 21 evals per iter. 200 iters -> 4200 evals.
                # Each eval is ~401 wavelengths * 20 layers matrix ops. Very fast.
                # We can afford more iterations if time permits.
                max_iter = 300
                
                res_ths, res_score = _nelder_mead(obj, x_init, bounds, max_iter=max_iter, timeout=deadline - 5)
                
                if res_score < best_score:
                    best_score = res_score
                    best_mats = seq
                    best_ths = res_ths
                    
    # If we still have time, do a final polish on the best result found
    if time.time() < deadline - 2:
        # Determine which objective to use for final polish
        if best_mats == seq_L_start:
            final_obj = obj_L
        else:
            final_obj = obj_H
            
        bounds = [(0.0, max_layer) for _ in range(N)]
        # Polish with more iterations
        final_ths, final_score = _nelder_mead(final_obj, best_ths, bounds, max_iter=500, timeout=deadline - 1)
        
        if final_score < best_score:
            best_ths = final_ths
            best_score = final_score
            
    # Ensure return types are correct
    return list(best_mats), list(best_ths)
