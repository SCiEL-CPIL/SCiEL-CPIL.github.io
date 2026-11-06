"""Design broadband antireflection coating via systematic local search."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband AR coating using local search from good initial stacks."""
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    n0 = problem["ambient_index"]       # 1.0
    ns = problem["substrate_index"]     # 1.52
    nL = problem["materials"]["L"]      # 1.38
    nH = problem["materials"]["H"]      # 2.35
    band = problem["band_nm"]           # [400, 800]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Wavelengths for evaluation (401 equally spaced points)
    wl = np.linspace(band[0], band[1], 401)
    
    def characteristic_matrix(n, d, lam):
        """Compute the 2x2 characteristic matrix for a single layer."""
        k = 2.0 * np.pi * n / lam
        phi = k * d
        c = np.cos(phi)
        s = np.sin(phi)
        return np.array([[c, 1j * s / n], [1j * n * s, c]])
    
    def reflectance_stack(materials, thicknesses, wavelengths):
        """Compute mean reflectance (%) using characteristic matrix method."""
        if len(materials) == 0:
            # Bare substrate
            r = (n0 - ns) / (n0 + ns)
            return 100.0 * np.mean(np.abs(r)**2)
        
        n_mat = np.array([nL if m == 'L' else nH for m in materials])
        d_mat = np.array(thicknesses)
        
        total_R = 0.0
        for w in wavelengths:
            # Start from substrate
            M = np.array([[1.0, 0.0], [0.0, 1.0]])
            for i in range(len(materials)):
                Mi = characteristic_matrix(n_mat[i], d_mat[i], w)
                M = M @ Mi
            # Admittance at substrate side
            Ys = ns  # real admittance of substrate
            # Total admittance seen from air side
            Y_total = (M[0, 0] * Ys + M[0, 1]) / (M[1, 0] * Ys + M[1, 1])
            # Reflectance
            r = (n0 - Y_total) / (n0 + Y_total)
            total_R += np.abs(r)**2
        return 100.0 * total_R / len(wavelengths)
    
    def objective(params):
        """Objective function: mean reflectance, with penalty for constraint violations."""
        # params: interleaved [mat0, th0, mat1, th1, ...] where mat is 0=L, 1=H
        n_layers = len(params) // 2
        if n_layers == 0:
            return 4.26  # bare glass baseline
        
        materials = []
        thicknesses = []
        for i in range(n_layers):
            mat_val = params[2*i]
            th_val = params[2*i+1]
            # Clip thickness to valid range
            th = max(0.0, min(th_val, max_layer_nm))
            materials.append('L' if mat_val < 0.5 else 'H')
            thicknesses.append(th)
        
        # Check total thickness constraint
        total_th = sum(thicknesses)
        if total_th > max_total_nm:
            return 100.0 + (total_th - max_total_nm)  # heavy penalty
        
        return reflectance_stack(materials, thicknesses, wl)
    
    def encode(materials, thicknesses):
        """Encode materials and thicknesses into parameter vector."""
        params = []
        for m, t in zip(materials, thicknesses):
            params.append(0.0 if m == 'L' else 1.0)
            params.append(t)
        return np.array(params)
    
    def decode(params):
        """Decode parameter vector into materials and thicknesses."""
        n_layers = len(params) // 2
        materials = []
        thicknesses = []
        for i in range(n_layers):
            materials.append('L' if params[2*i] < 0.5 else 'H')
            thicknesses.append(max(0.0, min(params[2*i+1], max_layer_nm)))
        return materials, thicknesses
    
    def try_local_search(materials, thicknesses, time_budget):
        """Perform local search: perturb thicknesses, swap materials, add/remove layers."""
        best_mat = list(materials)
        best_th = list(thicknesses)
        best_score = reflectance_stack(best_mat, best_th, wl)
        
        t_start = time.time()
        no_improve_count = 0
        step_sizes = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0]
        
        for ss in step_sizes:
            while time.time() - t_start < time_budget:
                improved = False
                
                # Try perturbing each thickness
                for i in range(len(best_th)):
                    for delta in [ss, -ss]:
                        new_th = list(best_th)
                        new_th[i] = max(0.0, min(best_th[i] + delta, max_layer_nm))
                        if sum(new_th) <= max_total_nm:
                            score = reflectance_stack(best_mat, new_th, wl)
                            if score < best_score - 1e-10:
                                best_th = new_th
                                best_score = score
                                improved = True
                                no_improve_count = 0
                                break
                    if improved:
                        break
                
                # Try swapping material of each layer
                if not improved:
                    for i in range(len(best_mat)):
                        new_mat = list(best_mat)
                        new_mat[i] = 'H' if best_mat[i] == 'L' else 'L'
                        if sum(best_th) <= max_total_nm:
                            score = reflectance_stack(new_mat, best_th, wl)
                            if score < best_score - 1e-10:
                                best_mat = new_mat
                                best_score = score
                                improved = True
                                no_improve_count = 0
                                break
                
                # Try adding a layer
                if not improved and len(best_mat) < max_layers:
                    for pos in range(len(best_mat) + 1):
                        for new_mat in ['L', 'H']:
                            new_mat_list = list(best_mat)
                            new_th_list = list(best_th)
                            new_mat_list.insert(pos, new_mat)
                            new_th_list.insert(pos, 50.0)
                            if sum(new_th_list) <= max_total_nm:
                                score = reflectance_stack(new_mat_list, new_th_list, wl)
                                if score < best_score - 1e-10:
                                    best_mat = new_mat_list
                                    best_th = new_th_list
                                    best_score = score
                                    improved = True
                                    no_improve_count = 0
                                    break
                        if improved:
                            break
                
                # Try removing a layer
                if not improved and len(best_mat) > 1:
                    for pos in range(len(best_mat)):
                        new_mat_list = list(best_mat)
                        new_th_list = list(best_th)
                        new_mat_list.pop(pos)
                        new_th_list.pop(pos)
                        if sum(new_th_list) <= max_total_nm:
                            score = reflectance_stack(new_mat_list, new_th_list, wl)
                            if score < best_score - 1e-10:
                                best_mat = new_mat_list
                                best_th = new_th_list
                                best_score = score
                                improved = True
                                no_improve_count = 0
                                break
                
                if not improved:
                    no_improve_count += 1
                    if no_improve_count > 2:
                        break
        
        return best_mat, best_th, best_score
    
    # Generate initial candidate stacks
    candidates = []
    
    # Quarter-wave stack designs
    centre = 0.5 * (band[0] + band[1])
    
    # Single layer
    candidates.append((["L"], [centre / (4.0 * nL)]))
    candidates.append((["H"], [centre / (4.0 * nH)]))
    
    # Two-layer quarter-wave stack
    dL = centre / (4.0 * nL)
    dH = centre / (4.0 * nH)
    candidates.append((["L", "H"], [dL, dH]))
    candidates.append((["H", "L"], [dH, dL]))
    
    # Three-layer quarter-wave stack
    candidates.append((["L", "H", "L"], [dL, dH, dL]))
    candidates.append((["H", "L", "H"], [dH, dL, dH]))
    
    # Four-layer quarter-wave stack
    candidates.append((["L", "H", "L", "H"], [dL, dH, dL, dH]))
    candidates.append((["H", "L", "H", "L"], [dH, dL, dH, dL]))
    
    # Five-layer quarter-wave stack
    candidates.append((["L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL]))
    candidates.append((["H", "L", "H", "L", "H"], [dH, dL, dH, dL, dH]))
    
    # Six-layer quarter-wave stack
    candidates.append((["L", "H", "L", "H", "L", "H"], [dL, dH, dL, dH, dL, dH]))
    candidates.append((["H", "L", "H", "L", "H", "L"], [dH, dL, dH, dL, dH, dL]))
    
    # Seven-layer
    candidates.append((["L", "H", "L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL, dH, dL]))
    candidates.append((["H", "L", "H", "L", "H", "L", "H"], [dH, dL, dH, dL, dH, dL, dH]))
    
    # Some non-quarter-wave initial designs
    # Half-wave + quarter-wave combinations
    halfL = centre / (2.0 * nL)
    halfH = centre / (2.0 * nH)
    candidates.append((["L", "H"], [halfL, dH]))
    candidates.append((["H", "L"], [halfH, dL]))
    
    # Evaluate all candidates and pick best few for local search
    scored_candidates = []
    for mat, th in candidates:
        if sum(th) > max_total_nm:
            continue
        if len(mat) > max_layers:
            continue
        score = reflectance_stack(mat, th, wl)
        scored_candidates.append((score, mat, th))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    best_mat, best_th, best_score = None, None, float('inf')
    
    # Run local search on top candidates
    time_per_candidate = 15.0  # seconds per candidate
    for i, (score, mat, th) in enumerate(scored_candidates[:8]):
        if time.time() - start_time > deadline - 10:
            break
        
        mat_ls, th_ls, score_ls = try_local_search(mat, th, time_per_candidate)
        
        if score_ls < best_score:
            best_score = score_ls
            best_mat = mat_ls
            best_th = th_ls
        
        # Early termination if we've found a very good solution
        if best_score < 1.0:
            break
    
    # Final high-precision polish using scipy minimize on the best solution
    if best_mat is not None and time.time() - start_time < deadline - 20:
        params = encode(best_mat, best_th)
        
        # Use Nelder-Mead on thicknesses only (materials are discrete)
        def thickness_objective(th_params):
            new_th = [max(0.0, min(t, max_layer_nm)) for t in th_params]
            if sum(new_th) > max_total_nm:
                return 100.0 + (sum(new_th) - max_total_nm)
            return reflectance_stack(best_mat, new_th, wl)
        
        try:
            res = minimize(thickness_objective, best_th, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 0.01, 'fatol': 1e-8})
            if res.fun < best_score:
                best_th = [max(0.0, min(t, max_layer_nm)) for t in res.x]
                if sum(best_th) <= max_total_nm:
                    best_score = res.fun
        except Exception:
            pass
        
        # Try swapping individual materials and re-optimizing
        for i in range(len(best_mat)):
            if time.time() - start_time > deadline - 10:
                break
            new_mat = list(best_mat)
            new_mat[i] = 'H' if best_mat[i] == 'L' else 'L'
            if sum(best_th) > max_total_nm:
                continue
            score = reflectance_stack(new_mat, best_th, wl)
            if score < best_score - 0.01:
                # Try optimizing thicknesses for this new material configuration
                def th_obj(th_p):
                    new_th = [max(0.0, min(t, max_layer_nm)) for t in th_p]
                    if sum(new_th) > max_total_nm:
                        return 100.0 + (sum(new_th) - max_total_nm)
                    return reflectance_stack(new_mat, new_th, wl)
                try:
                    res2 = minimize(th_obj, best_th, method='Nelder-Mead',
                                   options={'maxiter': 300, 'xatol': 0.01, 'fatol': 1e-8})
                    if res2.fun < best_score:
                        best_mat = new_mat
                        best_th = [max(0.0, min(t, max_layer_nm)) for t in res2.x]
                        best_score = res2.fun
                except Exception:
                    pass
    
    # Ensure final result is valid
    if best_mat is None:
        best_mat = ["L"]
        best_th = [centre / (4.0 * nL)]
    
    # Final validation
    best_th = [max(0.0, min(t, max_layer_nm)) for t in best_th]
    if sum(best_th) > max_total_nm:
        # Scale down thicknesses proportionally
        scale = max_total_nm / sum(best_th)
        best_th = [t * scale for t in best_th]
    
    return best_mat, best_th
# EVOLVE-BLOCK-END
