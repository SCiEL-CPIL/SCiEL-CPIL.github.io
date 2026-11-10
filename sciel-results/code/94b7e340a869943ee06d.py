import numpy as np
import time
import random
import sys
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design broadband AR coating using:
    1. Continuous profile optimization (L-BFGS) to find a good initial structure.
    2. Simulated Annealing (SA) to refine the discrete structure (materials + thicknesses).
    3. Nelder-Mead for final thickness polishing.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    lambda_c = 0.5 * (band[0] + band[1])
    n_wl = len(lambdas)
    
    def mean_reflectance(materials, thicknesses):
        """Compute mean reflectance in percent using vectorized characteristic matrix method."""
        n_layers = np.array([nL if m == 'L' else nH for m in materials], dtype=float)
        n = len(materials)
        if n == 0:
            r = (n0 - ns) / (n0 + ns)
            return (np.abs(r) ** 2) * 100.0
        
        m_total = np.zeros((n_wl, 2, 2), dtype=complex)
        m_total[:, 0, 0] = 1.0
        m_total[:, 1, 1] = 1.0
        
        for i in range(n):
            ni = n_layers[i]
            di = thicknesses[i]
            delta = 2.0 * np.pi * ni * di / lambdas
            c = np.cos(delta)
            s = np.sin(delta)
            
            m_layer = np.zeros((n_wl, 2, 2), dtype=complex)
            m_layer[:, 0, 0] = c
            m_layer[:, 0, 1] = 1j * s / ni
            m_layer[:, 1, 0] = 1j * ni * s
            m_layer[:, 1, 1] = c
            
            m_total = np.einsum('wij,wjk->wik', m_total, m_layer)
        
        A = m_total[:, 0, 0]
        B = m_total[:, 0, 1]
        C = m_total[:, 1, 0]
        D = m_total[:, 1, 1]
        
        y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - y_in) / (n0 + y_in)
        R = np.abs(r) ** 2
        
        return float(np.mean(R) * 100.0)

    def is_valid(materials, thicknesses):
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        return True

    def clip_thicknesses(thicknesses):
        t = np.array(thicknesses, dtype=float)
        t = np.clip(t, 0.0, max_layer_nm)
        total = np.sum(t)
        if total > max_total_nm:
            scale = max_total_nm / total
            t = t * scale
        return [float(x) for x in t]

    # --- Phase 1: Continuous Profile Optimization ---
    # Optimize continuous indices and thicknesses for a fixed number of layers
    # Then quantize indices to L/H
    
    def optimize_continuous_profile(n_target_layers):
        """
        Optimize continuous indices and thicknesses using L-BFGS.
        Returns quantized materials and thicknesses.
        """
        n0_cont = (nL + nH) / 2.0
        d0_cont = lambda_c / (4.0 * n0_cont)
        
        # Initial guess: alternating L/H
        x0 = []
        for i in range(n_target_layers):
            if i % 2 == 0:
                x0.extend([nL, d0_cont * (n0_cont / nL)])
            else:
                x0.extend([nH, d0_cont * (n0_cont / nH)])
        
        # Bounds: n in [nL, nH], d in [0, max_layer_nm]
        bounds = []
        for _ in range(n_target_layers):
            bounds.append((nL, nH))
            bounds.append((0.0, max_layer_nm))
        
        def objective(x):
            # Extract indices and thicknesses
            n_idx = x[0::2]
            d_idx = x[1::2]
            
            # Compute reflectance for continuous profile
            # We need a version of mean_reflectance that takes continuous indices
            n = len(n_idx)
            if n == 0:
                return 100.0
            
            m_total = np.zeros((n_wl, 2, 2), dtype=complex)
            m_total[:, 0, 0] = 1.0
            m_total[:, 1, 1] = 1.0
            
            for i in range(n):
                ni = n_idx[i]
                di = d_idx[i]
                delta = 2.0 * np.pi * ni * di / lambdas
                c = np.cos(delta)
                s = np.sin(delta)
                
                m_layer = np.zeros((n_wl, 2, 2), dtype=complex)
                m_layer[:, 0, 0] = c
                m_layer[:, 0, 1] = 1j * s / ni
                m_layer[:, 1, 0] = 1j * ni * s
                m_layer[:, 1, 1] = c
                
                m_total = np.einsum('wij,wjk->wik', m_total, m_layer)
            
            A = m_total[:, 0, 0]
            B = m_total[:, 0, 1]
            C = m_total[:, 1, 0]
            D = m_total[:, 1, 1]
            
            y_in = (C + D * ns) / (A + B * ns)
            r = (n0 - y_in) / (n0 + y_in)
            R = np.abs(r) ** 2
            
            return float(np.mean(R) * 100.0)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
            x_opt = res.x
        except Exception:
            x_opt = x0
        
        # Quantize the result
        n_idx = x_opt[0::2]
        d_idx = x_opt[1::2]
        
        materials = []
        thicknesses = []
        for i in range(n_target_layers):
            if n_idx[i] < (nL + nH) / 2:
                materials.append('L')
            else:
                materials.append('H')
            thicknesses.append(d_idx[i])
        
        # Clip thicknesses
        thicknesses = clip_thicknesses(thicknesses)
        
        # Merge adjacent same-material layers to reduce count if needed
        # (Optional, but helps with constraint satisfaction)
        new_mats = []
        new_thicks = []
        for i in range(len(materials)):
            if new_mats and new_mats[-1] == materials[i]:
                new_thicks[-1] += thicknesses[i]
            else:
                new_mats.append(materials[i])
                new_thicks.append(thicknesses[i])
        
        materials = new_mats
        thicknesses = new_thicks
        
        # Ensure valid
        if not is_valid(materials, thicknesses):
            # Fallback to simple quarter-wave
            materials = ['L']
            thicknesses = [lambda_c / (4 * nL)]
            
        return materials, thicknesses

    # Run continuous optimization for a few different layer counts
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for n_layers in [8, 10, 12]:
        if (time.time() - start_time) > DEADLINE:
            break
            
        mat, thick = optimize_continuous_profile(n_layers)
        score = mean_reflectance(mat, thick)
        
        if score < best_score:
            best_score = score
            best_materials = mat
            best_thicknesses = thick

    # Fallback if continuous optimization failed or didn't run
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [lambda_c / (4 * nL)]
        best_score = mean_reflectance(best_materials, best_thicknesses)

    # --- Phase 2: Simulated Annealing on Discrete Space ---
    def simulated_annealing(materials, thicknesses, t_start=10.0, t_end=0.01, iterations=20000):
        """
        Refine the discrete structure using Simulated Annealing.
        Moves: perturb thickness, flip material, add layer, remove layer, swap layers.
        """
        current_mat = list(materials)
        current_thick = list(thicknesses)
        current_f = mean_reflectance(current_mat, current_thick)
        
        best_mat = list(current_mat)
        best_thick = list(current_thick)
        best_f = current_f
        
        n_iter = 0
        while n_iter < iterations and (time.time() - start_time) < DEADLINE:
            # Temperature schedule
            progress = n_iter / iterations
            temp = t_start * (t_end / t_start) ** progress
            
            # Generate neighbor
            new_mat = list(current_mat)
            new_thick = list(current_thick)
            
            move_type = random.randint(0, 4)
            
            if move_type == 0: # Perturb thickness
                if len(new_thick) == 0:
                    continue
                i = random.randint(0, len(new_thick) - 1)
                noise = random.gauss(0, 10)
                new_thick[i] += noise
                new_thick = clip_thicknesses(new_thick)
                
            elif move_type == 1: # Flip material
                if len(new_mat) == 0:
                    continue
                i = random.randint(0, len(new_mat) - 1)
                new_mat[i] = 'H' if new_mat[i] == 'L' else 'L'
                
            elif move_type == 2: # Add layer
                if len(new_mat) >= max_layers:
                    continue
                pos = random.randint(0, len(new_mat))
                new_mat.insert(pos, random.choice(['L', 'H']))
                new_thick.insert(pos, random.uniform(20, 100))
                new_thick = clip_thicknesses(new_thick)
                
            elif move_type == 3: # Remove layer
                if len(new_mat) <= 1:
                    continue
                pos = random.randint(0, len(new_mat) - 1)
                new_mat.pop(pos)
                new_thick.pop(pos)
                
            elif move_type == 4: # Swap adjacent layers
                if len(new_mat) < 2:
                    continue
                pos = random.randint(0, len(new_mat) - 2)
                new_mat[pos], new_mat[pos+1] = new_mat[pos+1], new_mat[pos]
                new_thick[pos], new_thick[pos+1] = new_thick[pos+1], new_thick[pos]
            
            if not is_valid(new_mat, new_thick):
                n_iter += 1
                continue
                
            new_f = mean_reflectance(new_mat, new_thick)
            
            delta_f = new_f - current_f
            if delta_f < 0 or random.random() < np.exp(-delta_f / temp):
                current_mat = new_mat
                current_thick = new_thick
                current_f = new_f
                
                if current_f < best_f:
                    best_f = current_f
                    best_mat = list(current_mat)
                    best_thick = list(current_thick)
            
            n_iter += 1
            
            # Periodic cooling
            if n_iter % 1000 == 0:
                t_start = temp # Reset start for next segment? No, just keep going.
        
        return best_mat, best_thick, best_f

    # Run SA
    sa_mat, sa_thick, sa_score = simulated_annealing(best_materials, best_thicknesses)
    
    if sa_score < best_score:
        best_score = sa_score
        best_materials = sa_mat
        best_thicknesses = sa_thick

    # --- Phase 3: Nelder-Mead Refinement ---
    def nelder_mead(materials, x0, max_iter=500):
        """
        Nelder-Mead optimization for thicknesses.
        """
        n = len(x0)
        if n == 0:
            return x0, mean_reflectance(materials, x0)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        simplex = [list(x0)]
        for i in range(n):
            point = list(x0)
            delta = 5.0
            point[i] += delta
            point = clip_thicknesses(point)
            simplex.append(point)
        
        fvals = [mean_reflectance(materials, s) for s in simplex]
        
        for _ in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if fvals[-1] - fvals[0] < 1e-12:
                break
            
            centroid = np.zeros(n)
            for i in range(n):
                centroid += np.array(simplex[i])
            centroid /= n
            
            xr = centroid + alpha * (centroid - np.array(simplex[-1]))
            xr = clip_thicknesses(xr)
            fr = mean_reflectance(materials, xr)
            
            if fr < fvals[0]:
                xe = centroid + gamma * (np.array(xr) - centroid)
                xe = clip_thicknesses(xe)
                fe = mean_reflectance(materials, xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                xc = centroid - rho * (np.array(simplex[-1]) - centroid)
                xc = clip_thicknesses(xc)
                fc = mean_reflectance(materials, xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = clip_thicknesses(simplex[i])
                        fvals[i] = mean_reflectance(materials, simplex[i])
        
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    # Run Nelder-Mead on the best SA result
    if best_thicknesses:
        nm_thick, nm_score = nelder_mead(best_materials, best_thicknesses)
        
        if nm_score < best_score:
            best_score = nm_score
            best_thicknesses = nm_thick

    # Final validation
    best_thicknesses = clip_thicknesses(best_thicknesses)
    
    if not is_valid(best_materials, best_thicknesses):
        # Fallback to a valid simple structure
        return ["L"], [lambda_c / (4 * nL)]
        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
