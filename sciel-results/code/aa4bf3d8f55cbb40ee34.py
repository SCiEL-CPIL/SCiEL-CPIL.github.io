"""Combined design program: vectorized AR stack evaluation + DE global search + Nelder-Mead refinement."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband AR coating using vectorized evaluation + DE global search + Nelder-Mead optimization."""
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points from 400 to 800 nm)
    lambdas = np.linspace(band[0], band[1], 401)
    lambda_c = 0.5 * (band[0] + band[1])
    n_wl = len(lambdas)
    
    def mean_reflectance(materials, thicknesses):
        """Compute mean reflectance in percent over the band using vectorized characteristic matrix method."""
        n_layers = np.array([nL if m == 'L' else nH for m in materials], dtype=float)
        n = len(materials)
        if n == 0:
            r = (n0 - ns) / (n0 + ns)
            return (np.abs(r) ** 2) * 100.0
        
        # Vectorized computation over all wavelengths
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
        """Check constraints."""
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        return True
    
    def clip_thicknesses(materials, thicknesses):
        """Clip thicknesses to valid range and ensure total constraint."""
        t = np.array(thicknesses, dtype=float)
        t = np.clip(t, 0.0, max_layer_nm)
        total = np.sum(t)
        if total > max_total_nm:
            scale = max_total_nm / total
            t = t * scale
        return [float(x) for x in t]
    
    # --- Generate candidate structures with diverse starting points ---
    candidates = []
    
    qw_L = lambda_c / (4.0 * nL)
    qw_H = lambda_c / (4.0 * nH)
    
    # Single layer
    candidates.append(("L", [qw_L]))
    candidates.append(("H", [qw_H]))
    
    # Two layers
    candidates.append(("LH", [qw_L, qw_H]))
    candidates.append(("HL", [qw_H, qw_L]))
    
    # Three layers
    candidates.append(("LHL", [qw_L, qw_H, qw_L]))
    candidates.append(("HLH", [qw_H, qw_L, qw_H]))
    candidates.append(("LLH", [qw_L, qw_L, qw_H]))
    candidates.append(("HHL", [qw_H, qw_H, qw_L]))
    
    # Four layers
    candidates.append(("LHHL", [qw_L, qw_H, qw_H, qw_L]))
    candidates.append(("HLHL", [qw_H, qw_L, qw_L, qw_H]))
    candidates.append(("LHLH", [qw_L, qw_H, qw_L, qw_H]))
    
    # Five layers
    candidates.append(("LHLHL", [qw_L, qw_H, qw_L, qw_H, qw_L]))
    candidates.append(("HLHLH", [qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Six layers
    candidates.append(("LHLHHL", [qw_L, qw_H, qw_L, qw_H, qw_H, qw_L]))
    candidates.append(("HLHLHH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_H]))
    
    # Seven layers
    candidates.append(("LHLHLHL", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    candidates.append(("HLHLHLH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Eight layers
    candidates.append(("LHLHLHHL", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_H, qw_L]))
    candidates.append(("HLHLHLHH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_H]))
    
    # Nine layers
    candidates.append(("LHLHLHLHL", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    candidates.append(("HLHLHLHLH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Ten layers
    candidates.append(("LHLHLHLHHL", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_H, qw_L]))
    candidates.append(("HLHLHLHLHH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_H]))
    
    # Evaluate all candidates and find the best
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    for mat_str, thick in candidates:
        materials = list(mat_str)
        if not is_valid(materials, thick):
            continue
        score = mean_reflectance(materials, thick)
        if score < best_score:
            best_score = score
            best_materials = materials
            best_thicknesses = list(thick)
    
    if best_materials is None:
        return ["L"], [qw_L]
    
    # --- Nelder-Mead local refinement ---
    def nelder_mead_optimize(materials, x0, max_iter=1500):
        """Simple Nelder-Mead optimization for thickness vector."""
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        simplex = [list(x0)]
        for i in range(n):
            point = list(x0)
            delta = 10.0 if abs(x0[i]) < 1.0 else 0.1 * abs(x0[i])
            point[i] += delta
            point = clip_thicknesses(materials, point)
            simplex.append(point)
        
        fvals = []
        for i in range(n + 1):
            if (time.time() - start_time) > DEADLINE:
                return x0, mean_reflectance(materials, x0)
            fvals.append(mean_reflectance(materials, simplex[i]))
        
        for _ in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if fvals[-1] - fvals[0] < 1e-10:
                break
            
            centroid = np.zeros(n)
            for i in range(n):
                centroid += np.array(simplex[i])
            centroid /= n
            
            xr = centroid + alpha * (centroid - np.array(simplex[-1]))
            xr = clip_thicknesses(materials, xr)
            fr = mean_reflectance(materials, xr)
            
            if fr < fvals[0]:
                xe = centroid + gamma * (np.array(xr) - centroid)
                xe = clip_thicknesses(materials, xe)
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
                if fr < fvals[-1]:
                    xc = centroid + rho * (np.array(xr) - centroid)
                else:
                    xc = centroid - rho * (np.array(xr) - centroid)
                xc = clip_thicknesses(materials, xc)
                fc = mean_reflectance(materials, xc)
                
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = clip_thicknesses(materials, simplex[i])
                        fvals[i] = mean_reflectance(materials, simplex[i])
        
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
    
    # --- Differential Evolution global search ---
    def differential_evolution(materials, x0, pop_size=30, max_iter=50, F=0.8, CR=0.9):
        """Differential Evolution for global optimization of thicknesses."""
        n = len(x0)
        
        # Initialize population
        pop = [list(x0)]
        for _ in range(pop_size - 1):
            ind = []
            for i in range(n):
                val = random.uniform(0, max_layer_nm)
                ind.append(val)
            ind = clip_thicknesses(materials, ind)
            pop.append(ind)
        
        # Evaluate initial population
        fvals = [mean_reflectance(materials, ind) for ind in pop]
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx]
        best_f = fvals[best_idx]
        
        for gen in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
            
            new_pop = [list(pop[best_idx])]
            
            for i in range(pop_size - 1):
                # Select three distinct random indices different from i
                while True:
                    a, b, c = random.sample(range(pop_size), 3)
                    if a != i and b != i and c != i:
                        break
                
                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = []
                for j in range(n):
                    val = pop[a][j] + F * (pop[b][j] - pop[c][j])
                    v.append(val)
                v = clip_thicknesses(materials, v)
                
                # Crossover
                u = list(pop[i])
                for j in range(n):
                    if random.random() < CR or j == random.randint(0, n - 1):
                        u[j] = v[j]
                u = clip_thicknesses(materials, u)
                
                # Selection
                fu = mean_reflectance(materials, u)
                if fu <= fvals[i]:
                    new_pop.append(u)
                    fvals[i] = fu
                else:
                    new_pop.append(list(pop[i]))
            
            pop = new_pop
            
            # Track global best
            curr_best_idx = np.argmin(fvals)
            if fvals[curr_best_idx] < best_f:
                best_f = fvals[curr_best_idx]
                best_x = pop[curr_best_idx]
        
        return best_x, best_f
    
    # Run DE on multiple top candidates
    # Sort candidates by initial score
    scored_candidates = []
    for mat_str, thick in candidates:
        materials = list(mat_str)
        if not is_valid(materials, thick):
            continue
        score = mean_reflectance(materials, thick)
        scored_candidates.append((score, materials, list(thick)))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run DE on top 5 candidates
    for score, materials, thick in scored_candidates[:5]:
        if (time.time() - start_time) > DEADLINE:
            break
        
        x_de, score_de = differential_evolution(materials, thick)
        
        if score_de < best_score:
            best_score = score_de
            best_materials = materials
            best_thicknesses = x_de
    
    # Run Nelder-Mead on the best candidate after DE
    x0 = list(best_thicknesses)
    x_nm, score_nm = nelder_mead_optimize(best_materials, x0)
    
    if score_nm < best_score:
        best_thicknesses = x_nm
        best_score = score_nm
    
    # Final validation and clamping
    best_thicknesses = clip_thicknesses(best_materials, best_thicknesses)
    
    if not is_valid(best_materials, best_thicknesses):
        # Fallback to first valid candidate
        for mat_str, thick in candidates:
            materials = list(mat_str)
            if is_valid(materials, thick):
                return materials, thick
        return ["L"], [qw_L]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
