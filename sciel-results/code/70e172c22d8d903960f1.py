"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _reflectance_spectrum(thicknesses, materials, n_amb, n_sub, band_nm, n_wl=401):
    """Compute mean reflectance using characteristic matrix method."""
    wl = np.linspace(band_nm[0], band_nm[1], n_wl)
    n_L = materials['L']
    n_H = materials['H']
    
    n_layers = len(thicknesses)
    n_mat = np.zeros(n_layers)
    for i, mat in enumerate(materials):
        if mat == 'L':
            n_mat[i] = n_L
        else:
            n_mat[i] = n_H
    
    nd = n_mat * thicknesses
    k = 2 * np.pi / wl
    phi = k * nd
    
    m11 = np.ones(n_wl, dtype=complex)
    m12 = np.zeros(n_wl, dtype=complex)
    m21 = np.zeros(n_wl, dtype=complex)
    m22 = np.ones(n_wl, dtype=complex)
    
    for j in range(n_layers):
        cj = np.cos(phi[:, j])
        sj = np.sin(phi[:, j])
        nj = n_mat[j]
        
        nm11 = m11 * cj + m12 * (1j * sj / nj)
        nm12 = m11 * (1j * sj / nj) + m12 * cj
        nm21 = m21 * cj + m22 * (1j * nj * sj)
        nm22 = m21 * (1j * nj * sj) + m22 * cj
        
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22
    
    B = m11 + m12 * n_sub
    C = m21 + m22 * n_sub
    Y = B / C
    
    R = np.abs((Y - n_amb) / (Y + n_amb)) ** 2
    R = np.real(R)
    
    return np.mean(R) * 100.0

def _nelder_mead(objective, x0, bounds, max_iter=200, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tol=1e-8):
    """Nelder-Mead optimization."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    
    simplex = [x.copy() for _ in range(n + 1)]
    for i in range(n):
        simplex[i + 1] = x.copy()
        simplex[i + 1][i] += 0.05 * max(abs(x[i]), 1.0)
    
    f = [objective(v) for v in simplex]
    
    for iteration in range(max_iter):
        order = np.argsort(f)
        simplex = [simplex[i] for i in order]
        f = [f[i] for i in order]
        
        if abs(f[0] - f[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < f[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f[-1] = fe
            else:
                simplex[-1] = xr
                f[-1] = fr
        elif fr < f[-2]:
            simplex[-1] = xr
            f[-1] = fr
        else:
            if fr < f[-1]:
                xc = centroid + sigma * (xr - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    f[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        f[i] = objective(simplex[i])
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f[i] = objective(simplex[i])
        
        for i in range(n + 1):
            simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
            f[i] = objective(simplex[i])
    
    order = np.argsort(f)
    return simplex[order[0]], f[order[0]]

def _differential_evolution(objective, bounds, pop_size=30, max_iter=50, F=0.8, CR=0.9, seed=42):
    """Differential evolution for global optimization."""
    n = len(bounds)
    rng = np.random.default_rng(seed)
    
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = rng.uniform(lb, ub, size=(pop_size, n))
    f = np.array([objective(pop[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(f)
    best_x = pop[best_idx].copy()
    best_f = f[best_idx]
    
    for iteration in range(max_iter):
        for i in range(pop_size):
            # Select three distinct individuals
            a, b, c = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)
            
            # Create trial vector
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lb, ub)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.integers(n)
            for j in range(n):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Selection
            ft = objective(trial)
            if ft < f[i]:
                pop[i] = trial
                f[i] = ft
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
    
    return best_x, best_f

def design(problem):
    """Design broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Define diverse material patterns
    patterns = [
        # Alternating patterns
        ['L', 'H'] * 10,
        ['H', 'L'] * 10,
        # Block patterns
        ['L'] * 10 + ['H'] * 10,
        ['H'] * 10 + ['L'] * 10,
        ['L', 'L', 'H', 'H'] * 5,
        ['H', 'H', 'L', 'L'] * 5,
        ['L', 'L', 'L', 'H', 'H', 'H'] * 3 + ['L', 'L'],
        ['H', 'H', 'H', 'L', 'L', 'L'] * 3 + ['H', 'H'],
        # Symmetric patterns
        ['L', 'H', 'L', 'H', 'L', 'H', 'H', 'L', 'H', 'L', 'L', 'H', 'L', 'H', 'H', 'L', 'H', 'L'],
        ['H', 'L', 'H', 'L', 'H', 'L', 'L', 'H', 'L', 'H', 'H', 'L', 'H', 'L', 'L', 'H', 'L', 'H'],
        # Graded-like patterns
        ['L', 'L', 'H', 'L', 'H', 'H', 'L', 'H', 'H', 'H', 'H', 'H', 'L', 'H', 'H', 'L', 'H', 'L', 'L', 'L'],
        ['H', 'H', 'L', 'H', 'L', 'L', 'H', 'L', 'L', 'L', 'L', 'L', 'H', 'L', 'L', 'H', 'L', 'H', 'H', 'H'],
        # Fewer layers
        ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'],
    ]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    for pattern in patterns:
        if time.time() > deadline:
            break
        
        n_layers = len(pattern)
        
        # Initial thicknesses: quarter-wave at center wavelength
        init_thick = np.array([center_wl / (4.0 * n_L) if m == 'L' else center_wl / (4.0 * n_H) 
                              for m in pattern])
        
        # Scale to fit within total budget
        total = np.sum(init_thick)
        if total > max_total_nm:
            init_thick *= max_total_nm / total
        
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        def objective(thick):
            if np.sum(thick) > max_total_nm + 1e-6:
                return 1000.0
            return _reflectance_spectrum(thick, pattern, n_amb, n_sub, band_nm)
        
        # First: Differential Evolution for global search
        try:
            x_de, f_de = _differential_evolution(objective, bounds, pop_size=20, max_iter=30, seed=hash(tuple(pattern)) % 10000)
            
            # Then: Nelder-Mead for local polish
            x_opt, f_opt = _nelder_mead(objective, x_de, bounds, max_iter=100)
            
            if f_opt < best_score:
                best_score = f_opt
                best_materials = pattern
                best_thicknesses = x_opt
        except Exception:
            pass
    
    # Fallback
    if best_materials is None:
        return ["L"], [center_wl / (4.0 * n_L)]
    
    best_thicknesses = np.clip(best_thicknesses, 0.0, max_layer_nm)
    if np.sum(best_thicknesses) > max_total_nm:
        best_thicknesses *= max_total_nm / np.sum(best_thicknesses)
    
    return best_materials, list(best_thicknesses)
# EVOLVE-BLOCK-END
