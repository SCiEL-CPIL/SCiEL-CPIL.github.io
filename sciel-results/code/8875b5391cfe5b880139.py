import time
import math
import numpy as np
import random

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)

    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d != 0:
            beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent.
    Returns penalty if constraints are violated.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
        
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 100.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _project(x, max_layer, max_total):
    x = np.clip(x, 0.0, max_layer)
    if np.sum(x) > max_total:
        x = x * (max_total / np.sum(x))
        x = np.clip(x, 0.0, max_layer)
    return x

def _optimize_nm(mats, x0, problem, wls, n_mat, deadline, max_iter=200):
    n = len(x0)
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    def project(x):
        return _project(x, max_layer, max_total)

    x = np.array(x0, dtype=float)
    x = project(x)
    
    def objective(x):
        x = project(x)
        return _mean_reflectance_percent(mats, list(x), problem, wls, n_mat)

    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = min(80.0, 0.15 * (max_layer - 0.0))
        if step < 1.0: step = 1.0
        xi[i] += step
        simplex.append(project(xi))
        
    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        spread = abs(f_vals[0] - f_vals[-1])
        if spread < 1e-9:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = project(centroid + alpha * (centroid - simplex[-1]))
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
            
        if f_r < f_vals[0]:
            x_e = project(centroid + gamma * (x_r - centroid))
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
            
        x_c = project(centroid + rho * (simplex[-1] - centroid))
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
            
        for i in range(1, n + 1):
            simplex[i] = project(simplex[0] + sigma * (simplex[i] - simplex[0]))
            f_vals[i] = objective(simplex[i])
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return project(simplex[best_idx]), f_vals[best_idx]

def _reflectance_and_grad(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    n_layers = len(materials)
    
    R_base = _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat)
    f_base = float(np.mean(R_base))
    
    grad = np.zeros(n_layers)
    h = 1e-4
    for i in range(n_layers):
        th_p = list(thicknesses)
        th_p[i] += h
        R_p = _reflectance_array(materials, th_p, n_ambient, n_substrate, wls, n_mat)
        f_p = float(np.mean(R_p))
        
        if thicknesses[i] - h >= 0:
            th_m = list(thicknesses)
            th_m[i] -= h
            R_m = _reflectance_array(materials, th_m, n_ambient, n_substrate, wls, n_mat)
            f_m = float(np.mean(R_m))
            grad[i] = (f_p - f_m) / (2 * h)
        else:
            grad[i] = (f_p - f_base) / h
            
    return f_base, grad

def _gradient_polish(materials, x0, problem, wls, n_mat, deadline, max_iter=30):
    n = len(x0)
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    def project(x):
        return _project(x, max_layer, max_total)

    x = np.array(x0, dtype=float)
    x = project(x)
    f_curr = _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)
    
    step_size = 5.0
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        f, g = _reflectance_and_grad(materials, list(x), problem["ambient_index"], problem["substrate_index"], wls, n_mat)
        g = g * 100.0
        
        if np.max(np.abs(g)) < 1e-6:
            break
            
        x_new = x - step_size * g
        x_new = project(x_new)
        
        f_new = _mean_reflectance_percent(materials, list(x_new), problem, wls, n_mat)
        
        if f_new < f_curr:
            x = x_new
            f_curr = f_new
            step_size *= 1.05
        else:
            step_size *= 0.6
            if step_size < 0.01:
                break
                
    return x, f_curr

def _ga_step(pop, problem, wls, n_mat, deadline):
    """
    One generation of the genetic algorithm.
    """
    n_pop = len(pop)
    max_layers = problem["max_layers"]
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    # Sort by fitness (lower is better)
    pop.sort(key=lambda x: x[1])
    
    # Elitism: keep top 20%
    elite_count = max(2, n_pop // 5)
    elite = [pop[i][:] for i in range(elite_count)]
    
    # Tournament selection
    def select():
        cand = random.sample(range(n_pop), 3)
        best_idx = min(cand, key=lambda i: pop[i][1])
        return pop[best_idx][:]
    
    # Crossover
    def crossover(p1, p2):
        # Pad to same length with random layers
        max_len = max(len(p1[0]), len(p2[0]))
        if max_len > max_layers:
            max_len = max_layers
        
        # Determine new sequence length
        if random.random() < 0.3:
            new_len = random.randint(4, max_layers)
        else:
            new_len = random.choice([len(p1[0]), len(p2[0]), max_len])
            new_len = min(new_len, max_layers)
            new_len = max(2, new_len)
        
        mats = []
        ths = []
        
        for i in range(new_len):
            # Random choice of parent for this position
            if i < len(p1[0]) and random.random() < 0.5:
                mats.append(p1[0][i])
                ths.append(p1[1][i])
            elif i < len(p2[0]) and random.random() < 0.5:
                mats.append(p2[0][i])
                ths.append(p2[1][i])
            else:
                # New random layer
                mats.append(random.choice(['L', 'H']))
                ths.append(random.uniform(0, max_layer))
        
        return mats, ths
    
    # Mutation
    def mutate(mats, ths, rate=0.15):
        # Thickness mutation
        for i in range(len(ths)):
            if random.random() < rate:
                if random.random() < 0.2:
                    ths[i] = random.uniform(0, max_layer)
                else:
                    ths[i] = max(0, ths[i] + random.gauss(0, 20))
        
        # Material mutation
        for i in range(len(mats)):
            if random.random() < rate * 0.5:
                mats[i] = 'H' if mats[i] == 'L' else 'L'
        
        # Length mutation
        if random.random() < 0.1:
            if len(mats) > 2 and random.random() < 0.5:
                # Remove random layer
                idx = random.randint(0, len(mats) - 1)
                del mats[idx]
                del ths[idx]
            elif len(mats) < max_layers and random.random() < 0.5:
                # Add random layer
                pos = random.randint(0, len(mats))
                mats.insert(pos, random.choice(['L', 'H']))
                ths.insert(pos, random.uniform(0, max_layer))
        
        return mats, ths
    
    # Repair
    def repair(mats, ths):
        # Enforce constraints
        ths = [max(0, min(max_layer, t)) for t in ths]
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
            ths = [max(0, min(max_layer, t)) for t in ths]
        return mats, ths
    
    new_pop = elite
    
    # Generate new individuals
    while len(new_pop) < n_pop:
        if time.time() > deadline:
            break
            
        p1 = select()
        p2 = select()
        
        mats, ths = crossover(p1, p2)
        mats, ths = mutate(mats, ths)
        mats, ths = repair(mats, ths)
        
        # Evaluate
        val = _mean_reflectance_percent(mats, ths, problem, wls, n_mat)
        new_pop.append((mats, val, ths))
    
    # Local optimization on top few
    top_count = min(3, len(new_pop))
    new_pop.sort(key=lambda x: x[1])
    
    for i in range(top_count):
        if time.time() > deadline:
            break
        mats, val, ths = new_pop[i]
        x_opt, f_opt = _optimize_nm(mats, ths, problem, wls, n_mat, deadline - 1.0, max_iter=50)
        x_pol, f_pol = _gradient_polish(mats, x_opt, problem, wls, n_mat, deadline - 0.5, max_iter=10)
        new_pop[i] = (mats, f_pol, list(x_pol))
    
    return new_pop

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    # Initial population
    pop = []
    
    # Deterministic initial solutions
    for length in [4, 6, 8, 10]:
        # Start with L
        seq_l = ['L' if i % 2 == 0 else 'H' for i in range(length)]
        # Start with H
        seq_h = ['H' if i % 2 == 0 else 'L' for i in range(length)]
        
        for seq in [seq_l, seq_h]:
            # Quarter-wave initial guess
            lambda_gm = math.sqrt(band_start * band_end)
            ths = [lambda_gm / (4.0 * n_mat[m]) for m in seq]
            ths = [max(0, min(max_layer, t)) for t in ths]
            total = sum(ths)
            if total > max_total:
                scale = max_total / total
                ths = [t * scale for t in ths]
                ths = [max(0, min(max_layer, t)) for t in ths]
            
            val = _mean_reflectance_percent(seq, ths, problem, wls, n_mat)
            pop.append((seq, val, ths))
            
            if val < best_val:
                best_val = val
                best_mats = seq[:]
                best_ths = ths[:]
    
    # Add random initial solutions
    for _ in range(8):
        length = random.randint(4, 12)
        mats = [random.choice(['L', 'H']) for _ in range(length)]
        ths = [random.uniform(0, max_layer) for _ in range(length)]
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        
        val = _mean_reflectance_percent(mats, ths, problem, wls, n_mat)
        pop.append((mats, val, ths))
        
        if val < best_val:
            best_val = val
            best_mats = mats[:]
            best_ths = ths[:]
    
    # Run GA
    n_generations = 15
    for gen in range(n_generations):
        if time.time() > deadline - 10.0:
            break
        
        pop = _ga_step(pop, problem, wls, n_mat, deadline - 5.0)
        
        # Update best
        pop.sort(key=lambda x: x[1])
        if pop[0][1] < best_val:
            best_val = pop[0][1]
            best_mats = pop[0][0][:]
            best_ths = pop[0][2][:]
    
    # Final polish on the best solution
    if best_mats is not None and time.time() < deadline - 5.0:
        # Try multiple local optimization strategies
        candidates = []
        
        # 1. Nelder-Mead
        x_nm, f_nm = _optimize_nm(best_mats, best_ths, problem, wls, n_mat, deadline - 2.0, max_iter=100)
        candidates.append((f_nm, x_nm))
        
        # 2. Nelder-Mead + Gradient Polish
        x_nm2, f_nm2 = _optimize_nm(best_mats, best_ths, problem, wls, n_mat, deadline - 1.5, max_iter=60)
        x_gp, f_gp = _gradient_polish(best_mats, x_nm2, problem, wls, n_mat, deadline - 0.5, max_iter=20)
        candidates.append((f_gp, x_gp))
        
        # 3. Direct Gradient Polish
        x_gp2, f_gp2 = _gradient_polish(best_mats, best_ths, problem, wls, n_mat, deadline - 1.0, max_iter=30)
        candidates.append((f_gp2, x_gp2))
        
        # Pick the best
        candidates.sort(key=lambda x: x[0])
        if candidates:
            final_val, final_ths = candidates[0]
            if final_val < best_val:
                best_val = final_val
                best_ths = list(final_ths)
    
    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
    
    # Final constraint enforcement
    best_ths = [max(0.0, min(max_layer, t)) for t in best_ths]
    total = sum(best_ths)
    if total > max_total:
        scale = max_total / total
        best_ths = [t * scale for t in best_ths]
        best_ths = [max(0.0, min(max_layer, t)) for t in best_ths]
    
    return list(best_mats), list(best_ths)
