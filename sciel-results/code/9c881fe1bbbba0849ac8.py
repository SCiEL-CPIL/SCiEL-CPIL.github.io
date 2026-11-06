"""Broadband antireflection coating design via global search + local polish."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Compute mean reflectance in percent over 401 equally spaced wavelengths in band_nm."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return float(np.abs(r) ** 2 * 100.0)
    
    n_wl = len(wavelengths)
    n_layers = len(labels)
    
    # We'll compute the characteristic matrix for each wavelength
    # M_total = M1 * M2 * ... * Mn
    # where Mi = [[cos(d_i), (j/s_i)sin(d_i)], [(j*n_i)sin(d_i), cos(d_i)]]
    # d_i = 2*pi*n_i*t_i / lambda
    # s_i = n_i (normal incidence, lossless)
    
    total_R = 0.0
    chunk_size = 50
    for start in range(0, n_wl, chunk_size):
        end = min(start + chunk_size, n_wl)
        lams = wavelengths[start:end]
        nl = len(lams)
        
        # Initialize identity matrices
        M00 = np.ones(nl, dtype=complex)
        M01 = np.zeros(nl, dtype=complex)
        M10 = np.zeros(nl, dtype=complex)
        M11 = np.ones(nl, dtype=complex)
        
        for i in range(n_layers):
            ni = n_mat[i]
            ti = d[i]
            delta = 2.0 * np.pi * ni * ti / lams
            c = np.cos(delta)
            s = np.sin(delta)
            
            a = c
            b = 1j * s / ni
            cc = 1j * ni * s
            dd = c
            
            new_M00 = M00 * a + M01 * cc
            new_M01 = M00 * b + M01 * dd
            new_M10 = M10 * a + M11 * cc
            new_M11 = M10 * b + M11 * dd
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
        
        Y = M10 / M00
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2 * 100.0
        total_R += np.sum(R)
    
    return total_R / n_wl


def _nelder_mead(objective, x0, bounds, max_iter=100, tol=1e-8, deadline=None):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    # Create initial simplex
    simplex = [np.array(x0, dtype=float)]
    fvals = [objective(x0)]
    
    for i in range(n):
        xi = np.array(x0, dtype=float)
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        xi[i] += step
        xi = np.clip(xi, [b[0] for b in bounds], [b[1] for b in bounds])
        simplex.append(xi)
        fvals.append(objective(xi))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = objective(xe)
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
            # Contraction
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = objective(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = objective(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    def _eval(labels, thicknesses):
        """Evaluate reflectance and check constraints."""
        if len(labels) == 0:
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return float(np.abs(r) ** 2 * 100.0)
        
        # Check constraints
        if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
            return float('inf')
        if sum(thicknesses) > max_total_nm + 1e-9:
            return float('inf')
        
        # Clip tiny negatives
        thicknesses = [max(0.0, t) for t in thicknesses]
        
        return _compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm)
    
    def _optimize_thicknesses(labels, x0, deadline):
        """Optimize thicknesses for a fixed layer sequence."""
        n = len(labels)
        if n == 0:
            return None, float('inf')
        
        bounds = [(0.0, max_layer_nm)] * n
        
        # Ensure initial point is feasible
        x0 = np.clip(np.array(x0, dtype=float), 0.0, max_layer_nm)
        if np.sum(x0) > max_total_nm:
            x0 = x0 * (max_total_nm / np.sum(x0))
            x0 = np.clip(x0, 0.0, max_layer_nm)
        
        def obj(x):
            x = np.clip(x, 0.0, max_layer_nm)
            total = np.sum(x)
            if total > max_total_nm:
                return _eval(labels, list(x)) + 100.0 * (total - max_total_nm) / max_total_nm
            return _eval(labels, list(x))
        
        x_best, f_best = _nelder_mead(obj, x0, bounds, max_iter=80, deadline=deadline)
        
        if f_best < best_score and np.sum(x_best) <= max_total_nm + 1e-9:
            actual = _eval(labels, list(x_best))
            if actual < float('inf'):
                return list(x_best), actual
        
        return None, float('inf')
    
    # Phase 1: Evaluate structured initial configurations
    initial_configs = []
    for n_layers in range(1, min(11, max_layers + 1)):
        for start in ['L', 'H']:
            seq = [start if i % 2 == 0 else ('H' if start == 'L' else 'L') for i in range(n_layers)]
            thicks = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in seq]
            if sum(thicks) <= max_total_nm:
                initial_configs.append((seq, thicks))
    
    # Add some non-alternating patterns
    extra_patterns = [
        ["L", "H", "L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L", "H", "L"],
        ["L", "H", "H", "L", "H", "L"],
        ["H", "L", "L", "H", "L", "H"],
        ["L", "H", "L", "L", "H", "L", "H"],
        ["H", "L", "H", "H", "L", "H", "L"],
    ]
    for seq in extra_patterns:
        if len(seq) <= max_layers:
            thicks = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in seq]
            if sum(thicks) <= max_total_nm:
                initial_configs.append((seq, thicks))
    
    for seq, thicks in initial_configs:
        if time.time() > deadline - 20.0:
            break
        score = _eval(seq, thicks)
        if score < best_score:
            best_score = score
            best_labels = seq[:]
            best_thicknesses = thicks[:]
    
    # Phase 2: Optimize thicknesses for the best initial configurations
    for seq, thicks in initial_configs[:15]:
        if time.time() > deadline - 30.0:
            break
        opt_thick, opt_score = _optimize_thicknesses(seq, thicks, deadline)
        if opt_thick is not None and opt_score < best_score:
            best_score = opt_score
            best_labels = seq[:]
            best_thicknesses = opt_thick[:]
    
    # Phase 3: Genetic algorithm over layer sequences + thickness optimization
    if time.time() < deadline - 40.0:
        random.seed(12345)
        
        def _fitness(labels, thicknesses):
            return _eval(labels, thicknesses)
        
        # Initialize population
        pop_size = 30
        population = []
        
        # Add best solution
        if best_labels is not None:
            population.append((best_labels[:], best_thicknesses[:]))
        
        # Generate random individuals
        for _ in range(pop_size - 1):
            n_layers = random.randint(2, min(12, max_layers))
            labels = [random.choice(['L', 'H']) for _ in range(n_layers)]
            # Random thicknesses near quarter-wave
            thicks = []
            for l in labels:
                base = centre / (4.0 * (n_L if l == 'L' else n_H))
                t = base * random.uniform(0.5, 1.5)
                t = min(t, max_layer_nm)
                thicks.append(t)
            # Scale to fit if needed
            if sum(thicks) > max_total_nm:
                scale = max_total_nm / sum(thicks)
                thicks = [t * scale for t in thicks]
            population.append((labels, thicks))
        
        # Evaluate population
        pop_scores = [_fitness(labels, thicks) for labels, thicks in population]
        
        # Track best
        if best_labels is None or min(pop_scores) < best_score:
            best_idx = np.argmin(pop_scores)
            best_score = pop_scores[best_idx]
            best_labels = population[best_idx][0][:]
            best_thicknesses = population[best_idx][1][:]
        
        # GA loop
        for gen in range(50):
            if time.time() > deadline - 25.0:
                break
            
            # Sort by fitness
            combined = list(zip(pop_scores, population))
            combined.sort(key=lambda x: x[0])
            pop_scores = [c[0] for c in combined]
            population = [c[1] for c in combined]
            
            # Elitism: keep top 5
            new_population = [population[i][:] for i in range(5)]
            
            # Crossover and mutation
            while len(new_population) < pop_size:
                if time.time() > deadline - 20.0:
                    break
                
                # Tournament selection
                def _select():
                    idxs = random.sample(range(len(population)), min(3, len(population)))
                    best = min(idxs, key=lambda i: pop_scores[i])
                    return population[best]
                
                parent1 = _select()
                parent2 = _select()
                
                # Crossover: take layers from one parent, thicknesses from other
                if random.random() < 0.5:
                    child_labels = parent1[0][:]
                    child_thicks = parent2[1][:]
                else:
                    child_labels = parent2[0][:]
                    child_thicks = parent1[1][:]
                
                # Mutation: change a layer
                if random.random() < 0.3 and len(child_labels) > 0:
                    idx = random.randint(0, len(child_labels) - 1)
                    child_labels[idx] = 'H' if child_labels[idx] == 'L' else 'L'
                
                # Mutation: add/remove a layer
                if random.random() < 0.2:
                    if random.random() < 0.5 and len(child_labels) < max_layers:
                        pos = random.randint(0, len(child_labels))
                        new_layer = random.choice(['L', 'H'])
                        child_labels.insert(pos, new_layer)
                        child_thicks.insert(pos, centre / (4.0 * (n_L if new_layer == 'L' else n_H)))
                    elif random.random() < 0.5 and len(child_labels) > 1:
                        pos = random.randint(0, len(child_labels) - 1)
                        child_labels.pop(pos)
                        child_thicks.pop(pos)
                
                # Mutation: perturb thicknesses
                for i in range(len(child_thicks)):
                    if random.random() < 0.3:
                        child_thicks[i] *= random.uniform(0.7, 1.3)
                        child_thicks[i] = min(max(child_thicks[i], 0.0), max_layer_nm)
                
                # Ensure feasibility
                if sum(child_thicks) > max_total_nm:
                    scale = max_total_nm / sum(child_thicks)
                    child_thicks = [t * scale for t in child_thicks]
                
                new_population.append((child_labels, child_thicks))
            
            population = new_population
            pop_scores = [_fitness(labels, thicks) for labels, thicks in population]
            
            # Track best
            best_idx = np.argmin(pop_scores)
            if pop_scores[best_idx] < best_score:
                best_score = pop_scores[best_idx]
                best_labels = population[best_idx][0][:]
                best_thicknesses = population[best_idx][1][:]
    
    # Phase 4: Local polish on the best solution
    if best_labels is not None and time.time() < deadline - 15.0:
        opt_thick, opt_score = _optimize_thicknesses(best_labels, best_thicknesses, deadline)
        if opt_thick is not None and opt_score < best_score:
            best_score = opt_score
            best_thicknesses = opt_thick[:]
    
    # Phase 5: Try a few more random sequences with optimization
    if time.time() < deadline - 15.0:
        random.seed(99999)
        for trial in range(20):
            if time.time() > deadline - 10.0:
                break
            n_layers = random.randint(3, min(10, max_layers))
            labels = [random.choice(['L', 'H']) for _ in range(n_layers)]
            thicks = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels]
            if sum(thicks) > max_total_nm:
                scale = max_total_nm / sum(thicks)
                thicks = [t * scale for t in thicks]
            
            opt_thick, opt_score = _optimize_thicknesses(labels, thicks, deadline)
            if opt_thick is not None and opt_score < best_score:
                best_score = opt_score
                best_labels = labels[:]
                best_thicknesses = opt_thick[:]
    
    # Final constraint enforcement
    if best_labels is None:
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
