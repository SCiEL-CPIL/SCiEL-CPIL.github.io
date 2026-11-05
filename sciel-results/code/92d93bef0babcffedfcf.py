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

    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses)
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i in range(n_layers):
        n = ns[i]
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
    Compute mean reflectance in percent with constraint penalties.
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
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _project_to_feasible(x, max_layer, max_total):
    """Project x to feasible region: [0, max_layer] per layer, sum <= max_total."""
    x = np.clip(x, 0, max_layer)
    total = np.sum(x)
    if total > max_total:
        if total > 1e-12:
            x = x * (max_total / total)
        else:
            x = np.zeros_like(x)
    return x

def _nelder_mead(objective, x0, lo, hi, deadline, max_iter=300, tol=1e-10):
    """
    Nelder-Mead with adaptive step size and stagnation detection.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)

    if n == 0:
        return x0, objective(x0)

    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (hi[i] - lo[i]) if hi[i] - lo[i] > 1e-6 else 1.0
        xi[i] += step
        xi = _project_to_feasible(xi, hi[0], np.sum(hi))
        simplex.append(xi)

    f_vals = [objective(x) for x in simplex]

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    stagnation_count = 0
    best_f_so_far = min(f_vals)
    stagnation_window = 20

    for _ in range(max_iter):
        if time.time() > deadline:
            break

        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < tol:
            break

        if f_vals[0] < best_f_so_far:
            best_f_so_far = f_vals[0]
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_window:
            # Reset with larger steps
            for i in range(1, n + 1):
                simplex[i] = _project_to_feasible(simplex[0] + 2.0 * (simplex[i] - simplex[0]), hi[0], np.sum(hi))
                f_vals[i] = objective(simplex[i])
            stagnation_count = 0
            continue

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = _project_to_feasible(centroid + alpha * (centroid - simplex[-1]), hi[0], np.sum(hi))
        f_r = objective(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = _project_to_feasible(centroid + gamma * (x_r - centroid), hi[0], np.sum(hi))
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        x_c = _project_to_feasible(centroid + rho * (simplex[-1] - centroid), hi[0], np.sum(hi))
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = _project_to_feasible(simplex[0] + sigma * (simplex[i] - simplex[0]), hi[0], np.sum(hi))
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return simplex[best_idx], f_vals[best_idx]

def _coordinate_descent(objective, x0, lo, hi, deadline, max_iter=200):
    """
    Coordinate descent with adaptive step size.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    f = objective(x)

    step_sizes = np.array([(hi[i] - lo[i]) * 0.1 if hi[i] - lo[i] > 1e-6 else 1.0 for i in range(n)])

    for _ in range(max_iter):
        if time.time() > deadline:
            break

        improved = False
        for i in range(n):
            for direction in [-1, 1]:
                x_trial = x.copy()
                x_trial[i] += direction * step_sizes[i]
                x_trial = _project_to_feasible(x_trial, hi[0], np.sum(hi))
                f_trial = objective(x_trial)

                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
                    step_sizes[i] *= 1.5
                else:
                    step_sizes[i] *= 0.5

        if not improved:
            break

    return x, f

def _quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls):
    """
    Generate a quarter-wave stack with optimized thicknesses.
    """
    mats = []
    for i in range(n_layers):
        if (i % 2 == 0) == (start_char == 'L'):
            mats.append('L')
        else:
            mats.append('H')

    # Initial guess: quarter-wave at center wavelength
    wl_center = np.mean(wls)
    thicknesses = []
    for m in mats:
        n = n_L if m == 'L' else n_H
        d = wl_center / (4.0 * n)
        thicknesses.append(d)

    return mats, thicknesses

def _multi_quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls, center_wls):
    """
    Generate a multi-quarter-wave stack with different center wavelengths for different layers.
    """
    mats = []
    for i in range(n_layers):
        if (i % 2 == 0) == (start_char == 'L'):
            mats.append('L')
        else:
            mats.append('H')

    # Assign different center wavelengths to different layers
    thicknesses = []
    for i, m in enumerate(mats):
        n = n_L if m == 'L' else n_H
        # Cycle through center wavelengths
        wl = center_wls[i % len(center_wls)]
        d = wl / (4.0 * n)
        thicknesses.append(d)

    return mats, thicknesses

def _random_stack(n_layers, max_layer, max_total):
    """
    Generate a random stack.
    """
    mats = []
    for _ in range(n_layers):
        mats.append(random.choice(['L', 'H']))

    # Random thicknesses
    thicknesses = np.random.uniform(0, max_layer, n_layers)
    total = np.sum(thicknesses)
    if total > max_total:
        thicknesses = thicknesses * (max_total / total)

    return mats, list(thicknesses)

def _optimize_stack(mats, ths, problem, wls, n_mat, deadline, max_layer, max_total):
    """
    Optimize a single stack using coordinate descent followed by Nelder-Mead.
    """
    lo = np.zeros(len(ths))
    hi = np.full(len(ths), max_layer)

    def objective(th):
        th = _project_to_feasible(th, max_layer, max_total)
        return _mean_reflectance_percent(mats, list(th), problem, wls, n_mat)

    x0 = np.array(ths, dtype=float)
    x0 = _project_to_feasible(x0, max_layer, max_total)

    # First, coordinate descent for quick improvement
    if time.time() < deadline - 5:
        x_cd, f_cd = _coordinate_descent(objective, x0, lo, hi, deadline - 5, max_iter=100)
    else:
        x_cd, f_cd = x0, objective(x0)

    # Then, Nelder-Mead for local refinement
    if time.time() < deadline - 2:
        x_nm, f_nm = _nelder_mead(objective, x_cd, lo, hi, deadline - 2, max_iter=200, tol=1e-10)
        if f_nm < f_cd:
            x_cd = x_nm

    return x_cd

def _fast_optimize(mats, ths, problem, wls, n_mat, deadline, max_layer, max_total):
    """
    Fast local optimization: coordinate descent only (for GA evaluations).
    """
    lo = np.zeros(len(ths))
    hi = np.full(len(ths), max_layer)

    def objective(th):
        th = _project_to_feasible(th, max_layer, max_total)
        return _mean_reflectance_percent(mats, list(th), problem, wls, n_mat)

    x0 = np.array(ths, dtype=float)
    x0 = _project_to_feasible(x0, max_layer, max_total)

    if time.time() < deadline:
        x_cd, f_cd = _coordinate_descent(objective, x0, lo, hi, deadline, max_iter=60)
    else:
        x_cd, f_cd = x0, objective(x0)

    return x_cd, f_cd

def _repair_stack(mats, ths, max_layer, max_total):
    """
    Constraint-aware repair: drop zero-thickness layers, enforce max layers and total thickness.
    Returns (mats, ths) as lists.
    """
    # Drop layers with ~zero thickness
    filtered_mats = []
    filtered_ths = []
    for m, t in zip(mats, ths):
        if t > 1e-6:
            filtered_mats.append(m)
            filtered_ths.append(float(t))

    # Enforce max layers
    if len(filtered_mats) > max_layers:
        # Keep the first max_layers layers
        filtered_mats = filtered_mats[:max_layers]
        filtered_ths = filtered_ths[:max_layers]

    # Enforce per-layer max
    filtered_ths = [min(max(t, 0.0), max_layer) for t in filtered_ths]

    # Enforce total thickness
    total = sum(filtered_ths)
    if total > max_total + 1e-6:
        scale = max_total / total if total > 1e-12 else 0.0
        filtered_ths = [t * scale for t in filtered_ths]
        # Re-check per-layer max after scaling (should be fine, but just in case)
        filtered_ths = [min(max(t, 0.0), max_layer) for t in filtered_ths]

    return filtered_mats, filtered_ths

def _ga_optimize(seed_candidates, problem, wls, n_mat, deadline, max_layer, max_total, n_L, n_H):
    """
    Genetic algorithm that evolves material sequences (discrete) and thicknesses (continuous).
    seed_candidates: list of (mats, ths) tuples to seed the population.
    Returns the best (mats, ths) found.
    """
    # Build initial population from seed candidates
    population = []
    for mats, ths in seed_candidates:
        mats_r, ths_r = _repair_stack(mats, ths, max_layer, max_total)
        if len(mats_r) > 0:
            population.append((mats_r, list(ths_r)))

    # If population is empty, create a minimal one
    if len(population) == 0:
        mats_r, ths_r = _repair_stack(['L', 'H'], [600.0 / (4.0 * n_L), 600.0 / (4.0 * n_H)], max_layer, max_total)
        population.append((mats_r, list(ths_r)))

    # Evaluate initial population
    pop_scores = []
    for mats, ths in population:
        x_opt, f_opt = _fast_optimize(mats, ths, problem, wls, n_mat, deadline - 10, max_layer, max_total)
        pop_scores.append((mats, list(x_opt), f_opt))

    # Sort by score (lower is better)
    pop_scores.sort(key=lambda x: x[2])

    best_mats, best_ths, best_score = pop_scores[0]

    # GA parameters
    pop_size = min(12, len(pop_scores) + 8)
    elite_size = 3
    mutation_rate = 0.3
    crossover_rate = 0.7

    # Run GA until deadline
    while time.time() < deadline - 15:
        # Create new population
        new_pop = []

        # Elitism: keep best individuals
        for i in range(min(elite_size, len(pop_scores))):
            new_pop.append((pop_scores[i][0], pop_scores[i][1], pop_scores[i][2]))

        # Fill rest with offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            def tournament_select(k=3):
                candidates = random.sample(pop_scores, min(k, len(pop_scores)))
                return min(candidates, key=lambda x: x[2])

            parent1 = tournament_select()
            parent2 = tournament_select()

            if random.random() < crossover_rate and len(parent1[0]) >= 2 and len(parent2[0]) >= 2:
                # Crossover: pick a random split point
                min_len = min(len(parent1[0]), len(parent2[0]))
                if min_len >= 2:
                    split = random.randint(1, min_len - 1)
                    child_mats = parent1[0][:split] + parent2[0][split:]
                    # Thicknesses: use parent1's for the first part, parent2's for the rest
                    child_ths = parent1[1][:split] + parent2[1][split:]
                else:
                    child_mats = parent1[0][:]
                    child_ths = parent1[1][:]
            else:
                # No crossover: copy parent1
                child_mats = parent1[0][:]
                child_ths = parent1[1][:]

            # Mutation: randomly change material of a layer
            if random.random() < mutation_rate and len(child_mats) > 0:
                idx = random.randint(0, len(child_mats) - 1)
                child_mats[idx] = 'H' if child_mats[idx] == 'L' else 'L'

            # Mutation: randomly adjust a thickness
            if random.random() < mutation_rate and len(child_ths) > 0:
                idx = random.randint(0, len(child_ths) - 1)
                child_ths[idx] = max(0.0, min(max_layer, child_ths[idx] * random.uniform(0.7, 1.3)))

            # Mutation: add a layer
            if random.random() < 0.1 and len(child_mats) < max_layers:
                pos = random.randint(0, len(child_mats))
                new_m = random.choice(['L', 'H'])
                n_new = n_L if new_m == 'L' else n_H
                new_t = random.uniform(50.0, 200.0) / n_new
                child_mats.insert(pos, new_m)
                child_ths.insert(pos, new_t)

            # Mutation: remove a layer
            if random.random() < 0.1 and len(child_mats) > 2:
                pos = random.randint(0, len(child_mats) - 1)
                child_mats.pop(pos)
                child_ths.pop(pos)

            # Repair
            child_mats, child_ths = _repair_stack(child_mats, child_ths, max_layer, max_total)

            if len(child_mats) > 0:
                # Fast optimize
                x_opt, f_opt = _fast_optimize(child_mats, child_ths, problem, wls, n_mat, deadline - 10, max_layer, max_total)
                new_pop.append((child_mats, list(x_opt), f_opt))

        # Update population
        pop_scores = new_pop
        pop_scores.sort(key=lambda x: x[2])

        # Update best
        if pop_scores[0][2] < best_score:
            best_mats, best_ths, best_score = pop_scores[0]

    return best_mats, best_ths, best_score

def design(problem):
    """
    Design a broadband antireflection coating using a discrete-continuous hybrid
    genetic algorithm with constraint-aware repair, combined with Nelder-Mead and
    coordinate descent for local refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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

    def update_best(mats, ths):
        nonlocal best_val, best_mats, best_ths
        ths_arr = np.array(ths, dtype=float)
        ths_arr = _project_to_feasible(ths_arr, max_layer, max_total)

        # Remove zero layers to reduce layer count
        final_mats = []
        final_ths = []
        for m, t in zip(mats, ths_arr):
            if t > 1e-6:
                final_mats.append(m)
                final_ths.append(float(t))

        if len(final_mats) > max_layers:
            return

        val = _mean_reflectance_percent(final_mats, final_ths, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(final_mats)
            best_ths = list(final_ths)
        return val

    # Generate diverse initial guesses
    candidates = []

    # Quarter-wave stacks with different layer counts
    for n_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            mats, ths = _quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls)
            candidates.append((mats, ths))

    # Multi-quarter-wave stacks with different center wavelengths
    center_wls_options = [
        [500, 600, 700],
        [450, 550, 650],
        [500, 550, 600],
        [400, 600, 800],
    ]
    for n_layers in [6, 8, 10, 12]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            for cw in center_wls_options:
                mats, ths = _multi_quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls, cw)
                candidates.append((mats, ths))

    # Random stacks
    for _ in range(8):
        n_layers = random.randint(6, min(20, max_layers))
        mats, ths = _random_stack(n_layers, max_layer, max_total)
        candidates.append((mats, ths))

    # Shuffle candidates
    random.shuffle(candidates)

    # Phase 1: Quick local optimization of each candidate (to get good seeds for GA)
    ga_seeds = []
    for mats, ths in candidates:
        if time.time() > deadline - 40:
            break
        x_opt, f_opt = _fast_optimize(mats, ths, problem, wls, n_mat, deadline - 35, max_layer, max_total)
        ga_seeds.append((mats, list(x_opt)))
        update_best(mats, x_opt)
        update_best(mats, ths)

    # Phase 2: Genetic algorithm to evolve material sequences
    if time.time() < deadline - 25 and len(ga_seeds) > 0:
        ga_mats, ga_ths, ga_score = _ga_optimize(ga_seeds, problem, wls, n_mat, deadline - 20, max_layer, max_total, n_L, n_H)
        if ga_mats is not None and len(ga_mats) > 0:
            update_best(ga_mats, ga_ths)

    # Phase 3: Final polish on the global best
    if best_mats is not None and len(best_mats) > 0 and time.time() < deadline - 5:
        x_polished = _optimize_stack(best_mats, best_ths, problem, wls, n_mat, deadline - 2, max_layer, max_total)
        update_best(best_mats, x_polished)

    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)

    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]

    return list(best_mats), list(best_ths)
