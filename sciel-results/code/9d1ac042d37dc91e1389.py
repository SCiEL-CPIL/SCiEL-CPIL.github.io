"""Broadband antireflection coating design via discrete genetic algorithm."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _fast_compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Faster version using vectorization over wavelengths."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)

    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)

    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r) ** 2 * 100.0

    total_r = 0.0
    for start in range(0, len(wavelengths), 50):
        end = min(start + 50, len(wavelengths))
        lams = wavelengths[start:end]

        M = np.zeros((len(lams), 2, 2), dtype=complex)
        M[:, 0, 0] = 1.0
        M[:, 1, 1] = 1.0

        for i in range(len(labels)):
            delta = 2.0 * np.pi * n_mat[i] * d[i] / lams
            c = np.cos(delta)
            s = np.sin(delta)

            Mi_00 = c
            Mi_01 = 1j * s / n_mat[i]
            Mi_10 = 1j * n_mat[i] * s
            Mi_11 = c

            M_new = np.zeros_like(M)
            M_new[:, 0, 0] = M[:, 0, 0] * Mi_00 + M[:, 0, 1] * Mi_10
            M_new[:, 0, 1] = M[:, 0, 0] * Mi_01 + M[:, 0, 1] * Mi_11
            M_new[:, 1, 0] = M[:, 1, 0] * Mi_00 + M[:, 1, 1] * Mi_10
            M_new[:, 1, 1] = M[:, 1, 0] * Mi_01 + M[:, 1, 1] * Mi_11
            M = M_new

        Y = M[:, 1, 0] / M[:, 0, 0]
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2 * 100.0
        total_r += np.sum(R)

    return total_r / len(wavelengths)


def _decode_chromosome(chrom, max_layers, max_layer_nm, max_total_nm):
    """Decode a fixed-length chromosome into (labels, thicknesses).
    Chromosome layout: [n_layers (int 1..max_layers), then for each layer:
    material bit (0=L,1=H), thickness (int 0..max_layer_nm)]
    We encode n_layers as first element, then 2*n_layers elements after.
    To keep fixed size, we encode max_layers*2+1 integers.
    """
    n_layers = int(chrom[0])
    n_layers = max(1, min(n_layers, max_layers))
    
    labels = []
    thicknesses = []
    total = 0.0
    for i in range(n_layers):
        mat_bit = int(chrom[1 + 2*i]) % 2
        thick = int(chrom[2 + 2*i])
        thick = max(0, min(thick, max_layer_nm))
        labels.append('H' if mat_bit == 1 else 'L')
        thicknesses.append(float(thick))
        total += thick
    
    if total > max_total_nm and total > 0:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return labels, thicknesses


def _encode_chromosome(labels, thicknesses, max_layers):
    """Encode (labels, thicknesses) into a fixed-length integer chromosome."""
    n_layers = len(labels)
    chrom = [n_layers]
    for i in range(max_layers):
        if i < n_layers:
            mat = 1 if labels[i] == 'H' else 0
            thick = int(round(thicknesses[i]))
            thick = max(0, min(thick, 400))
            chrom.extend([mat, thick])
        else:
            chrom.extend([0, 0])
    return chrom


def _mutate(chrom, max_layers, max_layer_nm, mutation_rate=0.15):
    """Apply mutation to a chromosome."""
    new_chrom = chrom[:]
    n_layers = int(new_chrom[0])
    
    # Mutate n_layers with low probability
    if random.random() < 0.05:
        new_chrom[0] = max(1, min(max_layers, n_layers + random.choice([-2, -1, 1, 2])))
        n_layers = new_chrom[0]
    
    # Mutate each layer's material and thickness
    for i in range(max_layers):
        mat_idx = 1 + 2 * i
        thick_idx = 2 + 2 * i
        if i < n_layers:
            if random.random() < mutation_rate:
                new_chrom[mat_idx] = random.randint(0, 1)
            if random.random() < mutation_rate:
                # Gaussian-ish mutation on thickness
                old_t = new_chrom[thick_idx]
                delta = int(random.gauss(0, 30))
                new_t = max(0, min(max_layer_nm, old_t + delta))
                new_chrom[thick_idx] = new_t
        else:
            # For unused slots, occasionally set them to non-zero to allow growth
            if random.random() < 0.02:
                new_chrom[mat_idx] = random.randint(0, 1)
                new_chrom[thick_idx] = random.randint(10, max_layer_nm)
    
    return new_chrom


def _crossover(chrom1, chrom2):
    """Uniform crossover."""
    child = []
    for a, b in zip(chrom1, chrom2):
        child.append(a if random.random() < 0.5 else b)
    return child


def _tournament_select(pop, fitnesses, k=3):
    """Tournament selection."""
    indices = random.sample(range(len(pop)), min(k, len(pop)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return pop[best_idx][:]


def design(problem):
    """Design a broadband antireflection coating stack using a genetic algorithm."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    random.seed(42)
    np.random.seed(42)

    # Chromosome length: 1 (n_layers) + 2*max_layers (mat, thick per layer)
    chrom_len = 1 + 2 * max_layers

    # Initial population: mix of quarter-wave starts and random
    pop_size = 60
    population = []
    fitnesses = []

    # Add quarter-wave initial guesses
    centre = 0.5 * (band_nm[0] + band_nm[1])
    for n_l in range(1, min(8, max_layers + 1)):
        for start_mat in ['L', 'H']:
            labels = [start_mat if i % 2 == 0 else ('H' if start_mat == 'L' else 'L') for i in range(n_l)]
            thicknesses = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels]
            if sum(thicknesses) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicknesses):
                chrom = _encode_chromosome(labels, thicknesses, max_layers)
                population.append(chrom)

    # Fill rest with random chromosomes
    while len(population) < pop_size:
        n_l = random.randint(1, max_layers)
        chrom = [n_l]
        for i in range(max_layers):
            if i < n_l:
                chrom.append(random.randint(0, 1))
                chrom.append(random.randint(10, max_layer_nm))
            else:
                chrom.append(0)
                chrom.append(0)
        population.append(chrom)

    # Evaluate initial population
    for chrom in population:
        labels, thicknesses = _decode_chromosome(chrom, max_layers, max_layer_nm, max_total_nm)
        if len(labels) == 0:
            fitnesses.append(float('inf'))
        else:
            fitnesses.append(_fast_compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm))

    best_labels = None
    best_thicknesses = None
    best_score = float('inf')

    # Track best
    for i in range(len(population)):
        if fitnesses[i] < best_score:
            best_score = fitnesses[i]
            labels, thicknesses = _decode_chromosome(population[i], max_layers, max_layer_nm, max_total_nm)
            best_labels = labels
            best_thicknesses = thicknesses

    # GA loop
    gen = 0
    max_gens = 200
    while gen < max_gens:
        if time.time() > deadline - 15.0:
            break

        # Elitism: keep top 10
        sorted_idx = np.argsort(fitnesses)
        elite_count = min(10, len(population))
        new_population = [population[i][:] for i in sorted_idx[:elite_count]]
        new_fitnesses = [fitnesses[i] for i in sorted_idx[:elite_count]]

        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10.0:
                break

            parent1 = _tournament_select(population, fitnesses)
            parent2 = _tournament_select(population, fitnesses)
            child = _crossover(parent1, parent2)
            child = _mutate(child, max_layers, max_layer_nm)
            new_population.append(child)

        # Evaluate new population
        for i in range(len(new_population)):
            if i < elite_count:
                new_fitnesses.append(new_fitnesses[i])
            else:
                labels, thicknesses = _decode_chromosome(new_population[i], max_layers, max_layer_nm, max_total_nm)
                if len(labels) == 0:
                    new_fitnesses.append(float('inf'))
                else:
                    new_fitnesses.append(_fast_compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm))

        # Update best
        for i in range(len(new_population)):
            if new_fitnesses[i] < best_score:
                best_score = new_fitnesses[i]
                labels, thicknesses = _decode_chromosome(new_population[i], max_layers, max_layer_nm, max_total_nm)
                best_labels = labels
                best_thicknesses = thicknesses

        population = new_population
        fitnesses = new_fitnesses
        gen += 1

    # If we have time, do a local refinement on the best solution using Nelder-Mead
    if time.time() < deadline - 20.0 and best_labels is not None:
        labels = best_labels
        n = len(labels)
        if n > 0:
            x0 = np.array(best_thicknesses, dtype=float)
            x0 = np.clip(x0, 0.0, max_layer_nm)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            simplex = [x0.copy()]
            for i in range(n):
                perturbed = x0.copy()
                step = 0.05 * max_layer_nm if x0[i] == 0 else 0.05 * x0[i]
                perturbed[i] += step
                perturbed = np.clip(perturbed, 0.0, max_layer_nm)
                simplex.append(perturbed)

            fvals = []
            for x in simplex:
                if np.sum(x) > max_total_nm:
                    fvals.append(float('inf'))
                else:
                    fvals.append(_fast_compute_mean_reflectance(labels, list(x), n_amb, n_sub, n_L, n_H, band_nm))

            for it in range(100):
                if time.time() > deadline - 10.0:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < 1e-6:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, 0.0, max_layer_nm)
                if np.sum(xr) > max_total_nm:
                    fr = float('inf')
                else:
                    fr = _fast_compute_mean_reflectance(labels, list(xr), n_amb, n_sub, n_L, n_H, band_nm)

                if fr < fvals[0]:
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, 0.0, max_layer_nm)
                    if np.sum(xe) > max_total_nm:
                        fe = float('inf')
                    else:
                        fe = _fast_compute_mean_reflectance(labels, list(xe), n_amb, n_sub, n_L, n_H, band_nm)
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
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, 0.0, max_layer_nm)
                    if np.sum(xc) > max_total_nm:
                        fc = float('inf')
                    else:
                        fc = _fast_compute_mean_reflectance(labels, list(xc), n_amb, n_sub, n_L, n_H, band_nm)

                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], 0.0, max_layer_nm)
                            if np.sum(simplex[i]) > max_total_nm:
                                fvals[i] = float('inf')
                            else:
                                fvals[i] = _fast_compute_mean_reflectance(labels, list(simplex[i]), n_amb, n_sub, n_L, n_H, band_nm)

            best_idx = np.argmin(fvals)
            if fvals[best_idx] < best_score:
                best_score = fvals[best_idx]
                best_thicknesses = list(simplex[best_idx])

    # Fallback
    if best_labels is None:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]

    # Final constraint check
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]

    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
