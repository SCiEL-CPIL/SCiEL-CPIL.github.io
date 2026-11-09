"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

STRUCTURAL CHANGE:
  The previous approach treated the layer sequence as a fixed structure
  (e.g., alternating or gradient profiles) and only optimized thicknesses.
  This program unifies the sequence and thickness optimization using a
  genetic algorithm over the full binary sequence of materials. For each
  candidate sequence, thicknesses are optimized via differential evolution
  and local polish. This allows the search to discover novel layer
  arrangements that pure local optimization from hand-crafted starting
  points may miss.
"""

import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    from scipy.optimize import differential_evolution as _scipy_de
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _mean_reflectance(materials, thicknesses, problem):
    """Mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    n_ds = np.array([n * d for n, d in zip(ns, ds)])
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _constrain_thicknesses(ths, max_total, max_layer):
    """Constrain thicknesses to be within [0, max_layer] and sum <= max_total."""
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
    return ths


def _optimize_thicknesses(materials, problem, max_total, max_layer, n_starts=3, time_limit=15.0):
    """
    Optimize thicknesses for a given material sequence.
    Returns (thicknesses, score) or (None, inf) if no valid solution found.
    """
    start_time = time.time()
    n = len(materials)
    if n == 0:
        return [], 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                            (problem["substrate_index"] + problem["ambient_index"])) ** 2

    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    lo = np.full(n, math.log(1.0))
    hi = np.full(n, math.log(max_layer))

    def eval_x(x):
        t = np.exp(np.clip(x, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        return _mean_reflectance(materials, list(t), problem)

    best_t = None
    best_s = float("inf")

    # Try differential evolution
    if _HAVE_SCIPY and (time.time() - start_time) < time_limit * 0.5:
        try:
            de_res = _scipy_de(
                eval_x, bounds=list(zip(lo, hi)),
                maxiter=20, tol=1e-5, seed=42,
                polish=False, popsize=10, mutation=(0.5, 1.0),
                recombination=0.8, init='latinhypercube'
            )
            xc = np.clip(de_res.x, lo, hi)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            s = _mean_reflectance(materials, list(t), problem)
            if s < best_s:
                best_s = s
                best_t = list(t)
        except Exception:
            pass

    # Local polish from multiple starts
    for i in range(n_starts):
        if time.time() - start_time > time_limit - 2:
            break
        # Generate starting point
        x0 = []
        for m in materials:
            n_m = n_mat[m]
            # Quarter-wave at band center as starting guess
            qw = 600.0 / (4.0 * n_m)  # 600 nm center
            t = qw * random.uniform(0.5, 1.5)
            t = max(1.0, min(max_layer, t))
            x0.append(math.log(t))
        x0 = np.array(x0)

        if _HAVE_SCIPY:
            try:
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 150, "maxfun": 2000, "maxls": 20},
                )
                xc = np.clip(res.x, lo, hi)
            except Exception:
                xc = x0
        else:
            xc = x0

        t = np.exp(np.clip(xc, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        s = _mean_reflectance(materials, list(t), problem)
        if s < best_s:
            best_s = s
            best_t = list(t)

    return best_t, best_s


def design(problem):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before the 150s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(42)
    np.random.seed(42)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]

    best_ths = None
    best_mats = None
    best_score = float("inf")

    def update_best(mats, ths, score):
        nonlocal best_ths, best_mats, best_score
        ths = _constrain_thicknesses(ths, max_total, max_layer)
        if any(t < 0 or t > max_layer for t in ths):
            return
        if score < best_score:
            best_score = score
            best_ths = list(ths)
            best_mats = list(mats)

    # Phase 1: Initialize population with diverse structures
    population = []

    # Add alternating sequences of various lengths
    for n in range(4, max_layers + 1, 2):
        alt = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        population.append(alt)

    # Add all-L and all-H
    for n in range(2, max_layers + 1, 3):
        population.append(['L'] * n)
        population.append(['H'] * n)

    # Add gradient sequences
    for n in [8, 12, 16, 20]:
        if n <= max_layers:
            # Exponential gradient
            k = math.log(n_sub / n_amb)
            target_ns = [n_amb * math.exp(k * (i / (n - 1))) for i in range(n)]
            seq = ['L' if abs(nt - nL) <= abs(nt - nH) else 'H' for nt in target_ns]
            population.append(seq)

    # Add random sequences
    for n in range(6, max_layers + 1, 2):
        population.append(['L' if random.random() < 0.5 else 'H' for _ in range(n)])

    # Evaluate initial population
    scores = []
    for mats in population:
        ths, s = _optimize_thicknesses(mats, problem, max_total, max_layer, n_starts=2, time_limit=10.0)
        if ths is not None:
            scores.append(s)
            update_best(mats, ths, s)
        else:
            scores.append(float("inf"))

    # Phase 2: Evolution
    max_gen = 10
    for gen in range(max_gen):
        if time.time() >= deadline - 20:
            break

        # Sort by score
        indexed = sorted(enumerate(zip(scores, population)), key=lambda x: x[1][0])
        sorted_pairs = [(pair[0], pair[1]) for pair in indexed]
        scores_sorted = [s for s, _ in sorted_pairs]
        pop_sorted = [p for _, p in sorted_pairs]

        # Keep best 5
        n_elite = min(5, len(pop_sorted))
        next_pop = pop_sorted[:n_elite][:]

        # Generate offspring
        n_offspring = len(population) - n_elite
        for _ in range(n_offspring):
            if time.time() >= deadline - 15:
                break

            # Tournament selection
            def tournament_select(k=3):
                candidates = random.sample(range(len(pop_sorted)), min(k, len(pop_sorted)))
                best_idx = min(candidates, key=lambda i: scores_sorted[i])
                return pop_sorted[best_idx]

            p1 = tournament_select()
            p2 = tournament_select()

            # Crossover
            child_len = max(len(p1), len(p2))
            # Pad if necessary
            if len(p1) < child_len:
                p1 = p1 + ['L' if random.random() < 0.5 else 'H' for _ in range(child_len - len(p1))]
            if len(p2) < child_len:
                p2 = p2 + ['L' if random.random() < 0.5 else 'H' for _ in range(child_len - len(p2))]

            cut = random.randint(0, child_len - 1)
            child = p1[:cut] + p2[cut:]

            # Mutation
            for j in range(len(child)):
                if random.random() < 0.1:
                    child[j] = 'L' if child[j] == 'H' else 'H'

            # Insertion/Deletion
            if random.random() < 0.1 and len(child) > 2:
                idx = random.randint(0, len(child) - 1)
                child.pop(idx)
            elif random.random() < 0.1 and len(child) < max_layers:
                idx = random.randint(0, len(child))
                child.insert(idx, 'L' if random.random() < 0.5 else 'H')

            next_pop.append(child)

        # Evaluate offspring
        next_scores = []
        for i, mats in enumerate(next_pop):
            if i < n_elite:
                # Reuse score
                orig_idx = next_pop.index(mats)
                next_scores.append(scores_sorted[orig_idx])
            else:
                ths, s = _optimize_thicknesses(mats, problem, max_total, max_layer, n_starts=2, time_limit=10.0)
                if ths is not None:
                    next_scores.append(s)
                    update_best(mats, ths, s)
                else:
                    next_scores.append(float("inf"))

        population = next_pop
        scores = next_scores

    # Phase 3: Final polish on the best solution
    if best_mats is not None and time.time() < deadline - 10:
        ths, score = _optimize_thicknesses(
            best_mats, problem, max_total, max_layer,
            n_starts=8,
            time_limit=20.0
        )
        if ths is not None:
            update_best(best_mats, ths, score)

    # Fallback
    if best_ths is None:
        best_mats = ['L']
        best_ths = [max(0.0, min(max_layer, 600.0 / (4.0 * nL)))]
        best_score = _mean_reflectance(best_mats, best_ths, problem)

    return list(best_mats), list(best_ths)
