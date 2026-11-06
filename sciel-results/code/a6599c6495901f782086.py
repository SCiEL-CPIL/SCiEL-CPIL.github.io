"""Structural change: GA/SA hybrid over binary sequences, fitness = peak |P(z)| on unit circle.
We exploit the fact that for a ±1 sequence of length N, the values at the N-th roots of unity
are the DFT samples. The maximum over |z|=1 is bounded above by the max DFT magnitude (by
Bernstein/interpolation for low-degree polynomials, the peak is near a root). We use a
population-based genetic algorithm with simulated-annealing-like acceptance, evaluating
the true sup-norm via dense sampling + golden-section refinement on the top few peaks.
"""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize_scalar

_N = 24
_SQRT_N = math.sqrt(_N)
_DEADLINE = 145.0  # safety margin before 150s kill

def _poly_magnitude(coeffs, z):
    """Compute |sum c_k z^k| for complex z."""
    # Use Horner's method for stability
    acc = 0.0 + 0.0j
    for c in reversed(coeffs):
        acc = acc * z + c
    return abs(acc)

def _evaluate_sup_norm(coeffs, n_samples=1024):
    """Estimate max |P(z)| for |z|=1 via dense sampling, then refine top peaks."""
    # Dense sampling on the unit circle
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    z_vals = np.exp(1j * angles)
    
    # Evaluate polynomial at all points using numpy
    # coeffs[k] is coefficient of z^k
    # P(z) = sum_{k=0}^{23} c_k z^k
    # We can use np.polynomial.polynomial.polyval or manual
    # Manual with broadcasting:
    # z_vals[:, None] ** np.arange(_N)[None, :] * coeffs[None, :]
    # But that's 1024x24 complex ops, fine.
    
    powers = z_vals[:, None] ** np.arange(_N)[None, :]
    vals = powers @ coeffs  # shape (n_samples,)
    mags = np.abs(vals)
    
    # Find top few peaks and refine
    top_indices = np.argsort(mags)[-5:]
    best = 0.0
    for idx in top_indices:
        a = angles[max(0, idx - 1)]
        b = angles[min(n_samples - 1, idx + 1)]
        # Ensure a < b
        if a > b:
            a, b = b, a
        if b - a < 1e-10:
            a = angles[idx] - 2 * np.pi / n_samples
            b = angles[idx] + 2 * np.pi / n_samples
        
        def obj(theta):
            z = np.exp(1j * theta)
            return -_poly_magnitude(coeffs, z)
        
        res = minimize_scalar(obj, bounds=(a, b), method='bounded', options={'xatol': 1e-12})
        if res.success:
            val = -res.fun
            if val > best:
                best = val
    
    # Also check the global max from sampling in case refinement missed it
    sample_max = float(np.max(mags))
    return max(best, sample_max)

def _fitness(coeffs):
    """Lower is better: sup_norm / sqrt(N)."""
    return _evaluate_sup_norm(coeffs) / _SQRT_N

def _random_seq():
    return np.random.choice([-1, 1], size=_N)

def _crossover(a, b):
    """Uniform crossover."""
    mask = np.random.random(_N) < 0.5
    return np.where(mask, a, b)

def _mutate(seq, rate=0.05):
    """Bit-flip mutation."""
    s = seq.copy()
    flip_mask = np.random.random(_N) < rate
    s[flip_mask] *= -1
    return s

def construct():
    start_time = time.time()
    
    # Initialize population
    pop_size = 200
    population = [_random_seq() for _ in range(pop_size)]
    
    # Evaluate initial population
    fitnesses = [_fitness(p) for p in population]
    
    best_seq = population[np.argmin(fitnesses)]
    best_fit = min(fitnesses)
    
    # Also try the known-good Golay-Rudin-Shapiro as a seed
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    grs = np.array(p[:_N], dtype=int)
    grs_fit = _fitness(grs)
    if grs_fit < best_fit:
        best_fit = grs_fit
        best_seq = grs.copy()
    
    # Genetic algorithm with simulated annealing
    gen = 0
    while time.time() - start_time < _DEADLINE:
        gen += 1
        new_population = []
        
        # Elitism: keep best few
        elite_count = max(2, pop_size // 20)
        elite_indices = np.argsort(fitnesses)[:elite_count]
        new_population.extend([population[i].copy() for i in elite_indices])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if fitnesses[idx1] < fitnesses[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
            
            idx3 = np.random.randint(0, pop_size)
            idx4 = np.random.randint(0, pop_size)
            if fitnesses[idx3] < fitnesses[idx4]:
                parent2 = population[idx3]
            else:
                parent2 = population[idx4]
            
            child = _crossover(parent1, parent2)
            child = _mutate(child, rate=0.05)
            new_population.append(child)
        
        # Evaluate new population (skip elites which we already know)
        new_fitnesses = list(fitnesses[i] for i in elite_indices)
        for child in new_population[elite_count:]:
            new_fitnesses.append(_fitness(child))
        
        # Update best
        min_idx = np.argmin(new_fitnesses)
        if new_fitnesses[min_idx] < best_fit:
            best_fit = new_fitnesses[min_idx]
            best_seq = new_population[min_idx].copy()
        
        population = new_population
        fitnesses = new_fitnesses
        
        # Check deadline
        if time.time() - start_time > _DEADLINE - 5:
            break
    
    # Convert to list of ints
    return [int(c) for c in best_seq]
