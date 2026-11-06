"""
Search for the optimal 24-coefficient (+1/-1) polynomial minimizing sup norm on unit circle.
Uses Genetic Algorithm + local search (hill climbing) with golden-section refinement.
"""
import numpy as np
import time
import random
import math
from itertools import product

N = 24
TARGET = 1.2583758  # exhaustive optimum score (max|P|/sqrt(24))
MAX_TIME = 110.0  # safety margin before 120s limit (evaluator kills at 150s)


def eval_score(coeffs):
    """Compute max |P(z)| / sqrt(N) for |z|=1 via dense sampling + golden-section refinement."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n = len(coeffs)
    
    # Dense sampling at many points on unit circle
    # Use 24*1024 = 24576 points for initial scan
    npts = 24 * 1024
    theta = 2.0 * np.pi * np.arange(npts) / npts
    z = np.exp(1j * theta)
    
    # Evaluate polynomial P(z) = sum c_k z^k
    # Use Horner's method via numpy
    P = np.zeros(npts, dtype=np.complex128)
    for k in range(n - 1, -1, -1):
        P = P * z + coeffs[k]
    
    mag = np.abs(P)
    peak_idx = np.argmax(mag)
    peak_val = mag[peak_idx]
    
    # Golden-section refinement around the peak
    # The peak is near theta[peak_idx]
    # Search in a window around it
    dtheta = 2.0 * np.pi / npts
    lo = theta[peak_idx] - dtheta
    hi = theta[peak_idx] + dtheta
    
    def f(theta_val):
        z_val = np.exp(1j * theta_val)
        # Horner
        p_val = 0.0 + 0.0j
        for k in range(n - 1, -1, -1):
            p_val = p_val * z_val + coeffs[k]
        return abs(p_val)
    
    # Golden section search
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    
    for _ in range(60):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    
    refined_val = max(fc, fd)
    final_peak = max(peak_val, refined_val)
    
    return final_peak / math.sqrt(n)


def fast_eval(coeffs):
    """Faster evaluation using FFT-based approach for the GA fitness."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n = len(coeffs)
    
    # Use 24*512 points for faster GA evaluation
    npts = 24 * 512
    theta = 2.0 * np.pi * np.arange(npts) / npts
    z = np.exp(1j * theta)
    
    P = np.zeros(npts, dtype=np.complex128)
    for k in range(n - 1, -1, -1):
        P = P * z + coeffs[k]
    
    mag = np.abs(P)
    return np.max(mag) / math.sqrt(n)


def local_search(coeffs, time_limit=5.0):
    """Hill climbing: try flipping each coefficient, keep if improves."""
    coeffs = list(coeffs)
    best_score = eval_score(coeffs)
    start = time.time()
    
    improved = True
    while improved:
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(N):
            if time.time() - start > time_limit:
                break
            # Try flipping coefficient i
            coeffs[i] = -coeffs[i]
            new_score = eval_score(coeffs)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                coeffs[i] = -coeffs[i]  # revert
    
    return coeffs, best_score


def genetic_algorithm():
    """GA with local search to find optimal 24-coeff polynomial."""
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    pop_size = 40
    mutation_rate = 0.15
    elite_count = 4
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        ind = [random.choice([1, -1]) for _ in range(N)]
        population.append(ind)
    
    best_individual = None
    best_score = float('inf')
    
    generation = 0
    while time.time() - start_time < MAX_TIME:
        generation += 1
        
        # Evaluate fitness
        scored = []
        for ind in population:
            score = fast_eval(ind)
            scored.append((score, ind))
        
        scored.sort(key=lambda x: x[0])
        
        # Update best
        if scored[0][0] < best_score:
            best_score = scored[0][0]
            best_individual = list(scored[0][1])
        
        # Check if we've found the optimum
        if best_score <= TARGET + 1e-6:
            # Do a final precise evaluation
            precise_score = eval_score(best_individual)
            return best_individual, precise_score
        
        # Elitism
        new_pop = [list(ind) for _, ind in scored[:elite_count]]
        
        # Crossover and mutation to fill rest
        while len(new_pop) < pop_size:
            if time.time() - start_time > MAX_TIME - 10:
                break
            # Tournament selection
            i1 = random.randint(0, min(5, len(scored)-1))
            i2 = random.randint(0, min(5, len(scored)-1))
            parent1 = scored[i1][1] if i1 < i2 else scored[i2][1]
            parent2 = scored[i2][1] if i1 < i2 else scored[i1][1]
            
            # Single-point crossover
            point = random.randint(1, N-1)
            child = list(parent1[:point]) + list(parent2[point:])
            
            # Mutation
            for j in range(N):
                if random.random() < mutation_rate:
                    child[j] = -child[j]
            
            new_pop.append(child)
        
        population = new_pop
    
    # Final local search on best individual
    if best_individual is not None:
        refined, final_score = local_search(best_individual, time_limit=max(1.0, MAX_TIME - (time.time() - start_time)))
        if final_score < best_score:
            best_individual = refined
            best_score = final_score
    
    # Final precise evaluation
    if best_individual is not None:
        best_score = eval_score(best_individual)
    
    return best_individual, best_score


def construct():
    """Return 24 coefficients in {+1, -1} minimizing sup norm."""
    start_time = time.time()
    
    # Start with Golay-Rudin-Shapiro as initial best
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    grs = p[:24]
    grs_score = eval_score(grs)
    
    # Try the GA
    try:
        best_ind, best_score = genetic_algorithm()
    except Exception:
        best_ind = grs
        best_score = grs_score
    
    # If GA found something better, use it; otherwise use GRS
    if best_score < grs_score:
        return best_ind
    else:
        return grs
