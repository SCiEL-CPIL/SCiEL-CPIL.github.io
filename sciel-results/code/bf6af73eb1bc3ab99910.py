"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
New approach: Genetic Algorithm (GA) with Simulated Annealing (SA) polish.
Uses FFT for fast overlap calculation.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # Iterative correction to handle clipping effects
    for _ in range(10):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        mask = (h > 0) & (h < 1)
        if not np.any(mask):
            # If all are 0 or 1, we can't adjust easily, but sum should be close
            # Force adjustment on one element
            idx = np.argmin(h) if diff > 0 else np.argmax(h)
            h[idx] += diff
        else:
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
        
    # Final exact fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        # Add remainder to the element with most room
        if diff > 0:
            idx = np.argmax(1.0 - h)
        else:
            idx = np.argmin(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _ga_mutation(h, sigma):
    """
    Add Gaussian noise and re-project to feasible set.
    """
    noise = np.random.normal(0, sigma, N)
    h_new = h + noise
    return _fix_sum(h_new)


def _ga_crossover(h1, h2):
    """
    Blend crossover: h_new = alpha * h1 + (1-alpha) * h2
    """
    alpha = np.random.uniform(0.2, 0.8)
    h_new = alpha * h1 + (1 - alpha) * h2
    return _fix_sum(h_new)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    # Population parameters
    pop_size = 200
    elite_size = 5
    mutation_rate = 0.1
    initial_sigma = 0.1
    min_sigma = 0.001
    
    # Initialize population
    # Mix of random, cosine, and sine starts
    population = []
    i = np.arange(N)
    
    # Deterministic starts
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    population.append(h_cos)
    
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    population.append(h_sin)
    
    h_flat = np.full(N, 0.5)
    population.append(h_flat)
    
    # Random starts
    while len(population) < pop_size:
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _fix_sum(h_rand)
        population.append(h_rand)
        
    # Evaluate initial population
    scores = np.array([_score(h) for h in population])
    best_idx = np.argmin(scores)
    best_h = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # GA Loop
    gen = 0
    while time.time() - t0 < _DEADLINE - 20:
        gen += 1
        # Sort population by score
        sorted_indices = np.argsort(scores)
        new_population = [population[i].copy() for i in sorted_indices[:elite_size]]
        
        # Create offspring
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if scores[idx1] < scores[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
                
            idx3 = np.random.randint(0, pop_size)
            idx4 = np.random.randint(0, pop_size)
            if scores[idx3] < scores[idx4]:
                parent2 = population[idx3]
            else:
                parent2 = population[idx4]
                
            # Crossover
            if np.random.random() < 0.5:
                child = _ga_crossover(parent1, parent2)
            else:
                child = parent1.copy()
                
            # Mutation
            if np.random.random() < mutation_rate:
                sigma = initial_sigma * (0.95 ** gen)
                sigma = max(sigma, min_sigma)
                child = _ga_mutation(child, sigma)
                
            new_population.append(child)
            
        # Evaluate new population
        population = new_population
        scores = np.array([_score(h) for h in population])
        
        # Update best
        if np.min(scores) < best_score:
            best_idx = np.argmin(scores)
            best_score = scores[best_idx]
            best_h = population[best_idx].copy()
            
    # Simulated Annealing Polish on best_h
    if time.time() - t0 < _DEADLINE - 5:
        h_current = best_h.copy()
        score_current = best_score
        T = 0.01
        T_min = 0.0001
        cooling = 0.95
        
        while T > T_min and time.time() - t0 < _DEADLINE - 1:
            # Pick a random index to perturb
            idx = np.random.randint(0, N)
            delta = np.random.normal(0, 0.01)
            
            h_trial = h_current.copy()
            h_trial[idx] += delta
            h_trial = _fix_sum(h_trial)
            
            score_trial = _score(h_trial)
            delta_score = score_trial - score_current
            
            if delta_score < 0 or np.random.random() < np.exp(-delta_score / T):
                h_current = h_trial
                score_current = score_trial
                if score_current < best_score:
                    best_score = score_current
                    best_h = h_current.copy()
                    
            T *= cooling
            
    return best_h
