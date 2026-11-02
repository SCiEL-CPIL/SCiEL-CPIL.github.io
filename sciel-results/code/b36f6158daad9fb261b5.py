"""Optimized program for the third autocorrelation inequality task using a Memetic Genetic Algorithm."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _compute_peak_sidelobe(s):
    """
    Compute the peak absolute sidelobe of the autocorrelation of binary sequence s.
    s is a 1D numpy array of floats (+1.0 or -1.0).
    Returns the peak value (float).
    """
    # Use FFT for efficient autocorrelation computation
    n_fft = 1 << (2 * N - 1).bit_length()
    S = np.fft.rfft(s, n=n_fft)
    acf = np.fft.irfft(S * np.conj(S), n=n_fft)[:N]
    # acf[0] is the main lobe (energy). Sidelobes are acf[1:].
    # Due to symmetry, we only need to check positive lags 1 to N-1.
    return np.max(np.abs(acf[1:]))

def _local_search(s, max_iter=20):
    """
    Perform simple hill climbing by flipping single bits.
    Returns the improved sequence and its peak sidelobe.
    """
    current_s = s.copy()
    current_peak = _compute_peak_sidelobe(current_s)
    
    # Precompute FFT size for speed in loop
    n_fft = 1 << (2 * N - 1).bit_length()
    
    for _ in range(max_iter):
        improved = False
        # Randomly pick a few indices to try flipping to avoid O(N^2) full scan every step
        # But for N=400, a full scan is 400 FFTs. 20 iter * 400 flips = 8000 FFTs.
        # Each FFT is ~10-20us. 8000 * 20us = 0.16s. This might be too slow if done for many individuals.
        # Strategy: Try a random subset of flips per iteration.
        
        # Let's try flipping 10 random bits and accept the best improvement if any
        indices = np.random.choice(N, size=10, replace=False)
        best_idx = -1
        best_new_peak = current_peak
        
        for idx in indices:
            current_s[idx] *= -1.0
            new_peak = _compute_peak_sidelobe(current_s)
            if new_peak < best_new_peak:
                best_new_peak = new_peak
                best_idx = idx
            current_s[idx] *= -1.0 # revert
        
        if best_idx != -1:
            current_s[best_idx] *= -1.0
            current_peak = best_new_peak
            improved = True
        
        if not improved:
            break
            
    return current_s, current_peak

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    # Population parameters
    pop_size = 60
    elite_size = 5
    tournament_size = 5
    
    # Initialize population
    population = [rng.choice([-1.0, 1.0], size=N) for _ in range(pop_size)]
    fitnesses = [_compute_peak_sidelobe(s) for s in population]
    
    # Track global best
    best_s = population[np.argmin(fitnesses)].copy()
    best_peak = min(fitnesses)
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Sort population by fitness (ascending)
        indices = np.argsort(fitnesses)
        population = [population[i] for i in indices]
        fitnesses = [fitnesses[i] for i in indices]
        
        # Update global best
        if fitnesses[0] < best_peak:
            best_peak = fitnesses[0]
            best_s = population[0].copy()
            
        # Elitism: Keep the best elite_size individuals
        new_population = population[:elite_size]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
                
            # Tournament Selection
            parents = []
            for _ in range(2):
                candidates = rng.integers(0, pop_size, size=tournament_size)
                best_cand_idx = candidates[np.argmin([fitnesses[c] for c in candidates])]
                parents.append(population[best_cand_idx])
            
            # Uniform Crossover
            child = np.where(rng.random(N) < 0.5, parents[0], parents[1])
            
            # Mutation: Flip 1-3 random bits
            num_mut = rng.integers(1, 4)
            mut_indices = rng.integers(0, N, size=num_mut)
            child[mut_indices] *= -1.0
            
            new_population.append(child)
        
        # Evaluate new population
        # To save time, we can do local search on the top few of the new population
        new_fitnesses = [0.0] * len(new_population)
        
        # First, evaluate all without local search
        for i, s in enumerate(new_population):
            new_fitnesses[i] = _compute_peak_sidelobe(s)
            
        # Apply local search to the top 10 individuals of the new generation
        # This is the "Memetic" part
        top_indices = np.argsort(new_fitnesses)[:10]
        for idx in top_indices:
            if time.time() > deadline:
                break
            improved_s, improved_peak = _local_search(new_population[idx], max_iter=10)
            new_population[idx] = improved_s
            new_fitnesses[idx] = improved_peak
            
        # Update population and fitnesses
        population = new_population
        fitnesses = new_fitnesses
        
    return best_s
# EVOLVE-BLOCK-END
