import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] using FFT."""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=_FFT_SIZE)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_FFT_SIZE)
    return np.real(acf[:N])

def _hard_score(s):
    """Calculate the hard peak sidelobe magnitude."""
    A = _calc_acf(s)
    return float(np.max(np.abs(A[1:])))

def _chirp_init(f0, f1, phase, seed=0):
    """Generate a chirp sequence."""
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    # Quadratic phase
    phase_vals = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / N) + phase
    # Add small noise to break symmetries
    noise = rng.normal(0, 0.01, size=N)
    h = np.sign(np.sin(phase_vals + noise))
    h[h == 0] = 1.0
    return h.astype(np.int8)

def _vdc_init(seed=0):
    """Generate a +-1 sequence based on the van der Corput sequence."""
    vdc_vals = np.zeros(N)
    for i in range(N):
        val = 0.0
        div = 1
        n = i
        while n > 0:
            div *= 2
            if n % 2 == 1:
                val += 1.0 / div
            n //= 2
        vdc_vals[i] = val
    shift = seed % N
    vdc_vals = np.roll(vdc_vals, shift)
    h = np.where(vdc_vals < 0.5, -1.0, 1.0)
    return h.astype(np.int8)

def _local_search(s, time_budget, rng):
    """
    Greedy local search: flip bits if it improves the score.
    Uses a time budget to stop early.
    """
    s = s.copy()
    current_score = _hard_score(s)
    start_time = time.time()
    
    # Pre-calculate indices
    indices = np.arange(N)
    
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        # Randomize order to avoid bias
        rng.shuffle(indices)
        
        for i in indices:
            if (time.time() - start_time) >= time_budget:
                break
            
            # Flip bit
            s[i] = -s[i]
            new_score = _hard_score(s)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                s[i] = -s[i]
                
    return s, current_score

def _bde_mutation(pop, x_target, F, rng):
    """
    Binary Differential Evolution mutation.
    v = x_r1 XOR (x_r2 XOR x_r3)
    """
    N_pop = len(pop)
    # Select 3 distinct random individuals
    r1, r2, r3 = rng.choice(N_pop, size=3, replace=False)
    
    # Ensure they are not the target index
    # (Simple retry logic or just accept if population is large enough)
    while r1 == x_target or r2 == x_target or r3 == x_target:
        r1, r2, r3 = rng.choice(N_pop, size=3, replace=False)
        
    x1 = pop[r1]
    x2 = pop[r2]
    x3 = pop[r3]
    
    # XOR operation for binary sequences
    # Note: -1 and 1 representation. 
    # a XOR b is 1 if a != b, -1 if a == b? 
    # Let's map to 0/1 for logic then back, or use arithmetic.
    # Arithmetic: 
    # If we use 0/1: v = x1 ^ (x2 ^ x3)
    # If we use +/-1:
    # Let's convert to 0/1 (0 for -1, 1 for 1)
    # Or simpler: v[i] = x1[i] * x2[i] * x3[i] ? No, that's product.
    # BDE usually uses: v = x1 + F*(x2-x3) for real.
    # For binary, a common mutation is: v[i] = x1[i] if random < 0.5 else (x2[i] XOR x3[i])
    # Let's stick to a robust bit-flip mutation with adaptive rate, as BDE crossover is tricky with +/-1.
    # Actually, let's use a "Differential" approach:
    # diff = x2 - x3 (values in {-2, 0, 2})
    # If diff[i] != 0, x2[i] and x3[i] are different.
    # We can use this to guide mutation: flip bits where x2 and x3 differ, with probability F.
    
    diff = x2 - x3
    mask = (diff != 0)
    # Flip bits in mask with probability F
    flip_prob = F
    flips = rng.random(N) < flip_prob
    mutation_mask = mask & flips
    
    v = x_target.copy()
    v[mutation_mask] = -v[mutation_mask]
    
    return v

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    rng = np.random.default_rng(42)
    
    # 1. Generate Initial Population
    pop_size = 30
    population = []
    scores = []
    
    # Add Chirps
    for f0 in [0.1, 0.5, 1.0]:
        for f1 in [0.1, 0.5, 1.0, 2.0]:
            for phase in [0, np.pi/2, np.pi]:
                if len(population) >= pop_size:
                    break
                s = _chirp_init(f0, f1, phase, seed=len(population))
                population.append(s)
    
    # Add VDC
    for shift in range(0, N, 50):
        if len(population) >= pop_size:
            break
        s = _vdc_init(shift)
        population.append(s)
        
    # Add Random
    while len(population) < pop_size:
        s = rng.choice([-1, 1], size=N).astype(np.int8)
        population.append(s)
        
    # 2. Initial Local Search for all members
    # Allocate small time budget per individual initially
    # Total budget for init polish: 10s
    time_per_init_polish = 10.0 / pop_size
    
    for i in range(pop_size):
        if time.time() > deadline - 20.0:
            break
        population[i], score = _local_search(population[i], time_per_init_polish, rng)
        scores.append(score)
        
        if score < best_val:
            best_val = score
            best_seq = population[i].copy()
            
    # 3. Evolutionary Loop
    # BDE parameters
    F = 0.5  # Mutation strength
    CR = 0.9  # Crossover rate (probability of taking from mutant)
    
    # We will iterate until time runs out
    # Each generation:
    # - For each individual, create a mutant
    # - Crossover with parent
    # - Local search the offspring
    # - If offspring is better, replace parent
    
    gen = 0
    while time.time() < deadline - 10.0: # Reserve 10s for final polish
        gen += 1
        # Update F adaptively? Start high, decrease?
        # Let's keep F constant for now, maybe adjust if stuck.
        
        new_population = []
        new_scores = []
        
        for i in range(pop_size):
            if time.time() > deadline - 10.0:
                # Break early if tight on time
                new_population.append(population[i])
                new_scores.append(scores[i])
                continue
                
            # Mutation
            mutant = _bde_mutation(population, i, F, rng)
            
            # Crossover
            # Binomial crossover: for each bit, take from mutant with prob CR, else from parent
            crossover_mask = rng.random(N) < CR
            offspring = population[i].copy()
            offspring[crossover_mask] = mutant[crossover_mask]
            
            # Local Search on Offspring
            # Time budget for offspring polish: 
            # We have ~120s total. Pop size 30.
            # If we do 10 generations, 300 evaluations.
            # 120s / 300 = 0.4s per eval.
            # This is very generous. We can be more aggressive.
            # Let's cap it at 0.2s per offspring to allow more generations.
            time_budget_offspring = 0.2
            offspring, off_score = _local_search(offspring, time_budget_offspring, rng)
            
            # Selection
            if off_score < scores[i]:
                new_population.append(offspring)
                new_scores.append(off_score)
                
                if off_score < best_val:
                    best_val = off_score
                    best_seq = offspring.copy()
            else:
                new_population.append(population[i])
                new_scores.append(scores[i])
                
        population = new_population
        scores = new_scores
        
        # Adaptive F: if no improvement in last 5 generations, increase F?
        # Simple heuristic: if best_val hasn't changed in a while, perturb best_seq
        # We'll just rely on the diversity of the population.

    # 4. Final Polish
    if time.time() < deadline - 1.0:
        polish_budget = deadline - 1.0 - time.time()
        # Polish the best sequence found
        best_seq, best_val = _local_search(best_seq, polish_budget, rng)
        
    return best_seq
