import numpy as np
import time

N = 400
KMAX = (N - 1) // 2
DEADLINE = 115.0  # Safety margin

def _compute_c3(h):
    """Compute all C3[k] for k=0..KMAX using FFT."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N + 1
    H = np.fft.rfft(h, n=size)
    HH = H * H
    C3 = np.fft.irfft(HH * H, n=size)[:N]
    return C3[:KMAX + 1]

def _score(h):
    """Max absolute value of C3."""
    C3 = _compute_c3(h)
    return float(np.max(np.abs(C3)))

def _c3_lag(h, k):
    """Compute single lag C3[k]."""
    m = N - 2 * k
    return np.dot(h[:m], h[k:k + m] * h[2 * k:2 * k + m])

def _local_search(h, rng, deadline):
    """
    Hill Climbing with single-bit flips.
    Returns best h and score.
    """
    C3 = _compute_c3(h)
    best_score = float(np.max(np.abs(C3)))
    cur_score = best_score
    best_h = h.copy()
    
    # Pre-compute which lags are affected by which bit?
    # For N=400, KMAX=199.
    # Bit j affects lag k if j is in {i, i+k, i+2k} for some i.
    # This is complex to precompute statically for all k.
    # Instead, for each flip, we identify affected k's.
    
    while time.time() < deadline:
        # Pick a random bit to flip
        j = int(rng.integers(0, N))
        old_val = h[j]
        h[j] = -old_val
        
        # Determine affected lags
        # j is involved in C3[k] if:
        # 1. j = i => k <= (N-1-j)/2
        # 2. j = i+k => i = j-k >= 0 => k <= j, and j+2k < N => k < (N-j)/2
        # 3. j = i+2k => i = j-2k >= 0 => k <= j/2, and j < N (always true)
        
        # Max k such that j is in range for lag k:
        # Case 1: k <= (N-1-j)//2
        # Case 2: k <= min(j, (N-1-j)//2) ... wait, j+2k < N => 2k < N-j => k <= (N-1-j)//2. And i=j-k>=0 => k<=j.
        # Case 3: k <= j//2.
        
        # Union of these ranges is k in [0, KMAX] such that any condition holds.
        # Simplest: just check all k? No, 199 lags.
        # Let's just recompute all lags? 199 dots of length ~400. 199*400 = 80k ops.
        # This is very fast in numpy.
        
        new_C3 = _compute_c3(h)
        new_score = float(np.max(np.abs(new_C3)))
        
        if new_score < cur_score:
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
            C3 = new_C3
        else:
            # Revert
            h[j] = old_val
            # C3 is unchanged, no need to recompute
            
    return best_h, best_score

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = np.random.default_rng(42)
    
    # Phase 1: Genetic Algorithm
    pop_size = 200
    elite_count = 20
    crossover_rate = 0.8
    mutation_rate = 0.05
    
    # Initialize Population
    population = []
    scores = []
    
    # Add some diverse initial individuals
    # 1. Random
    for _ in range(pop_size // 2):
        h = np.where(rng.random(N) < 0.5, 1.0, -1.0)
        population.append(h)
        
    # 2. Sine waves
    for f in range(1, 10):
        x = np.linspace(0, 1, N, endpoint=False)
        h = np.sign(np.sin(2 * np.pi * f * x))
        if np.sum(h) == 0: h[0] *= -1
        population.append(h)
        
    # 3. Chaotic/Structured
    h = np.ones(N)
    h[::2] = -1.0
    population.append(h)
    
    # Fill rest with random
    while len(population) < pop_size:
        h = np.where(rng.random(N) < 0.5, 1.0, -1.0)
        population.append(h)
        
    # Evaluate Initial Population
    for h in population:
        scores.append(_score(h))
        
    scores = np.array(scores)
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_h = population[best_idx].copy()
    
    generation = 0
    # Run GA until 80% of time is used
    ga_deadline = t0 + 0.8 * DEADLINE
    
    while time.time() < ga_deadline:
        # Sort by score
        indices = np.argsort(scores)
        population = [population[i] for i in indices]
        scores = scores[indices]
        
        # Elitism
        new_population = [population[i].copy() for i in range(elite_count)]
        new_scores = list(scores[:elite_count])
        
        # Generate Offspring
        while len(new_population) < pop_size:
            # Selection: Tournament of 3
            parents = []
            for _ in range(2):
                tourney = rng.integers(0, pop_size, size=3)
                p_idx = tourney[np.argmin(scores[tourney])]
                parents.append(population[p_idx])
            
            # Crossover
            if rng.random() < crossover_rate:
                point = rng.integers(1, N)
                child = np.concatenate([parents[0][:point], parents[1][point:]])
            else:
                child = parents[0].copy()
                
            # Mutation
            if rng.random() < mutation_rate:
                num_flips = int(N * mutation_rate)
                if num_flips > 0:
                    flip_indices = rng.choice(N, size=num_flips, replace=False)
                    child[flip_indices] *= -1.0
            
            new_population.append(child)
            
        # Evaluate New Population
        for i in range(len(new_population)):
            new_scores.append(_score(new_population[i]))
            
        population = new_population
        scores = np.array(new_scores)
        
        # Update Global Best
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_h = population[min_idx].copy()
            
        generation += 1
        
    # Phase 2: Local Search Refinement
    # Use remaining time
    if time.time() < deadline:
        # Run local search on the best GA solution
        bh, bs = _local_search(best_h, rng, deadline)
        if bs < best_score:
            best_score = bs
            best_h = bh.copy()
            
        # Also try local search on a few other good GA solutions if time permits
        # (Optional, but might find a better basin)
        # For now, stick to the best one to maximize refinement time.
        
    # Final check: ensure sum is not 0?
    # The problem doesn't strictly forbid sum=0, but it's a valid state.
    # However, some definitions of "signed step function" might imply non-zero mean?
    # The prompt says "signed step function".
    # If sum=0, C3[0]=0. This is good.
    # We will return the best_h as is.
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print("Constructed (first 20):", h[:20])
    print("Score:", _score(h))
