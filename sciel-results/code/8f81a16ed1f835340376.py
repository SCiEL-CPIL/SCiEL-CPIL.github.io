"""Genetic Algorithm approach for signed step functions minimizing max |autoconv| / (sum)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute full linear autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2 for k != 0."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    if len(c) > 1:
        max_sidelobe = np.max(np.abs(c[1:]))
    else:
        max_sidelobe = 0.0
    return max_sidelobe / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # 1. Initialize Population
    pop_size = 50
    # Mix of random and biased initializations
    rng = np.random.default_rng(123)
    
    population = []
    # Add some structured seeds
    population.append(np.ones(N, dtype=np.int8))
    population.append(np.ones(N, dtype=np.int8) * -1)
    # Alternating
    alt = np.ones(N, dtype=np.int8)
    alt[1::2] = -1
    population.append(alt)
    
    # Random biased towards +1
    for _ in range(pop_size - 3):
        x = np.random.choice([1, -1], size=N, p=[0.7, 0.3]).astype(np.int8)
        population.append(x)
    
    # Evaluate initial population
    scores = [_eval_ratio(x) for x in population]
    
    # 2. Evolution Loop
    elite_size = 5
    mutation_rate = 0.05
    crossover_prob = 0.8
    
    gen = 0
    while time.time() < deadline:
        # Sort by score (ascending, lower is better)
        sorted_indices = np.argsort(scores)
        sorted_pop = [population[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # Elitism: keep best elite_size
        next_pop = [x.copy() for x in sorted_pop[:elite_size]]
        next_scores = sorted_scores[:elite_size]
        
        # Fill rest of population
        while len(next_pop) < pop_size:
            # Tournament selection
            def tournament_select():
                idxs = rng.integers(0, pop_size, size=3)
                best_idx = idxs[np.argmin([scores[i] for i in idxs])]
                return population[best_idx]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            if rng.random() < crossover_prob:
                # Uniform crossover
                mask = rng.random(N) < 0.5
                child = np.where(mask, parent1, parent2).astype(np.int8)
            else:
                child = parent1.copy()
            
            # Mutation
            if rng.random() < mutation_rate:
                # Flip random bits
                n_flips = int(rng.random() * 5) + 1
                flip_idx = rng.choice(N, size=n_flips, replace=False)
                child[flip_idx] *= -1
            
            next_pop.append(child)
            next_scores.append(_eval_ratio(child))
            
            # Early exit if we've filled the population
            if len(next_pop) >= pop_size:
                break
        
        population = next_pop
        scores = next_scores
        gen += 1
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    # 3. Final Local Search on the best candidate
    best_idx = np.argmin(scores)
    best_x = population[best_idx].copy()
    best_val = scores[best_idx]
    
    # Perform a few rounds of local search (flip single bits) to polish
    s = best_x.copy()
    v = _eval_ratio(s)
    
    # Limit local search time
    local_deadline = start_time + 135
    
    for _round in range(3):
        if time.time() > local_deadline:
            break
        improved = False
        ac = _autoconv(s)
        peak = np.max(np.abs(ac[1:]))
        s_sum = s.sum()
        denom = s_sum * s_sum
        
        for i in range(N):
            if time.time() > local_deadline:
                break
            # Try flipping i
            old_val = s[i]
            s[i] = -s[i]
            
            # Efficient update of autoconv is complex, recompute for simplicity 
            # but N=400 is small enough for a few hundred FFTs
            ac2 = _autoconv(s)
            peak2 = np.max(np.abs(ac2[1:]))
            s_sum2 = s.sum()
            denom2 = s_sum2 * s_sum2
            v2 = peak2 / denom2 if denom2 != 0 else peak2
            
            if v2 < v - 1e-12:
                v = v2
                improved = True
            else:
                s[i] = old_val
                
        if not improved:
            break
            
    if v < best_val:
        best_val = v
        best_x = s.copy()
        
    return best_x
# EVOLVE-BLOCK-END
