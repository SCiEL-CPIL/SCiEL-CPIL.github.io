import numpy as np
import time
import random

N = 256
DEADLINE = 115  # seconds, safety margin before 120s limit

def _build_aps():
    """Build all 4-term APs as index arrays for vectorized counting."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return np.array(aps, dtype=np.int32)

def _count_mono_fast(aps, colors):
    """Count monochromatic 4-APs using vectorized numpy."""
    # colors is a numpy array of 0/1
    c = colors[aps]  # shape (num_aps, 4)
    # Check if all 4 elements are equal
    return np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3]))

def _local_search(aps, colors, max_iter=200, deadline=None):
    """Greedy local search: flip bits that reduce mono-AP count."""
    colors = colors.copy()
    n = len(colors)
    current_score = _count_mono_fast(aps, colors)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        best_improvement = 0
        best_pos = -1
        # Try flipping each position
        for i in range(n):
            colors[i] ^= 1
            new_score = _count_mono_fast(aps, colors)
            improvement = current_score - new_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_pos = i
            colors[i] ^= 1  # revert
        
        if best_pos >= 0:
            colors[best_pos] ^= 1
            current_score -= best_improvement
        else:
            break
    
    return colors, current_score

def _lfsr_init(seed):
    """Generate a structured binary sequence using LFSR."""
    random.seed(seed)
    # Use a simple LFSR to generate pseudo-random but structured bits
    # Start with a non-zero state
    state = random.getrandbits(16)
    if state == 0:
        state = 1
    bits = []
    for _ in range(N):
        # Tap positions for a 16-bit LFSR
        feedback = ((state >> 0) ^ (state >> 3) ^ (state >> 12) ^ (state >> 15)) & 1
        state = ((state << 1) | feedback) & 0xFFFF
        bits.append(state & 1)
    return np.array(bits, dtype=np.int8)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    aps = _build_aps()
    
    # Initialize population with diverse structured colorings
    pop_size = 40
    population = []
    
    # LFSR-based initializations with different seeds
    for i in range(pop_size // 2):
        colors = _lfsr_init(seed=12345 + i * 7919)
        population.append(colors)
    
    # Random initializations
    for i in range(pop_size // 2):
        colors = np.random.randint(0, 2, N).astype(np.int8)
        population.append(colors)
    
    # Evaluate initial population
    scores = [_count_mono_fast(aps, c) for c in population]
    
    # Local search on the best few
    best_colors = None
    best_score = float('inf')
    
    # Sort by score and refine top candidates
    indices = np.argsort(scores)
    for idx in indices[:10]:
        if time.time() > deadline - 10:
            break
        refined, refined_score = _local_search(aps, population[idx], max_iter=100, deadline=deadline - 5)
        if refined_score < best_score:
            best_score = refined_score
            best_colors = refined.copy()
    
    # Genetic algorithm phase
    if time.time() < deadline - 20:
        gen = 0
        while time.time() < deadline - 10:
            gen += 1
            # Evaluate current population
            scores = [_count_mono_fast(aps, c) for c in population]
            
            # Update best
            for i, s in enumerate(scores):
                if s < best_score:
                    best_score = s
                    best_colors = population[i].copy()
            
            # Selection: tournament selection
            new_population = []
            # Keep the best individual
            best_idx = np.argmin(scores)
            new_population.append(population[best_idx].copy())
            
            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() > deadline - 15:
                    break
                # Tournament selection
                t1 = random.randint(0, len(population) - 1)
                t2 = random.randint(0, len(population) - 1)
                parent1 = population[t1] if scores[t1] <= scores[t2] else population[t2]
                
                t1 = random.randint(0, len(population) - 1)
                t2 = random.randint(0, len(population) - 1)
                parent2 = population[t1] if scores[t1] <= scores[t2] else population[t2]
                
                # Uniform crossover
                mask = np.random.randint(0, 2, N)
                child = np.where(mask == 1, parent1, parent2).astype(np.int8)
                
                # Mutation
                mutation_rate = 0.05
                mutate_mask = np.random.random(N) < mutation_rate
                child[mutate_mask] ^= 1
                
                new_population.append(child)
            
            population = new_population[:pop_size]
            
            # Local search on a few random members
            if time.time() < deadline - 15:
                for _ in range(3):
                    if time.time() > deadline - 12:
                        break
                    idx = random.randint(0, len(population) - 1)
                    refined, refined_score = _local_search(aps, population[idx], max_iter=50, deadline=deadline - 8)
                    population[idx] = refined
                    if refined_score < best_score:
                        best_score = refined_score
                        best_colors = refined.copy()
    
    # Final local search on the best
    if time.time() < deadline - 5 and best_colors is not None:
        refined, refined_score = _local_search(aps, best_colors, max_iter=100, deadline=deadline - 2)
        if refined_score < best_score:
            best_score = refined_score
            best_colors = refined
    
    # Return as list of ints
    if best_colors is None:
        return [0] * N
    return list(best_colors.astype(int))
