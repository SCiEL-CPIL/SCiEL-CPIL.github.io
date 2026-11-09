"""
Snake-in-the-box in Q_9: construct longest induced path.

Approach: Genetic Algorithm (GA).
Distinct from Simulated Annealing (single trajectory, temperature-based).
GA maintains a population of snakes.
Crossover: Take prefix of Parent 1, extend using Greedy strategy biased towards vertices in Parent 2.
Mutation: Truncate and extend with high randomness.
This allows exploration of multiple basins of attraction simultaneously.
"""

import random
import time

N = 9
MAX_VERTEX = 1 << N

# EVOLVE-BLOCK-START

def _is_valid_extension(candidate, path, used):
    """Check if candidate can be appended to path to keep it induced."""
    if candidate in used:
        return False
    # Must differ from last vertex by exactly 1 bit
    if (candidate ^ path[-1]).bit_count() != 1:
        return False
    # Must differ from all other vertices by at least 2 bits
    for p in path[:-1]:
        if (candidate ^ p).bit_count() == 1:
            return False
    return True

def _get_valid_neighbors(vertex, path, used):
    """Get all valid neighbors of vertex that can be appended to path."""
    neighbors = []
    for b in range(N):
        v = vertex ^ (1 << b)
        if v in used:
            continue
        if (v ^ path[-1]).bit_count() != 1:
            continue
        # Check against all other path vertices
        valid = True
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                valid = False
                break
        if valid:
            neighbors.append(v)
    return neighbors

def _greedy_init(start=0):
    """Quick greedy initialization to get a decent starting path."""
    path = [start]
    used = {start}
    while True:
        cur = path[-1]
        candidates = _get_valid_neighbors(cur, path, used)
        if not candidates:
            break
        next_v = random.choice(candidates)
        path.append(next_v)
        used.add(next_v)
    return path

def _extend_greedy(path, used, bias_set=None, noise=0.5):
    """
    Extend path greedily.
    bias_set: Set of vertices to prefer (if valid).
    noise: Probability of picking a random valid move instead of a biased/best move.
    """
    while True:
        cur = path[-1]
        candidates = _get_valid_neighbors(cur, path, used)
        if not candidates:
            break
        
        if noise > 0 and random.random() < noise:
            next_v = random.choice(candidates)
        else:
            # Prefer candidates in bias_set
            if bias_set:
                biased = [v for v in candidates if v in bias_set]
                if biased:
                    next_v = random.choice(biased)
                else:
                    next_v = random.choice(candidates)
            else:
                next_v = random.choice(candidates)
        
        path.append(next_v)
        used.add(next_v)
    return path

def construct():
    """
    Construct the longest snake in Q_9 using Genetic Algorithm.
    Returns a list of distinct integers representing the vertices in order.
    """
    start_time = time.time()
    time_limit = 115.0
    
    random.seed(42)
    
    POP_SIZE = 30
    NOISE_CROSS = 0.4
    NOISE_MUT = 0.8
    MUT_PROB = 0.3
    
    # Initialize population
    population = []
    for _ in range(POP_SIZE):
        path = _greedy_init(0)
        population.append(path)
    
    best_path = max(population, key=len)
    best_len = len(best_path)
    
    # Main GA Loop
    while time.time() - start_time < time_limit:
        # Create new population (or update in place)
        new_population = []
        
        # Elitism: keep the best
        new_population.append(best_path)
        
        for i in range(1, POP_SIZE):
            # Selection: Tournament
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover
            # Take prefix of p1
            cut = random.randint(1, min(len(p1), len(p2), 150)) # Cap cut to ensure extension time
            child_path = list(p1[:cut])
            child_used = set(child_path)
            
            # Bias set from p2
            bias_set = set(p2)
            
            # Extend child
            child_path = _extend_greedy(child_path, child_used, bias_set, NOISE_CROSS)
            
            # Mutation
            if random.random() < MUT_PROB:
                # Truncate at random point
                if len(child_path) > 2:
                    mut_cut = random.randint(2, len(child_path) - 1)
                    child_path = child_path[:mut_cut]
                    child_used = set(child_path)
                    # Extend with high noise, no bias
                    child_path = _extend_greedy(child_path, child_used, None, NOISE_MUT)
            
            new_population.append(child_path)
            
            # Update best
            if len(child_path) > best_len:
                best_len = len(child_path)
                best_path = list(child_path)
        
        population = new_population
        
        # Optional: Re-seed worst individuals if stagnation?
        # For now, simple replacement.

    return best_path

# EVOLVE-BLOCK-END
