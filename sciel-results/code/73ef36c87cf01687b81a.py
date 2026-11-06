"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

def _compute_reset_length_bitmask(a, b):
    """BFS on subset automaton using bitmasks for speed."""
    full_mask = (1 << N) - 1
    
    # Precompute transition tables for each letter
    # trans_a[mask] = image of mask under a
    # trans_b[mask] = image of mask under b
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        sa = 0
        sb = 0
        for i in range(N):
            if mask & (1 << i):
                sa |= (1 << a[i])
                sb |= (1 << b[i])
        trans_a[mask] = sa
        trans_b[mask] = sb
    
    # BFS from full subset
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        mask = queue.popleft()
        dist = visited[mask]
        
        if mask & (mask - 1) == 0:  # singleton
            return dist
        
        for new_mask in (trans_a[mask], trans_b[mask]):
            if new_mask not in visited:
                visited[new_mask] = dist + 1
                queue.append(new_mask)
    
    return -1  # not synchronizing

def _is_synchronizing(a, b):
    """Quick check if automaton synchronizes using BFS."""
    full_mask = (1 << N) - 1
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        sa = 0
        sb = 0
        for i in range(N):
            if mask & (1 << i):
                sa |= (1 << a[i])
                sb |= (1 << b[i])
        trans_a[mask] = sa
        trans_b[mask] = sb
    
    visited = {full_mask}
    queue = deque([full_mask])
    
    while queue:
        mask = queue.popleft()
        
        if mask & (mask - 1) == 0:
            return True
        
        for new_mask in (trans_a[mask], trans_b[mask]):
            if new_mask not in visited:
                visited.add(new_mask)
                queue.append(new_mask)
    
    return False

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # safety margin
    
    # Start with a known good construction
    a0 = [(i + 1) % N for i in range(N)]
    b0 = list(range(N // 2)) + [i - N // 2 for i in range(N // 2, N)]
    
    best_a, best_b = a0, b0
    best_score = _compute_reset_length_bitmask(a0, b0)
    
    if best_score <= 0:
        # Fallback
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    # Genetic algorithm
    population_size = 50
    mutation_rate = 0.15
    crossover_rate = 0.7
    
    # Initialize population with variations
    population = [(a0[:], b0[:])]
    for _ in range(population_size - 1):
        a = a0[:]
        b = b0[:]
        for i in range(N):
            if random.random() < mutation_rate:
                a[i] = random.randint(0, N - 1)
            if random.random() < mutation_rate:
                b[i] = random.randint(0, N - 1)
        population.append((a, b))
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Evaluate population
        scored = []
        for a, b in population:
            score = _compute_reset_length_bitmask(a, b)
            if score > 0:
                scored.append((score, a, b))
        
        if not scored:
            break
        
        # Sort by score descending
        scored.sort(key=lambda x: -x[0])
        
        # Update best
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_a, best_b = scored[0][1][:], scored[0][2][:]
        
        # Elitism: keep top individuals
        elite_count = max(5, population_size // 5)
        new_population = [(a[:], b[:]) for _, a, b in scored[:elite_count]]
        
        # Generate offspring
        while len(new_population) < population_size:
            if random.random() < crossover_rate and len(scored) >= 2:
                # Crossover
                _, a1, b1 = scored[random.randint(0, min(10, len(scored)-1))]
                _, a2, b2 = scored[random.randint(0, min(10, len(scored)-1))]
                a_new = a1[:]
                b_new = b1[:]
                for i in range(N):
                    if random.random() < 0.5:
                        a_new[i] = a2[i]
                        b_new[i] = b2[i]
            else:
                # Clone from elite
                _, a1, b1 = scored[random.randint(0, min(5, len(scored)-1))]
                a_new = a1[:]
                b_new = b1[:]
            
            # Mutation
            for i in range(N):
                if random.random() < mutation_rate:
                    a_new[i] = random.randint(0, N - 1)
                if random.random() < mutation_rate:
                    b_new[i] = random.randint(0, N - 1)
            
            new_population.append((a_new, b_new))
        
        population = new_population
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
