"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices has a common dominator.
    Primary: Paley tournament on p=67 (known valid).
    Secondary: attempts a genetic algorithm search for smaller n (31-66) within time budget.
    """
    start_time = time.time()
    deadline = 110  # Safety margin before 120s kill

    # 1. Precompute the known valid Paley tournament on 67 vertices
    # This is our guaranteed fallback and current best score.
    p_best = 67
    squares_best = [False] * p_best
    for x in range(1, p_best):
        squares_best[(x * x) % p_best] = True

    orientation_best = []
    for i in range(p_best):
        for j in range(i + 1, p_best):
            diff = (j - i) % p_best
            if squares_best[diff]:
                orientation_best.append(0)
            else:
                orientation_best.append(1)

    best_result = {'n': p_best, 'orientation': orientation_best}

    # 2. Check time remaining
    if time.time() - start_time > deadline - 5:
        return best_result

    # 3. Attempt Genetic Algorithm search for smaller n
    # We search n from 31 (lower bound) up to 66.
    # For each n, we run a short GA. If it finds a valid tournament, we return it.
    
    def build_in_neighbors(n, orientation):
        """Build in_neighbors bitsets from orientation list."""
        in_neighbors = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    # i -> j
                    in_neighbors[j] |= (1 << i)
                else:
                    # j -> i
                    in_neighbors[i] |= (1 << j)
        return in_neighbors

    def count_uncovered(in_neighbors, n):
        """Count the number of 4-sets without a common dominator."""
        uncovered = 0
        for a in range(n):
            for b in range(a + 1, n):
                ab = in_neighbors[a] & in_neighbors[b]
                for c in range(b + 1, n):
                    abc = ab & in_neighbors[c]
                    for d in range(c + 1, n):
                        if (abc & in_neighbors[d]) == 0:
                            uncovered += 1
        return uncovered

    def random_orientation(n):
        return [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]

    def flip_bits(orientation, n, num_flips):
        new_orient = orientation[:]
        m = n * (n - 1) // 2
        for _ in range(num_flips):
            idx = random.randint(0, m - 1)
            new_orient[idx] = 1 - new_orient[idx]
        return new_orient

    # Search range: 31 to 66
    # We allocate a small time budget for each n
    # Let's try to be efficient.
    
    # We'll iterate n from 31 to 66.
    # If we find a valid one, we return it immediately.
    
    # To save time, we can skip some n if time is running out.
    # Let's just loop and check time at the start of each n.
    
    for n in range(31, 67):
        if time.time() - start_time > deadline - 2:
            break
            
        # Population size
        pop_size = 100
        # Generations
        # We want to keep this fast. Let's say 50 generations.
        # But we must check time.
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            orient = random_orientation(n)
            in_neigh = build_in_neighbors(n, orient)
            fitness = count_uncovered(in_neigh, n)
            population.append((fitness, orient, in_neigh))
        
        # Sort by fitness (lower is better)
        population.sort(key=lambda x: x[0])
        
        best_fitness = population[0][0]
        if best_fitness == 0:
            return {'n': n, 'orientation': population[0][1]}
        
        # Evolution
        max_gens = 20
        for g in range(max_gens):
            if time.time() - start_time > deadline - 2:
                break
            
            new_population = []
            
            # Elitism: keep top 10%
            elite_count = max(1, pop_size // 10)
            for i in range(elite_count):
                new_population.append(population[i])
            
            # Fill rest with mutated copies
            while len(new_population) < pop_size:
                # Select parent (tournament selection or roulette)
                # Simple: pick random from top half
                parent_idx = random.randint(0, pop_size // 2)
                parent_orient = population[parent_idx][1]
                
                # Mutation
                num_flips = random.randint(1, 3)
                child_orient = flip_bits(parent_orient, n, num_flips)
                
                # Evaluate
                in_neigh = build_in_neighbors(n, child_orient)
                fitness = count_uncovered(in_neigh, n)
                
                new_population.append((fitness, child_orient, in_neigh))
                
                if fitness == 0:
                    # Found valid solution
                    return {'n': n, 'orientation': child_orient}
            
            population = new_population
            population.sort(key=lambda x: x[0])
            
            if population[0][0] == 0:
                return {'n': n, 'orientation': population[0][1]}

    # If GA didn't find anything better, return the best known (67)
    return best_result

# EVOLVE-BLOCK-END
