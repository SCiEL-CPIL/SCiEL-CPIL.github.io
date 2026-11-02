"""Genetic algorithm approach for covering design C(19,8,5)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design using a genetic algorithm with multi-objective fitness.
    Fitness penalizes both the number of blocks and the number of uncovered t-subsets.
    Uses bitset representation for fast coverage computation.
    """
    # Parameters
    population_size = 150
    max_generations = 200
    crossover_rate = 0.7
    mutation_rate = 0.3
    elite_size = 15
    tournament_size = 3
    time_limit = 105  # seconds, leaving buffer for evaluation
    
    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute coverage bitmask for every possible k-subset
    # C(19,8) = 75582 possible blocks, each covers C(8,5) = 56 t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    num_blocks_total = len(all_blocks)
    
    # Precompute: for each block, which t-subsets it covers (as a set of indices)
    block_cover_masks = np.zeros((num_blocks_total, num_tsubsets), dtype=np.uint8)
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            block_cover_masks[i, idx] = 1
    
    # For fast lookup: map a block tuple to its precomputed mask
    def get_block_mask(block):
        """Get the precomputed coverage mask for a block."""
        idx = block_to_idx[tuple(block)]
        return block_cover_masks[idx]
    
    def compute_coverage(blocks):
        """Compute total coverage as a boolean array."""
        cov = np.zeros(num_tsubsets, dtype=np.uint8)
        for block in blocks:
            cov |= get_block_mask(block)
        return cov
    
    def fitness(blocks):
        """
        Multi-objective fitness: lower is better.
        Primary: number of uncovered t-subsets
        Secondary: number of blocks (tie-breaker)
        """
        cov = compute_coverage(blocks)
        uncovered_count = num_tsubsets - int(np.sum(cov))
        return uncovered_count * 10000 + len(blocks)
    
    def random_block():
        """Generate a random k-subset of range(v)."""
        return sorted(random.sample(range(v), k))
    
    def greedy_individual():
        """Generate an individual using greedy covering."""
        uncovered = set(all_tsubsets)
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset for diversity
            target = random.choice(list(uncovered))
            # Build a block containing this target
            block_set = set(target)
            remaining_points = [x for x in range(v) if x not in block_set]
            
            # Greedily add points that maximize new coverage
            while len(block_set) < k:
                best_point = None
                best_gain = -1
                for x in remaining_points:
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    for ts in itertools.combinations(block_set, t - 1):
                        new_ts = tuple(sorted(ts + (x,)))
                        if new_ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None or best_gain <= 0:
                    # If no point helps, just add a random one
                    if remaining_points:
                        best_point = random.choice(remaining_points)
                    else:
                        break
                block_set.add(best_point)
                remaining_points = [x for x in remaining_points if x not in block_set]
            
            block = sorted(block_set)
            blocks.append(block)
            # Update uncovered
            for ts in itertools.combinations(block, t):
                uncovered.discard(tuple(ts))
        
        return blocks
    
    def random_individual():
        """Generate a random individual (list of blocks)."""
        num_blocks = random.randint(290, 380)
        blocks = set()
        attempts = 0
        while len(blocks) < num_blocks and attempts < num_blocks * 5:
            b = tuple(random_block())
            blocks.add(b)
            attempts += 1
        return [list(b) for b in blocks]
    
    def hybrid_individual():
        """Start with greedy, then add some random blocks for diversity."""
        blocks = greedy_individual()
        # Add a few random blocks
        extra = random.randint(0, 10)
        seen = set(tuple(b) for b in blocks)
        for _ in range(extra * 3):
            b = tuple(random_block())
            if b not in seen:
                blocks.append(list(b))
                seen.add(b)
                if len(blocks) - len(greedy_individual()) >= extra:
                    break
        return blocks
    
    # Initialize population
    population = []
    
    # Add greedy individuals for quality
    for _ in range(40):
        population.append(greedy_individual())
    
    # Add hybrid individuals
    for _ in range(30):
        population.append(hybrid_individual())
    
    # Add random individuals for diversity
    while len(population) < population_size:
        population.append(random_individual())
    
    # Evaluate initial population
    fitnesses = [fitness(ind) for ind in population]
    
    # Find initial best
    best_idx = int(np.argmin(fitnesses))
    best_individual = population[best_idx][:]
    best_fitness = fitnesses[best_idx]
    
    start_time = time.time()
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
        
        # Sort population by fitness
        indexed_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        
        # Update best
        if indexed_pop[0][0] < best_fitness:
            best_fitness = indexed_pop[0][0]
            best_individual = indexed_pop[0][1][:]
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individuals
        for i in range(elite_size):
            new_population.append(indexed_pop[i][1][:])
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parents = []
            for _ in range(2):
                candidates = random.sample(range(len(indexed_pop)), min(tournament_size, len(indexed_pop)))
                parent_idx = min(candidates, key=lambda i: indexed_pop[i][0])
                parents.append(indexed_pop[parent_idx][1])
            
            # Crossover
            if random.random() < crossover_rate:
                child = crossover(parents[0], parents[1])
            else:
                child = parents[0][:]
            
            # Mutation
            if random.random() < mutation_rate:
                child = mutate(child, v, k)
            
            new_population.append(child)
        
        # Evaluate new population
        fitnesses = [fitness(ind) for ind in new_population]
        population = new_population
    
    return best_individual


def crossover(parent1, parent2):
    """
    Crossover operator: combine blocks from two parents.
    Use a biased approach: start with the better parent's blocks, 
    then optionally replace some with blocks from the other parent.
    """
    # Ensure parent1 is the "better" one (shorter or equal length)
    if len(parent1) > len(parent2):
        parent1, parent2 = parent2, parent1
    
    # Start with parent1's blocks
    child = parent1[:]
    
    # With some probability, replace a few blocks with ones from parent2
    num_replacements = random.randint(0, max(1, len(child) // 10))
    for _ in range(num_replacements):
        if len(parent2) > 0:
            idx = random.randint(0, len(child) - 1)
            replacement = random.choice(parent2)
            # Only replace if it's a different block
            if tuple(replacement) != tuple(child[idx]):
                child[idx] = replacement[:]
    
    return child


def mutate(individual, v, k):
    """
    Mutation operator: replace some blocks with new random blocks,
    and possibly add or remove blocks.
    """
    n = len(individual)
    
    # Replace some blocks
    num_replacements = max(1, n // 15)
    for _ in range(num_replacements):
        idx = random.randint(0, n - 1)
        new_block = sorted(random.sample(range(v), k))
        individual[idx] = new_block
    
    # With small probability, add a block
    if random.random() < 0.2:
        new_block = sorted(random.sample(range(v), k))
        if tuple(new_block) not in [tuple(b) for b in individual]:
            individual.append(new_block)
    
    # With small probability, remove a block (only if we have more than 270)
    if random.random() < 0.1 and len(individual) > 270:
        idx = random.randint(0, len(individual) - 1)
        individual.pop(idx)
    
    return individual
# EVOLVE-BLOCK-END
