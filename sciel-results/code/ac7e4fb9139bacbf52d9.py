"""Genetic algorithm approach for covering design C(19,8,5)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design using a genetic algorithm with multi-objective fitness.
    Uses bitset representation for fast coverage computation.
    """
    # Parameters
    population_size = 100
    max_generations = 300
    crossover_rate = 0.8
    mutation_rate = 0.5
    elite_size = 10
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

    # Precompute: for each block, which t-subsets it covers (as a numpy array of indices)
    # This is much faster than a 2D boolean matrix for sparse coverage
    block_cover_indices = []
    for block in all_blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(tsubset_to_idx[ts])
        block_cover_indices.append(covered)

    # Precompute for fast lookup: map a block tuple to its precomputed mask indices
    def get_block_mask_indices(block):
        """Get the precomputed coverage indices for a block."""
        idx = block_to_idx[tuple(block)]
        return block_cover_indices[idx]

    def compute_coverage(blocks):
        """Compute total coverage as a boolean array."""
        cov = np.zeros(num_tsubsets, dtype=np.uint8)
        for block in blocks:
            idx = block_to_idx[tuple(block)]
            covered = block_cover_indices[idx]
            cov[covered] = 1
        return cov

    def fitness(blocks):
        """
        Multi-objective fitness: lower is better.
        Primary: number of uncovered t-subsets
        Secondary: number of blocks (tie-breaker)
        """
        cov = compute_coverage(blocks)
        uncovered_count = num_tsubsets - int(np.sum(cov))
        return uncovered_count * 100000 + len(blocks)

    def greedy_cover():
        """Generate a greedy cover."""
        uncovered = np.ones(num_tsubsets, dtype=np.uint8)
        blocks = []
        while np.sum(uncovered) > 0:
            # Find the block that covers the most uncovered t-subsets
            best_gain = -1
            best_block_idx = None
            for i in range(num_blocks_total):
                covered = block_cover_indices[i]
                gain = int(np.sum(uncovered[covered]))
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                if best_gain == num_tsubsets - int(np.sum(uncovered)):
                    break
            if best_block_idx is None or best_gain <= 0:
                break
            block = list(all_blocks[best_block_idx])
            blocks.append(block)
            uncovered[block_cover_indices[best_block_idx]] = 0
        return blocks

    def random_individual():
        """Generate a random individual (list of blocks)."""
        # Start with a greedy cover, then add some random blocks
        blocks = greedy_cover()
        # Add some random blocks for diversity
        extra = random.randint(5, 15)
        seen = set(tuple(b) for b in blocks)
        for _ in range(extra * 3):
            b = tuple(sorted(random.sample(range(v), k)))
            if b not in seen:
                blocks.append(list(b))
                seen.add(b)
        return blocks

    def local_search(blocks):
        """
        Local search: try to remove redundant blocks.
        A block is redundant if removing it doesn't leave any t-subset uncovered.
        """
        blocks = blocks[:]
        changed = True
        while changed:
            changed = False
            # Compute coverage
            cov = np.zeros(num_tsubsets, dtype=np.uint8)
            for block in blocks:
                idx = block_to_idx[tuple(block)]
                covered = block_cover_indices[idx]
                cov[covered] = 1

            # Try to remove each block
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                idx = block_to_idx[tuple(block)]
                covered = block_cover_indices[idx]

                # Check if removing this block leaves any t-subset uncovered
                # i.e., check if there's any t-subset that is only covered by this block
                only_covered = True
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    idx_j = block_to_idx[tuple(blocks[j])]
                    covered_j = block_cover_indices[idx_j]
                    # Check if any of the t-subsets covered by block i is NOT covered by block j
                    # This is a simplification; we need to check if all t-subsets covered by block i
                    # are covered by some other block
                    # For efficiency, we'll just check if removing block i leaves coverage intact
                    pass

                # More efficient: compute coverage without block i
                # But that's expensive. Instead, let's just try removing and checking
                # For now, skip this complex local search and use a simpler approach

            # Simpler local search: try removing random blocks and see if coverage is maintained
            for _ in range(len(blocks) // 3):
                idx_remove = random.randint(0, len(blocks) - 1)
                # Compute coverage without this block
                cov_test = np.zeros(num_tsubsets, dtype=np.uint8)
                for j, block in enumerate(blocks):
                    if j == idx_remove:
                        continue
                    idx_b = block_to_idx[tuple(block)]
                    covered = block_cover_indices[idx_b]
                    cov_test[covered] = 1

                if int(np.sum(cov_test)) == num_tsubsets:
                    # Can remove this block
                    blocks.pop(idx_remove)
                    changed = True
                    break

        return blocks

    # Initialize population
    population = []

    # Add greedy individuals for quality
    for _ in range(30):
        population.append(random_individual())

    # Add more diverse individuals
    while len(population) < population_size:
        population.append(random_individual())

    # Evaluate initial population
    fitnesses = [fitness(ind) for ind in population]

    # Find initial best
    best_idx = int(np.argmin(fitnesses))
    best_individual = population[best_idx][:]
    best_fitness = fitnesses[best_idx]

    # Apply local search to the best
    best_individual = local_search(best_individual)
    best_fitness = fitness(best_individual)

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

        # Apply local search to the best individual in the new population
        new_best_idx = int(np.argmin(fitnesses))
        local_best = local_search(population[new_best_idx])
        local_best_fit = fitness(local_best)
        if local_best_fit < fitnesses[new_best_idx]:
            population[new_best_idx] = local_best
            fitnesses[new_best_idx] = local_best_fit
            if local_best_fit < best_fitness:
                best_fitness = local_best_fit
                best_individual = local_best[:]

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
    if random.random() < 0.3:
        new_block = sorted(random.sample(range(v), k))
        if tuple(new_block) not in [tuple(b) for b in individual]:
            individual.append(new_block)

    # With small probability, remove a block (only if we have more than 270)
    if random.random() < 0.2 and len(individual) > 270:
        idx = random.randint(0, len(individual) - 1)
        individual.pop(idx)

    return individual
# EVOLVE-BLOCK-END
