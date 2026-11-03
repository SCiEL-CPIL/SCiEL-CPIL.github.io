import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i
    
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
    
    # Compute coverage frequency for weighting
    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1
    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)
    
    def get_coverage(block_indices):
        """Return coverage count array for a set of block indices."""
        if len(block_indices) == 0:
            return np.zeros(n_triples, dtype=np.int32)
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        return coverage
    
    def fitness(block_indices):
        """Lower is better: number of blocks + penalty for uncovered triples."""
        if len(block_indices) == 0:
            return float(n_triples) + 1000
        coverage = get_coverage(block_indices)
        uncovered = np.sum(coverage == 0)
        # Heavily penalize uncovered triples
        return len(block_indices) + uncovered * 100
    
    def is_valid(block_indices):
        """Check if all triples are covered."""
        if len(block_indices) == 0:
            return False
        coverage = get_coverage(block_indices)
        return np.all(coverage >= 1)
    
    def greedy_init():
        """Initial solution via weighted greedy."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break
            gains = np.sum(uncovered[block_triples].astype(np.float64) * triple_weights[block_triples], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        
        return selected
    
    def remove_redundant(block_indices):
        """Remove blocks that are not needed."""
        if len(block_indices) == 0:
            return block_indices
        
        coverage = get_coverage(block_indices)
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline - 5.0:
                    return result
                bi = result[i]
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break
        return result
    
    def mutate(block_indices, num_mutations=1):
        """Replace some blocks with random alternatives that improve coverage."""
        if len(block_indices) == 0:
            return block_indices
        
        coverage = get_coverage(block_indices)
        current = list(block_indices)
        
        for _ in range(num_mutations):
            if time.time() > deadline - 5.0:
                break
            # Pick a random position to mutate
            pos = random.randint(0, len(current) - 1)
            old_bi = current[pos]
            
            # Try to find a replacement that covers at least the critical triples
            critical = block_triples[old_bi]
            critical_mask = coverage[critical] <= 1
            
            # Try random candidates
            found = False
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                new_bi = random.randint(0, n_blocks - 1)
                if new_bi == old_bi:
                    continue
                # Check if new block covers all critical triples
                new_triples = block_triples[new_bi]
                if np.all(np.isin(critical[critical_mask], new_triples)):
                    # Valid replacement
                    coverage[block_triples[old_bi]] -= 1
                    coverage[block_triples[new_bi]] += 1
                    current[pos] = new_bi
                    found = True
                    break
            
            if not found:
                # Just replace with a random block
                new_bi = random.randint(0, n_blocks - 1)
                coverage[block_triples[old_bi]] -= 1
                coverage[block_triples[new_bi]] += 1
                current[pos] = new_bi
        
        return current
    
    def crossover(parent1, parent2):
        """Crossover: take union of parents, then remove redundant blocks."""
        if len(parent1) == 0:
            return list(parent2)
        if len(parent2) == 0:
            return list(parent1)
        
        # Combine both parents
        combined = list(set(parent1) | set(parent2))
        # Remove redundant
        combined = remove_redundant(combined)
        return combined
    
    def tournament_select(population, k=3):
        """Tournament selection."""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: -fitness(population[i]))
        return population[best_idx]
    
    # Initialize population
    pop_size = 30
    population = []
    
    # First individual: greedy
    pop0 = greedy_init()
    pop0 = remove_redundant(pop0)
    population.append(pop0)
    
    # Other individuals: greedy with random tie-breaking + mutation
    for i in range(1, pop_size):
        if time.time() > deadline - 30.0:
            break
        # Weighted greedy with randomness
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break
            gains = np.sum(uncovered[block_triples].astype(np.float64) * triple_weights[block_triples], axis=1)
            # Add randomness
            gains += np.random.randn(n_blocks) * 0.1
            best_idx = np.argmax(gains)
            if gains[best_idx] <= 0:
                break
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        selected = remove_redundant(selected)
        selected = mutate(selected, num_mutations=random.randint(1, 3))
        population.append(selected)
    
    # Track best solution
    best_solution = min(population, key=fitness)
    best_size = len(best_solution)
    
    # Genetic algorithm loop
    generation = 0
    while time.time() < deadline - 10.0:
        generation += 1
        
        # Create new population
        new_population = []
        
        # Elitism: keep best few
        sorted_pop = sorted(population, key=fitness)
        elite_count = max(2, pop_size // 5)
        for i in range(elite_count):
            new_population.append(list(sorted_pop[i]))
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10.0:
                break
            
            # Selection
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            if random.random() < 0.7:
                child = mutate(child, num_mutations=random.randint(1, 2))
            
            # Remove redundant
            child = remove_redundant(child)
            
            new_population.append(child)
        
        # Update best
        for sol in new_population:
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = list(sol)
        
        population = new_population
        
        # Early stop if we reach the known best
        if best_size <= 76:
            break
    
    # Final cleanup: remove redundant from best solution
    best_solution = remove_redundant(best_solution)
    
    return [list(all_blocks[i]) for i in best_solution]
