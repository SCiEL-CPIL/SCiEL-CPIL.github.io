import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Parameters for q=3, n=6, radius=1
    total_words = q ** n  # 729
    num_neighbors = 1 + n * (q - 1)  # 1 + 6*2 = 13
    
    # Convert between word (list of ints) and integer representation
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
    
    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w
    
    # Precompute neighbor sets (closed ball of radius 1) for each word
    # neighbors[i] is a set of ints within Hamming distance <= 1 of word i
    neighbors = []
    for i in range(total_words):
        w = int_to_word(i)
        ball = {i}
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                ball.add(word_to_int(w))
            w[pos] = orig
        neighbors.append(ball)
    
    # Precompute for each word, the list of codeword indices that cover it
    # This is the reverse mapping: word -> set of codewords that cover it
    # We'll use this for efficient coverage counting
    
    # Initial solution: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    current_code = set(word_to_int(c) for c in initial_code)
    
    # Coverage count: for each word, how many codewords cover it
    # coverage_count[w] = number of codewords c such that w is in neighbors[c]
    def compute_coverage_counts(code_set):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for w in neighbors[c]:
                counts[w] += 1
        return counts
    
    def is_covering(code_set):
        """Check if code_set covers all words"""
        counts = compute_coverage_counts(code_set)
        return np.all(counts > 0)
    
    def fitness(code_set):
        """
        Fitness based on sum of squared coverage counts.
        Lower is better. A perfect covering has all counts >= 1.
        We want to minimize redundancy (high counts) while maintaining coverage.
        """
        counts = compute_coverage_counts(code_set)
        # Penalize uncovered words heavily
        uncovered = np.sum(counts == 0)
        if uncovered > 0:
            return 1e9 + uncovered * 1e6 + np.sum(counts ** 2)
        return np.sum(counts ** 2)
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Phase 1: Remove redundant codewords greedily
    def try_remove_redundant():
        removed = False
        candidates = list(current_code)
        random.shuffle(candidates)
        for c in candidates:
            if c not in current_code:
                continue
            test_set = current_code - {c}
            if is_covering(test_set):
                current_code.remove(c)
                removed = True
        return removed
    
    # Phase 2: Genetic algorithm with crossover
    def crossover(parent1, parent2):
        """
        Crossover operator that preserves Hamming ball structure.
        We combine codewords from both parents, keeping the best subset.
        """
        combined = parent1 | parent2
        # Try to reduce the combined set to a valid covering
        # Start with the smaller parent and add from the other if needed
        if len(parent1) <= len(parent2):
            base = set(parent1)
            extra = parent2 - parent1
        else:
            base = set(parent2)
            extra = parent1 - parent2
        
        # If base is already covering, try to remove from base
        if is_covering(base):
            # Try to remove redundant from base
            for c in list(base):
                if not is_covering(base - {c}):
                    break
                base.remove(c)
            return base
        
        # Otherwise, add from extra until covering
        for c in extra:
            if is_covering(base):
                break
            base.add(c)
        
        if is_covering(base):
            # Try to remove redundant
            for c in list(base):
                if not is_covering(base - {c}):
                    break
                base.remove(c)
            return base
        
        # Fallback: return the smaller parent
        return parent1 if len(parent1) <= len(parent2) else parent2
    
    def mutation(code_set):
        """
        Mutation: replace a random codeword with a random neighbor or new word.
        """
        if not code_set:
            return code_set
        c = random.choice(list(code_set))
        # Try to find a replacement that maintains coverage
        for _ in range(50):
            new_c = random.randrange(total_words)
            if new_c in code_set:
                continue
            test_set = (code_set - {c}) | {new_c}
            if is_covering(test_set):
                return test_set
        return code_set
    
    # Main optimization loop
    best_code = set(current_code)
    best_fitness = fitness(best_code)
    
    # Initial cleanup
    while try_remove_redundant():
        if time.time() - start_time > deadline:
            break
    
    best_code = set(current_code)
    best_fitness = fitness(best_code)
    
    # Genetic algorithm
    population_size = 5
    population = [set(current_code) for _ in range(population_size)]
    
    while time.time() - start_time < deadline:
        # Evaluate fitness
        fitnesses = [fitness(ind) for ind in population]
        
        # Update best
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_fitness:
            best_fitness = fitnesses[min_idx]
            best_code = set(population[min_idx])
        
        # Selection: keep the best half
        sorted_indices = np.argsort(fitnesses)
        elite_count = max(1, population_size // 2)
        new_population = [set(population[i]) for i in sorted_indices[:elite_count]]
        
        # Generate offspring
        while len(new_population) < population_size:
            if time.time() - start_time > deadline:
                break
            p1 = random.choice(new_population)
            p2 = random.choice(new_population)
            child = crossover(p1, p2)
            # With some probability, mutate
            if random.random() < 0.3:
                child = mutation(child)
            new_population.append(child)
        
        population = new_population
    
    # Final cleanup on the best code found
    final_code = set(best_code)
    while True:
        removed = False
        candidates = list(final_code)
        random.shuffle(candidates)
        for c in candidates:
            if c not in final_code:
                continue
            if is_covering(final_code - {c}):
                final_code.remove(c)
                removed = True
        if not removed or time.time() - start_time > deadline:
            break
    
    # Convert back to lists of ints
    result = [int_to_word(c) for c in final_code]
    return result
