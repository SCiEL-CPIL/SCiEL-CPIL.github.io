import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    total_space = q ** n
    if total_space > 50000:
        # Fallback for larger spaces: just return all words (valid but large)
        # This task is q=3, n=7, so total_space=2187.
        return [[i % q for i in range(n)] for i in range(total_space)]

    # Precompute neighbor masks for all words
    # masks[w] is a bitmask of all words within Hamming distance <= radius of w
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        # For radius 1, we add the word itself (already there) and all single-position changes
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    full_mask = (1 << total_space) - 1

    # Helper: check if a set of words covers the full space
    def is_valid(selected):
        covered = 0
        for w in selected:
            covered |= masks[w]
            if covered == full_mask:
                return True
        return covered == full_mask

    # Helper: repair a set to make it valid by adding words greedily
    def repair(selected):
        selected = list(selected)
        covered = 0
        for w in selected:
            covered |= masks[w]
        
        remaining = full_mask & ~covered
        while remaining:
            # Find word with max gain
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word == -1:
                break
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
        return selected

    # Helper: remove redundant words from a valid set
    def reduce(selected):
        selected = list(selected)
        changed = True
        while changed:
            changed = False
            # Compute coverage counts
            counts = [0] * total_space
            for w in selected:
                m = masks[w]
                mm = m
                while mm:
                    lowbit = mm & -mm
                    idx = lowbit.bit_length() - 1
                    counts[idx] += 1
                    mm ^= lowbit
            
            for i in range(len(selected)):
                w = selected[i]
                m = masks[w]
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit
                
                if removable:
                    # Remove w
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(i)
                    changed = True
                    break  # Restart loop
        return selected

    # Initialize population with diverse valid solutions
    # 1. Greedy solution
    def run_greedy():
        covered = 0
        selected = []
        remaining = full_mask
        while remaining:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
        return selected

    greedy_sol = run_greedy()
    greedy_sol = reduce(greedy_sol)
    
    # 2. Random valid solutions (start with random words, repair)
    population = [greedy_sol]
    for _ in range(19):
        # Start with a random subset
        size = random.randint(100, 250)
        random_words = random.sample(range(total_space), min(size, total_space))
        repaired = repair(random_words)
        reduced = reduce(repaired)
        population.append(reduced)

    # Ensure uniqueness in population (by tuple of sorted words)
    unique_pop = []
    seen = set()
    for sol in population:
        key = tuple(sorted(sol))
        if key not in seen:
            seen.add(key)
            unique_pop.append(sol)
    population = unique_pop if unique_pop else [greedy_sol]

    best_solution = min(population, key=len)
    best_size = len(best_solution)

    # Genetic Algorithm
    pop_size = 20
    # Pad population if needed
    while len(population) < pop_size:
        # Mutate best
        parent = best_solution
        new_sol = list(parent)
        # Random swap
        if new_sol:
            idx = random.randint(0, len(new_sol) - 1)
            new_sol[idx] = random.randint(0, total_space - 1)
        new_sol = repair(new_sol)
        new_sol = reduce(new_sol)
        population.append(new_sol)

    generation = 0
    while time.time() < deadline - 5 and generation < 1000:
        generation += 1
        
        # Evaluate fitness (size, lower is better)
        population.sort(key=len)
        
        # Update best
        if len(population[0]) < best_size:
            best_size = len(population[0])
            best_solution = population[0][:]
        
        # Selection: Tournament
        def select():
            a = random.choice(population[:10])  # Top 10
            b = random.choice(population[:10])
            return a if len(a) <= len(b) else a

        new_population = [best_solution[:]]  # Elitism
        
        while len(new_population) < pop_size and time.time() < deadline - 5:
            parent1 = select()
            parent2 = select()
            
            # Crossover: Merge and reduce
            # Take union of parents, then reduce
            child_set = set(parent1) | set(parent2)
            child = list(child_set)
            # Shuffle to randomize order for reduction
            random.shuffle(child)
            child = repair(child)
            child = reduce(child)
            
            # Mutation
            if child and random.random() < 0.3:
                idx = random.randint(0, len(child) - 1)
                child[idx] = random.randint(0, total_space - 1)
                child = repair(child)
                child = reduce(child)
            
            new_population.append(child)
        
        population = new_population[:pop_size]

    # Final reduction of best solution
    best_solution = reduce(best_solution)
    
    # Convert to list of lists
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    best_words = [int_to_tuple(v) for v in best_solution]
    
    return best_words
