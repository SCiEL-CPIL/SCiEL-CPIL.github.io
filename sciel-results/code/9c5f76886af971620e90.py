import numpy as np
import random
import itertools
import time

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        # Use a simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        neighbors = [set() for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    neighbors[i].add(word_to_idx[new_word])
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = len(neighbors[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            uncovered -= neighbors[best_c]
        return [list(all_words[i]) for i in solution]

    # Specific optimization for q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words) # 2187
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as lists of indices
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                new_word = list(w)
                new_word[pos] = val
                new_word = tuple(new_word)
                neighbors[i].append(word_to_idx[new_word])
        
    # Convert neighbors to numpy arrays for faster intersection if needed, 
    # but for small N, lists are okay. 
    # However, for the GA, we need fast "is this a valid cover" check and "prune".
    
    # Helper: Check if a set of codewords covers all words
    # Returns True if valid, False otherwise
    def is_valid(code):
        if not code:
            return N == 0
        covered = np.zeros(N, dtype=bool)
        for c in code:
            for nb in neighbors[c]:
                covered[nb] = True
        return np.all(covered)
        
    # Helper: Prune redundant codewords
    def prune(code):
        if not code:
            return []
        # Use a list for code to allow removal
        code_list = list(code)
        # Calculate cover count
        cover_count = np.zeros(N, dtype=int)
        for c in code_list:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                # Check if c is redundant
                # c is redundant if all its neighbors are covered by others
                # i.e., cover_count[nb] > 1 for all nb in neighbors[c]
                # Note: cover_count includes c itself.
                redundant = True
                for nb in neighbors[c]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    code_list.pop(i)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    changed = True
                    break # Restart loop
        return code_list

    # Helper: Greedy repair for a given set of uncovered words
    def greedy_repair(uncovered_set):
        if not uncovered_set:
            return []
        solution = []
        remaining = set(uncovered_set)
        while remaining:
            best_c = -1
            best_gain = -1
            # To speed up, we can just scan all N
            # N=2187, this is fast.
            for c in range(N):
                gain = 0
                for nb in neighbors[c]:
                    if nb in remaining:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            for nb in neighbors[best_c]:
                if nb in remaining:
                    remaining.remove(nb)
        return solution

    # Initial Solution: Greedy
    current_code = greedy_repair(set(range(N)))
    current_code = prune(current_code)
    best_code = current_code
    best_size = len(best_code)
    
    start_time = time.time()
    timeout = 115.0
    
    random.seed(42)
    
    # Genetic Algorithm Parameters
    pop_size = 40
    population = [list(current_code) for _ in range(pop_size)]
    # Add some random variations to initial population
    for i in range(pop_size):
        if i > 0:
            # Perturb the first solution to create diversity
            code = list(population[0])
            k = random.randint(1, 5)
            for _ in range(k):
                if code:
                    idx = random.randint(0, len(code)-1)
                    code.pop(idx)
                    # Repair
                    # Find uncovered
                    covered = np.zeros(N, dtype=bool)
                    for c in code:
                        for nb in neighbors[c]:
                            covered[nb] = True
                    uncovered = [i for i in range(N) if not covered[i]]
                    if uncovered:
                        repair = greedy_repair(set(uncovered))
                        code.extend(repair)
                    code = prune(code)
            population[i] = code
            
    # Main Loop
    while time.time() - start_time < timeout:
        new_population = []
        
        for _ in range(pop_size):
            # Selection: Tournament of 3
            participants = random.sample(population, 3)
            parent = min(participants, key=lambda x: len(x))
            
            # Crossover: Uniform
            # Since order doesn't matter, we can just take a random subset?
            # No, uniform crossover on sequences is standard.
            # Let's do single point crossover for simplicity, then deduplicate.
            # Actually, for sets, a "set crossover" is better:
            # Child = (Parent A U Parent B) with some random drops?
            # Or just: Child = Parent A, then mutate.
            # Let's stick to a simple mutation-heavy approach (Genetic Drift)
            # or standard crossover.
            
            # Let's try: Child is a copy of parent, then apply mutations.
            child = list(parent)
            
            # Mutation
            num_mutations = random.randint(1, 3)
            for _ in range(num_mutations):
                mutation_type = random.choice(['remove', 'add', 'swap'])
                
                if mutation_type == 'remove' and child:
                    idx = random.randint(0, len(child)-1)
                    child.pop(idx)
                    # Repair if invalid
                    covered = np.zeros(N, dtype=bool)
                    for c in child:
                        for nb in neighbors[c]:
                            covered[nb] = True
                    uncovered = [i for i in range(N) if not covered[i]]
                    if uncovered:
                        repair = greedy_repair(set(uncovered))
                        child.extend(repair)
                        
                elif mutation_type == 'add':
                    c = random.randint(0, N-1)
                    child.append(c)
                    
                elif mutation_type == 'swap' and child:
                    idx = random.randint(0, len(child)-1)
                    new_c = random.randint(0, N-1)
                    if new_c not in child:
                        child[idx] = new_c
                        # Repair if invalid
                        covered = np.zeros(N, dtype=bool)
                        for c in child:
                            for nb in neighbors[c]:
                                covered[nb] = True
                        uncovered = [i for i in range(N) if not covered[i]]
                        if uncovered:
                            repair = greedy_repair(set(uncovered))
                            child.extend(repair)

            # Local Search (Prune)
            child = prune(child)
            
            # Ensure validity (prune should maintain validity if input was valid, 
            # but if we added/swapped, we might have introduced redundancy or broken coverage?
            # Prune only removes redundant. It doesn't add.
            # If the mutation broke coverage, prune won't fix it.
            # We need to check validity and repair if broken.
            
            # Check validity
            covered = np.zeros(N, dtype=bool)
            for c in child:
                for nb in neighbors[c]:
                    covered[nb] = True
            if not np.all(covered):
                uncovered = [i for i in range(N) if not covered[i]]
                repair = greedy_repair(set(uncovered))
                child.extend(repair)
                child = prune(child)
                
            new_population.append(child)
            
            # Update best
            if len(child) < best_size:
                best_size = len(child)
                best_code = list(child)
                
        # Elitism: Keep the best individual
        new_population.sort(key=lambda x: len(x))
        population = new_population
        
        # Update population with best if it's better
        if len(best_code) < len(population[0]):
            population[0] = list(best_code)

    return [list(all_words[i]) for i in best_code]
