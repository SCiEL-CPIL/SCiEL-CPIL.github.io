import random
import time
from collections import deque

N = 12

def compute_reset_length(aut):
    """
    Compute the exact shortest reset word length using BFS on subsets.
    Returns 0 if not synchronizing.
    """
    start = (1 << N) - 1
    
    if N == 1:
        return 0

    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        current_subset = queue.popleft()
        dist = visited[current_subset]
        
        if current_subset & (current_subset - 1) == 0:
            return dist

        for l in range(2):
            new_subset = 0
            sub = current_subset
            while sub:
                lsb = sub & -sub
                i = (lsb.bit_length() - 1)
                dest = aut[l][i]
                new_subset |= (1 << dest)
                sub ^= lsb
            
            if new_subset not in visited:
                visited[new_subset] = dist + 1
                queue.append(new_subset)
                
    return 0

def construct():
    """
    Use a Genetic Algorithm to find a synchronizing automaton with a long reset word.
    """
    start_time = time.time()
    timeout = 55
    
    # Initialize population
    pop_size = 80
    population = []
    
    # Base Černý automaton
    letter_a = [(i + 1) % N for i in range(N)]
    letter_b = [0] + [i - 1 for i in range(1, N)]
    base = [letter_a, letter_b]
    
    # Add base
    population.append([list(row) for row in base])
    
    # Add mutated versions of base
    for _ in range(pop_size - 1):
        mut = [list(row) for row in base]
        # Apply a few random mutations
        for _ in range(random.randint(1, 3)):
            l = random.randint(0, 1)
            i = random.randint(0, N - 1)
            mut[l][i] = random.randint(0, N - 1)
        population.append(mut)
        
    # Evaluate initial population
    fitnesses = [compute_reset_length(ind) for ind in population]
    
    best_aut = population[fitnesses.index(max(fitnesses))]
    best_len = max(fitnesses)
    
    random.seed(42)
    
    # GA Loop
    while time.time() - start_time < timeout:
        # Selection: Tournament selection
        new_population = []
        
        # Elitism: Copy top 10%
        sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            new_population.append([list(row) for row in population[sorted_indices[i]]])
            
        # Fill rest of population
        while len(new_population) < pop_size:
            # Pick two parents
            parent1_idx = tournament_select(population, fitnesses, k=3)
            parent2_idx = tournament_select(population, fitnesses, k=3)
            
            p1 = population[parent1_idx]
            p2 = population[parent2_idx]
            
            # Crossover: Uniform crossover on flattened list
            flat1 = [x for row in p1 for x in row]
            flat2 = [x for row in p2 for x in row]
            
            flat_child = []
            for i in range(len(flat1)):
                if random.random() < 0.5:
                    flat_child.append(flat1[i])
                else:
                    flat_child.append(flat2[i])
                    
            # Unflatten
            child = [flat_child[:N], flat_child[N:]]
            
            # Mutation
            for l in range(2):
                for i in range(N):
                    if random.random() < 0.05:
                        child[l][i] = random.randint(0, N-1)
                        
            new_population.append(child)
            
        population = new_population
        fitnesses = [compute_reset_length(ind) for ind in population]
        
        current_best = max(fitnesses)
        if current_best > best_len:
            best_len = current_best
            best_aut = [list(row) for row in population[fitnesses.index(current_best)]]
            
    return best_aut

def tournament_select(population, fitnesses, k=3):
    """
    Select an individual using tournament selection.
    """
    indices = random.sample(range(len(population)), k)
    best_idx = indices[0]
    best_fit = fitnesses[best_idx]
    for idx in indices[1:]:
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best_idx = idx
    return best_idx
