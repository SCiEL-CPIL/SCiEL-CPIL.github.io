import numpy as np
import time
import random

N = 18
DEADLINE = 85.0  # safety margin before 90s kill

def _apply_comparator_bitparallel(state, i, j):
    """Apply comparator (i,j) to bit-parallel state.
    state shape: (n_inputs, N) uint8
    Writes min to wire i, max to wire j.
    """
    a = state[:, i]
    b = state[:, j]
    # min: a & b (for 0/1 values, min = AND)
    # max: a | b (for 0/1 values, max = OR)
    state[:, i] = a & b
    state[:, j] = a | b
    return state

def _simulate_network(comparators, n_inputs):
    """Simulate a network on all 2^N binary inputs using bit-parallel.
    Returns True if network sorts all inputs.
    """
    # Generate all 2^N binary inputs as (n_inputs, N) array
    # n_inputs should be 2^N = 262144
    # Use uint8 for memory efficiency
    # Generate all binary vectors
    bits = np.arange(n_inputs, dtype=np.uint32)
    # Extract bits for each wire position
    # wire k gets bit k of each number
    state = np.zeros((n_inputs, N), dtype=np.uint8)
    for k in range(N):
        state[:, k] = ((bits >> k) & 1).astype(np.uint8)
    
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = a & b
        state[:, j] = a | b
    
    # Check if sorted: for each row, values should be non-decreasing
    # i.e., state[:, k] <= state[:, k+1] for all k
    # For 0/1: this means no 1 before a 0, i.e., all 0s then all 1s
    # Equivalent: for each row, once we see a 0 after a 1, it's unsorted
    # Faster: check that for each wire k, if state[:,k]==0 then state[:,k+1:]==0
    # Even faster: the sorted form of a binary vector with m ones is [0,...,0,1,...,1]
    # So check: for each row, the number of 1s should equal the value at the last position
    # where 1s start... Actually simplest: check adjacent pairs
    # A row is sorted iff for all k < N-1: state[k] <= state[k+1]
    # i.e., not (state[k]==1 and state[k+1]==0)
    
    # Check all adjacent pairs at once
    # For each adjacent pair (k, k+1), check if state[:,k] > state[:,k+1]
    # That is state[:,k]==1 and state[:,k+1]==0
    unsorted_mask = np.zeros(n_inputs, dtype=bool)
    for k in range(N - 1):
        unsorted_mask |= (state[:, k] == 1) & (state[:, k+1] == 0)
    
    return not unsorted_mask.any()

def _count_unsorted(comparators, n_inputs=None):
    """Count number of unsorted 0-1 inputs after applying the network.
    Returns (is_sorted, count_unsorted).
    """
    if n_inputs is None:
        n_inputs = 1 << N
    
    bits = np.arange(n_inputs, dtype=np.uint32)
    state = np.zeros((n_inputs, N), dtype=np.uint8)
    for k in range(N):
        state[:, k] = ((bits >> k) & 1).astype(np.uint8)
    
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = a & b
        state[:, j] = a | b
    
    unsorted_mask = np.zeros(n_inputs, dtype=bool)
    for k in range(N - 1):
        unsorted_mask |= (state[:, k] == 1) & (state[:, k+1] == 0)
    
    cnt = unsorted_mask.sum()
    return (cnt == 0), int(cnt)

def _generate_random_network(num_comparators):
    """Generate a random sorting network with given number of comparators."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def _odd_even_transposition_network():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _crossover(parent1, parent2):
    """Uniform crossover between two networks."""
    child = []
    for (a1, b1), (a2, b2) in zip(parent1, parent2):
        if random.random() < 0.5:
            child.append((a1, b1))
        else:
            child.append((a2, b2))
    return child

def _crossover_length(parent1, parent2):
    """Crossover that takes a prefix from one and suffix from other."""
    if not parent1 or not parent2:
        return parent1 if parent1 else parent2
    cut = random.randint(0, min(len(parent1), len(parent2)))
    child = parent1[:cut] + parent2[cut:]
    return child

def _mutate_network(network, mutation_rate=0.1):
    """Apply mutations to a network."""
    network = list(network)
    # Mutation 1: change a comparator
    for idx in range(len(network)):
        if random.random() < mutation_rate:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            network[idx] = (i, j)
    # Mutation 2: insert a comparator
    if random.random() < mutation_rate * 0.5 and len(network) < 4096:
        pos = random.randint(0, len(network))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        network.insert(pos, (i, j))
    # Mutation 3: delete a comparator
    if random.random() < mutation_rate * 0.5 and len(network) > 1:
        pos = random.randint(0, len(network) - 1)
        network.pop(pos)
    return network

def _fitness(network):
    """Fitness: negative number of unsorted inputs (lower is better).
    Returns (is_sorted, unsorted_count).
    """
    try:
        is_sorted, cnt = _count_unsorted(network)
        return is_sorted, cnt
    except Exception:
        return False, 1 << N

def construct():
    start_time = time.time()
    
    # Start with the seed network
    seed_network = _odd_even_transposition_network()
    is_sorted, cnt = _count_unsorted(seed_network)
    best_network = seed_network
    best_count = len(seed_network)
    best_unsorted = cnt
    
    if is_sorted:
        # We have a valid network. Try to improve via GA.
        pass
    
    population = []
    # Initialize population with variations
    pop_size = 200
    for i in range(pop_size):
        if i == 0:
            population.append(list(seed_network))
        elif i == 1:
            # Slightly mutated seed
            population.append(_mutate_network(seed_network, 0.05))
        else:
            # Random networks of varying sizes
            size = random.randint(80, 160)
            population.append(_generate_random_network(size))
    
    # Evaluate initial population
    fitnesses = []
    for net in population:
        is_s, c = _count_unsorted(net)
        fitnesses.append((c, len(net)))
    
    # Sort by (unsorted_count, length) - we want low unsorted, then low length
    indexed_fitness = list(zip(range(len(population)), fitnesses, population))
    indexed_fitness.sort(key=lambda x: (x[1][0], x[1][1]))
    
    # Track best
    for idx, (unsorted_c, length), net in indexed_fitness:
        if unsorted_c < best_unsorted or (unsorted_c == best_unsorted and length < best_count):
            best_unsorted = unsorted_c
            best_count = length
            best_network = list(net)
    
    # Genetic algorithm loop
    gen = 0
    while time.time() - start_time < DEADLINE:
        gen += 1
        # Select parents (tournament selection)
        new_population = []
        
        # Elitism: keep best few
        for i in range(min(10, len(indexed_fitness))):
            new_population.append(list(indexed_fitness[i][2]))
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = random.randint(0, len(indexed_fitness) - 1)
            p2_idx = random.randint(0, len(indexed_fitness) - 1)
            p1 = indexed_fitness[p1_idx][2]
            p2 = indexed_fitness[p2_idx][2]
            
            # Crossover
            if random.random() < 0.5:
                child = _crossover(p1, p2)
            else:
                child = _crossover_length(p1, p2)
            
            # Mutation
            child = _mutate_network(child, 0.1)
            
            new_population.append(child)
        
        # Evaluate new population
        fitnesses = []
        for net in new_population:
            is_s, c = _count_unsorted(net)
            fitnesses.append((c, len(net)))
        
        indexed_fitness = list(zip(range(len(new_population)), fitnesses, new_population))
        indexed_fitness.sort(key=lambda x: (x[1][0], x[1][1]))
        
        # Update best
        for idx, (unsorted_c, length), net in indexed_fitness:
            if unsorted_c < best_unsorted or (unsorted_c == best_unsorted and length < best_count):
                best_unsorted = unsorted_c
                best_count = length
                best_network = list(net)
        
        # If we found a fully sorted network, try to reduce its size
        if best_unsorted == 0:
            # Try local search to reduce comparator count
            # Try removing each comparator and see if still sorted
            improved = True
            while improved and time.time() - start_time < DEADLINE:
                improved = False
                for idx in range(len(best_network)):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    trial = best_network[:idx] + best_network[idx+1:]
                    is_s, c = _count_unsorted(trial)
                    if is_s:
                        best_network = trial
                        best_count = len(trial)
                        improved = True
                        break
    
    # Time check: make sure we return a valid network
    if time.time() - start_time > DEADLINE - 2:
        # Return whatever we have
        pass
    
    # Final verification of best network
    is_s, c = _count_unsorted(best_network)
    if not is_s:
        # Fallback to seed
        return seed_network
    
    return best_network
