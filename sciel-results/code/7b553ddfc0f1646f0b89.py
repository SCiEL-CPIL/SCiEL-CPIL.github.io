"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 16
DEADLINE = time.time() + 80  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _apply_comparator(arr, i, j):
    if arr[i] > arr[j]:
        arr[i], arr[j] = arr[j], arr[i]

def _verify_network(comparators, sample_size=0):
    """Verify network sorts all 2^16 inputs (or a sample). Returns True if valid."""
    total = 1 << N
    if sample_size == 0 or sample_size >= total:
        indices = range(total)
    else:
        indices = random.sample(range(total), sample_size)
    for mask in indices:
        arr = [(mask >> k) & 1 for k in range(N)]
        for i, j in comparators:
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
        # Check sorted
        for k in range(N - 1):
            if arr[k] > arr[k + 1]:
                return False
    return True

def _compute_depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    # Greedy layering: assign each comparator to earliest possible layer
    wire_last_layer = [-1] * N
    max_layer = 0
    for i, j in comparators:
        layer = max(wire_last_layer[i], wire_last_layer[j]) + 1
        wire_last_layer[i] = layer
        wire_last_layer[j] = layer
        if layer > max_layer:
            max_layer = layer
    return max_layer

def _random_permutation_layer():
    """Generate a layer of 8 disjoint comparators from a random permutation."""
    wires = list(range(N))
    random.shuffle(wires)
    return [(wires[2*k], wires[2*k+1]) if wires[2*k] < wires[2*k+1]
            else (wires[2*k+1], wires[2*k]) for k in range(N // 2)]

def _random_network(num_layers):
    """Generate a random network with given number of layers."""
    comparators = []
    for _ in range(num_layers):
        comparators.extend(_random_permutation_layer())
    return comparators

def _crossover(parent1, parent2):
    """Uniform crossover: for each position, take comparator from one parent."""
    if len(parent1) != len(parent2):
        # Pad shorter with random layer
        shorter = parent1 if len(parent1) < len(parent2) else parent2
        longer = parent2 if len(parent1) < len(parent2) else parent1
        while len(shorter) < len(longer):
            shorter.extend(_random_permutation_layer())
        parent1, parent2 = shorter, longer
    child = []
    for a, b in zip(parent1, parent2):
        child.append(a if random.random() < 0.5 else b)
    return child

def _mutate(comparators, mutation_rate=0.1):
    """Mutate: randomly swap, insert, delete, or replace comparators."""
    comparators = list(comparators)
    n = len(comparators)
    num_mutations = max(1, int(n * mutation_rate))
    for _ in range(num_mutations):
        op = random.randint(0, 4)
        if op == 0 and n > 0:
            # Swap two comparators
            if n >= 2:
                a, b = random.sample(range(n), 2)
                comparators[a], comparators[b] = comparators[b], comparators[a]
        elif op == 1:
            # Replace a comparator with random one
            if n > 0:
                idx = random.randint(0, n - 1)
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                comparators[idx] = (i, j)
        elif op == 2 and n > 0:
            # Delete a comparator
            idx = random.randint(0, n - 1)
            comparators.pop(idx)
        elif op == 3:
            # Insert a random comparator
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            idx = random.randint(0, len(comparators))
            comparators.insert(idx, (i, j))
        else:
            # Insert a random layer
            comparators.extend(_random_permutation_layer())
    return comparators

def _greedy_construct():
    """Greedy construction: build network layer by layer, keeping useful comparators."""
    comparators = []
    # Start with a few random layers
    for _ in range(4):
        comparators.extend(_random_permutation_layer())
    # Try to extend until valid or max length
    max_len = 120
    while len(comparators) < max_len:
        if _verify_network(comparators, sample_size=1000):
            break
        # Add a random layer
        comparators.extend(_random_permutation_layer())
    return comparators

def _ga_search(best_comparators, best_score, time_limit=70):
    """Genetic algorithm to find better sorting network."""
    start_time = time.time()
    population = []
    
    # Initialize population
    pop_size = 30
    # Include best known
    population.append(list(best_comparators))
    
    # Add greedy constructions
    for _ in range(5):
        if time.time() > DEADLINE - 5:
            break
        population.append(_greedy_construct())
    
    # Add random networks of various sizes
    for size in [40, 50, 60, 70, 80, 90]:
        for _ in range(3):
            if time.time() > DEADLINE - 5:
                break
            population.append(_random_network(size // 8))
    
    # Add odd-even transposition as baseline
    population.append([(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)])
    
    # Evaluate initial population
    valid_pop = []
    for comp in population:
        if time.time() > DEADLINE - 5:
            break
        if _verify_network(comp):
            valid_pop.append(comp)
    
    if not valid_pop:
        return best_comparators, best_score
    
    # Sort by length
    valid_pop.sort(key=lambda x: len(x))
    best_comparators = valid_pop[0]
    best_score = len(best_comparators)
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        if time.time() > DEADLINE:
            break
        
        if not valid_pop:
            break
            
        # Select parents (tournament selection)
        new_pop = []
        for _ in range(pop_size):
            if time.time() > DEADLINE:
                break
                
            # Tournament of 3
            sample_size = min(3, len(valid_pop))
            if sample_size == 0:
                break
            candidates = random.sample(valid_pop, sample_size)
            candidates.sort(key=lambda x: len(x))
            parent = candidates[0]
            
            # Crossover
            if len(valid_pop) > 1 and random.random() < 0.7:
                sample_size2 = min(3, len(valid_pop))
                other_candidates = random.sample(valid_pop, sample_size2)
                other_candidates.sort(key=lambda x: len(x))
                parent2 = other_candidates[0]
                child = _crossover(parent, parent2)
            else:
                child = list(parent)
            
            # Mutate
            child = _mutate(child, mutation_rate=0.15)
            
            # Check validity with sample first
            if _verify_network(child, sample_size=500):
                if _verify_network(child):
                    new_pop.append(child)
        
        # Add new valid individuals
        valid_pop.extend(new_pop)
        
        # Keep only best (by length, then by depth)
        valid_pop.sort(key=lambda x: (len(x), _compute_depth(x)))
        valid_pop = valid_pop[:pop_size]
        
        # Update best
        if valid_pop and len(valid_pop[0]) < best_score:
            best_score = len(valid_pop[0])
            best_comparators = valid_pop[0]
    
    return best_comparators, best_score

def construct():
    # Start with odd-even transposition
    best = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    best_score = len(best)
    
    # Try greedy constructions
    for _ in range(5):
        if time.time() > DEADLINE - 10:
            break
        try:
            comp = _greedy_construct()
            if _verify_network(comp) and len(comp) < best_score:
                best = comp
                best_score = len(comp)
        except Exception:
            pass
    
    # Run genetic algorithm
    if time.time() < DEADLINE - 10:
        try:
            best, best_score = _ga_search(best, best_score, time_limit=55)
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
