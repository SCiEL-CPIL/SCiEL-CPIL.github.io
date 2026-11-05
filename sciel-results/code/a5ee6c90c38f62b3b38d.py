"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import math

def _verify_network_fast(network):
    """
    Verify network sorts all 2^N Boolean inputs using the zero-one principle.
    Represents each wire's state across all 65536 inputs as a Python int bitmask.
    """
    n = N
    # Initialize wire states: wire w for input k is bit w of k
    wire_state = []
    for w in range(n):
        period = 1 << (w + 1)
        half = 1 << w
        period_mask = 0
        for bit_pos in range(half, period):
            period_mask |= (1 << bit_pos)
        mask = 0
        for rep in range(1 << (n - w - 1)):
            mask |= (period_mask << (rep * period))
        wire_state.append(mask)
    
    # Apply each comparator
    for (i, j) in network:
        a = wire_state[i]
        b = wire_state[j]
        wire_state[i] = a & b
        wire_state[j] = a | b
    
    # Check sorted
    for k in range(n - 1):
        if (wire_state[k] & ~wire_state[k + 1]) != 0:
            return False
    return True

def _generate_bitonic_network(n):
    """Generate a Bitonic Sorting Network for n wires (power of 2)."""
    network = []
    
    def bitonic_sort_recursive(size, offset):
        if size == 2:
            network.append((offset, offset + 1))
            return
        half = size // 2
        bitonic_sort_recursive(half, offset)
        bitonic_sort_recursive(half, offset + half)
        
        def merge_recursive(m_size, m_offset):
            if m_size == 2:
                network.append((m_offset, m_offset + 1))
                return
            m_half = m_size // 2
            for i in range(m_half):
                network.append((m_offset + i, m_offset + i + m_half))
            merge_recursive(m_half, m_offset)
            merge_recursive(m_half, m_offset + m_half)
        
        merge_recursive(size, offset)
    
    bitonic_sort_recursive(n, 0)
    return network

def _generate_odd_even_network(n):
    """Generate odd-even transposition network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _generate_random_network(n, num_comparators):
    """Generate a random network with given number of comparators."""
    network = []
    for _ in range(num_comparators):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        network.append((i, j))
    return network

def _prune_network(network, deadline):
    """Prune redundant comparators using random-order deletion."""
    if not network:
        return network
    if not _verify_network_fast(network):
        return network
    
    current = list(network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                improved = True
                break
    return current

def _crossover(parent1, parent2):
    """Uniform crossover: for each position, take from parent1 or parent2."""
    if len(parent1) != len(parent2):
        # Pad shorter with random comparators
        shorter = parent1 if len(parent1) < len(parent2) else parent2
        longer = parent2 if len(parent1) < len(parent2) else parent1
        while len(shorter) < len(longer):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            shorter.append((i, j))
        parent1, parent2 = shorter, longer
    
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def _mutate(network, mutation_rate=0.1):
    """Apply random mutations: add, remove, swap, reorder comparators."""
    child = list(network)
    
    # Random removals
    num_removals = int(len(child) * mutation_rate * 0.5)
    for _ in range(num_removals):
        if child:
            idx = random.randint(0, len(child) - 1)
            child.pop(idx)
    
    # Random additions
    num_additions = int(len(child) * mutation_rate * 0.5)
    for _ in range(num_additions):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(child))
        child.insert(pos, (i, j))
    
    # Random swaps of adjacent comparators
    for _ in range(int(len(child) * mutation_rate * 0.3)):
        if len(child) >= 2:
            idx = random.randint(0, len(child) - 2)
            child[idx], child[idx+1] = child[idx+1], child[idx]
    
    # Random reordering of a segment
    if len(child) >= 4:
        start = random.randint(0, len(child) - 4)
        end = random.randint(start + 3, len(child) - 1)
        segment = child[start:end+1]
        random.shuffle(segment)
        child[start:end+1] = segment
    
    return child

def _tournament_select(population, scores, k=3):
    """Tournament selection."""
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: scores[i])
    return population[best_idx]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Generate initial diverse population
    population = []
    
    # Bitonic network (should be ~60 comparators)
    bitonic = _generate_bitonic_network(N)
    population.append(bitonic)
    
    # Odd-even network (120 comparators)
    odd_even = _generate_odd_even_network(N)
    population.append(odd_even)
    
    # Several random networks of varying sizes
    for size in [50, 55, 60, 65, 70, 80, 90, 100]:
        random_net = _generate_random_network(N, size)
        population.append(random_net)
    
    # Evaluate initial population
    scores = []
    for net in population:
        if _verify_network_fast(net):
            scores.append(len(net))
        else:
            scores.append(float('inf'))
    
    best_idx = scores.index(min(scores))
    best_network = list(population[best_idx])
    best_score = scores[best_idx]
    
    # Prune the best initial network
    best_network = _prune_network(best_network, deadline)
    if _verify_network_fast(best_network):
        best_score = len(best_network)
    else:
        best_network = list(population[best_idx])
        best_score = scores[best_idx]
    
    # Also prune other valid networks in population
    for i in range(len(population)):
        if scores[i] < float('inf') and i != best_idx:
            pruned = _prune_network(population[i], deadline)
            if _verify_network_fast(pruned) and len(pruned) < scores[i]:
                population[i] = pruned
                scores[i] = len(pruned)
                if len(pruned) < best_score:
                    best_network = list(pruned)
                    best_score = len(pruned)
    
    # Evolutionary search
    pop_size = 20
    # Fill population with mutated copies of best
    while len(population) < pop_size:
        if best_network:
            mutant = _mutate(best_network, mutation_rate=0.15)
            population.append(mutant)
        else:
            population.append(_generate_random_network(N, 60))
    
    # Keep only valid networks or those close to valid
    # Re-evaluate all
    for i in range(len(population)):
        if not _verify_network_fast(population[i]):
            scores[i] = float('inf')
        else:
            scores[i] = len(population[i])
    
    generations = 0
    while time.time() < deadline - 2.0:
        generations += 1
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individual
        valid_indices = [i for i in range(len(population)) if scores[i] < float('inf')]
        if valid_indices:
            elite_idx = max(valid_indices, key=lambda i: -scores[i])  # smallest score
            new_population.append(list(population[elite_idx]))
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            parent1 = _tournament_select(population, scores, k=3)
            parent2 = _tournament_select(population, scores, k=3)
            child = _crossover(parent1, parent2)
            child = _mutate(child, mutation_rate=0.08)
            new_population.append(child)
        
        # Evaluate new population
        new_scores = []
        for net in new_population:
            if _verify_network_fast(net):
                new_scores.append(len(net))
            else:
                new_scores.append(float('inf'))
        
        # Prune the best in new population
        valid_new = [i for i in range(len(new_population)) if new_scores[i] < float('inf')]
        if valid_new:
            best_new_idx = max(valid_new, key=lambda i: -new_scores[i])
            pruned = _prune_network(new_population[best_new_idx], deadline)
            if _verify_network_fast(pruned):
                new_population[best_new_idx] = pruned
                new_scores[best_new_idx] = len(pruned)
        
        # Update best
        for i in range(len(new_population)):
            if new_scores[i] < best_score:
                best_score = new_scores[i]
                best_network = list(new_population[i])
        
        # Replace population
        population = new_population
        scores = new_scores
        
        # Occasionally inject fresh random networks to maintain diversity
        if generations % 10 == 0 and time.time() < deadline - 5.0:
            for _ in range(3):
                idx = random.randint(0, len(population) - 1)
                size = random.randint(50, 75)
                population[idx] = _generate_random_network(N, size)
                if _verify_network_fast(population[idx]):
                    scores[idx] = len(population[idx])
                else:
                    scores[idx] = float('inf')
    
    # Final prune of best
    final = _prune_network(best_network, deadline)
    if _verify_network_fast(final):
        best_network = final
    
    return best_network
# EVOLVE-BLOCK-END
