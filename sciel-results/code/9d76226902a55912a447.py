"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def _batcher_merge(n):
    """Generate a bitonic merge network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Step 1: Compare (i, i+half) for i in 0..half-1
    step1 = [(i, i + half) for i in range(half)]
    
    # Step 2: Recursively merge even indices and odd indices
    sub_merge = _batcher_merge(half)
    evens = [(2*i, 2*j) for i, j in sub_merge]
    odds = [(2*i+1, 2*j+1) for i, j in sub_merge]
    
    return step1 + evens + odds

def _batcher(n):
    """Generate a bitonic sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort first half ascending
    left = _batcher(half)
    
    # Sort second half descending
    # To get descending, take ascending network and swap i, j
    right_asc = _batcher(half)
    right = [(j + half, i + half) for i, j in right_asc]
    
    # Merge
    merge = _batcher_merge(n)
    
    return left + right + merge

def _construct_initial_network():
    """Construct a valid sorting network for N=18 using Batcher(16) + Insertion."""
    net = _batcher(16)
    # Insert wire 16
    for i in range(15, -1, -1):
        net.append((i, 16))
    # Insert wire 17
    for i in range(16, -1, -1):
        net.append((i, 17))
    return net

def _is_correct_vec(network):
    """Check correctness using vectorized NumPy operations."""
    # Generate all 2^18 boolean inputs
    # Using a list comprehension is slow, let's use bit manipulation on arange
    # Or just use the list comp, it's only 262k items.
    # inputs = np.array([(mask >> i) & 1 for mask in range(1 << N) for i in range(N)]).reshape(-1, N)
    
    # Faster generation:
    # Create a range of 2^N
    # Expand to N bits
    # This might be memory heavy if done in one go, but 262k * 18 is fine.
    
    # Let's stick to the list comp for clarity and safety, it's fast enough for 262k.
    inputs_list = []
    for mask in range(1 << N):
        inputs_list.append([(mask >> i) & 1 for i in range(N)])
    inputs = np.array(inputs_list, dtype=np.int8)
    
    wires = inputs.copy()
    
    for i, j in network:
        temp = wires[:, i].copy()
        wires[:, i] = np.minimum(temp, wires[:, j])
        wires[:, j] = np.maximum(temp, wires[:, j])
        
    return np.all(wires[:, :-1] <= wires[:, 1:])

def _crossover(parent1, parent2):
    """Crossover two networks."""
    if not parent1 or not parent2:
        return list(parent1) if parent1 else list(parent2)
    cut = random.randint(0, min(len(parent1), len(parent2)))
    child = list(parent1[:cut]) + list(parent2[cut:])
    return child

def _mutation(network):
    """Mutate a network."""
    network = list(network)
    r = random.random()
    
    if r < 0.3 and network:
        # Remove a random comparator
        pos = random.randint(0, len(network) - 1)
        network.pop(pos)
    elif r < 0.4:
        # Add a random comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(network))
        network.insert(pos, (i, j))
    elif r < 0.7 and len(network) >= 2:
        # Swap two comparators
        i = random.randint(0, len(network) - 2)
        j = random.randint(i + 1, len(network) - 1)
        network[i], network[j] = network[j], network[i]
    else:
        # Move a comparator to a new position
        if network:
            i = random.randint(0, len(network) - 1)
            comp = network.pop(i)
            j = random.randint(0, len(network))
            network.insert(j, comp)
            
    return network

def construct():
    """Construct a sorting network using a genetic algorithm."""
    start_time = time.time()
    timeout = 55
    
    # Initialize population with the constructive heuristic
    seed_net = _construct_initial_network()
    
    population = [seed_net]
    # Add some mutated versions of the seed
    for _ in range(19):
        population.append(_mutation(seed_net))
    
    best_network = seed_net
    best_size = len(seed_net)
    
    # Verify seed is correct
    if _is_correct_vec(seed_net):
        best_size = len(seed_net)
    else:
        # If seed is wrong (shouldn't happen), fallback to odd-even transposition
        seed_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_network = seed_net
        best_size = len(seed_net)
        population = [seed_net] * 20

    generation = 0
    pop_size = 20
    
    while time.time() - start_time < timeout:
        generation += 1
        
        # Evaluate fitness
        scored = []
        for net in population:
            # Only check correctness if it's potentially better or we need to filter
            # To save time, we can first check size.
            # But we need to know if it's valid to update best.
            
            is_correct = _is_correct_vec(net)
            
            if is_correct:
                if len(net) < best_size:
                    best_size = len(net)
                    best_network = net
                scored.append((len(net), net))
            else:
                # Penalize invalid networks heavily
                scored.append((len(net) + 10000, net))
        
        # Sort by fitness
        scored.sort(key=lambda x: x[0])
        population = [net for _, net in scored]
        
        # Create next generation
        next_gen = []
        # Elitism: keep the best 2
        for net in population[:2]:
            next_gen.append(list(net))
        
        # Crossover and mutation
        while len(next_gen) < pop_size:
            # Tournament selection
            p1 = random.choice(population[:10])
            p2 = random.choice(population[:10])
            child = _crossover(p1, p2)
            child = _mutation(child)
            next_gen.append(child)
        
        population = next_gen
        
        # Early exit if we found a very good network
        if best_size <= 70:
            break
    
    return best_network
# EVOLVE-BLOCK-END
