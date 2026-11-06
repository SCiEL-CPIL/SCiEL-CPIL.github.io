"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

N = 18

# EVOLVE-BLOCK-START
def _apply_network(network, inputs):
    """Apply a compare-exchange network to a list of inputs.
    Returns the sorted list if the network is correct for this input."""
    wires = list(inputs)
    for i, j in network:
        if wires[i] > wires[j]:
            wires[i], wires[j] = wires[j], wires[i]
    return wires

def _is_correct(network):
    """Check if the network sorts all 2^N boolean inputs."""
    for mask in range(1 << N):
        inputs = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, inputs)
        for i in range(N - 1):
            if result[i] > result[i + 1]:
                return False
    return True

def _depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    wire_last_step = [0] * N
    max_depth = 0
    for i, j in network:
        d = max(wire_last_step[i], wire_last_step[j]) + 1
        max_depth = max(max_depth, d)
        wire_last_step[i] = d
        wire_last_step[j] = d
    return max_depth

def _bitonic_merge_network(n, start=0):
    """Generate a bitonic merge network for n wires starting at wire 'start'.
    n must be a power of 2. Returns list of (i, j) comparators."""
    if n <= 1:
        return []
    comps = []
    logn = int(math.log2(n))
    for k in range(logn, 0, -1):
        step = 1 << k
        for i in range(start, start + n - step):
            comps.append((i, i + step))
    return comps

def _bitonic_sort_network(n, start=0):
    """Generate a bitonic sort network for n wires starting at wire 'start'.
    n must be a power of 2."""
    if n <= 1:
        return []
    if n == 2:
        return [(start, start + 1)]
    
    half = n // 2
    comps = []
    comps.extend(_bitonic_sort_network(half, start))
    comps.extend(_bitonic_sort_network(half, start + half))
    comps.extend(_bitonic_merge_network(n, start))
    return comps

def _generate_valid_parent():
    """Generate a valid sorting network using constructive heuristics.
    Returns a list of (i, j) comparators."""
    bitonic_32 = _bitonic_sort_network(32, 0)
    filtered = [(i, j) for i, j in bitonic_32 if i < N and j < N]
    if filtered:
        return filtered
    
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _remove_redundant_comparators(network):
    """Remove comparators that don't change the output for any input.
    Uses a subset of inputs for quick checking."""
    if not network:
        return network
    
    test_inputs = list(range(0, 1 << N, max(1, (1 << N) // 200)))
    
    def _is_correct_subset(net):
        for mask in test_inputs:
            inputs = [(mask >> i) & 1 for i in range(N)]
            wires = list(inputs)
            for i, j in net:
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
            for i in range(N - 1):
                if wires[i] > wires[i + 1]:
                    return False
        return True
    
    result = list(network)
    for idx in range(len(result) - 1, -1, -1):
        if len(result) <= 1:
            break
        test_net = result[:idx] + result[idx+1:]
        if _is_correct_subset(test_net):
            result.pop(idx)
    
    return result

def _crossover(parent1, parent2):
    """Crossover: take prefix from parent1 and suffix from parent2."""
    if not parent1:
        return list(parent2)
    if not parent2:
        return list(parent1)
    
    cut1 = random.randint(0, len(parent1))
    cut2 = random.randint(0, len(parent2))
    
    child = list(parent1[:cut1]) + list(parent2[cut2:])
    return child

def _mutation_add_comparator(network):
    """Add a random comparator at a random position."""
    if not network:
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        return [(i, j)]
    
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    pos = random.randint(0, len(network))
    new_net = list(network)
    new_net.insert(pos, (i, j))
    return new_net

def _mutation_remove_comparator(network):
    """Remove a random comparator."""
    if len(network) <= 1:
        return list(network)
    
    pos = random.randint(0, len(network) - 1)
    new_net = list(network)
    new_net.pop(pos)
    return new_net

def _mutation_swap_comparators(network):
    """Swap two comparators in the network."""
    if len(network) < 2:
        return list(network)
    
    i = random.randint(0, len(network) - 2)
    j = random.randint(i + 1, len(network) - 1)
    new_net = list(network)
    new_net[i], new_net[j] = new_net[j], new_net[i]
    return new_net

def _mutation(network, mutation_rate=0.3):
    """Apply one or more mutations to the network."""
    new_net = list(network)
    num_mutations = random.randint(1, 3)
    
    for _ in range(num_mutations):
        op = random.random()
        if op < 0.4:
            new_net = _mutation_add_comparator(new_net)
        elif op < 0.7:
            new_net = _mutation_remove_comparator(new_net)
        else:
            new_net = _mutation_swap_comparators(new_net)
    
    return new_net

def construct():
    """Construct a sorting network using a genetic algorithm."""
    start_time = time.time()
    timeout = 55
    
    population = []
    
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    population.append(seed)
    
    bitonic_net = _generate_valid_parent()
    if bitonic_net != seed:
        population.append(bitonic_net)
    
    for _ in range(8):
        base = _generate_valid_parent()
        net = list(base)
        for _ in range(random.randint(0, 5)):
            if len(net) > 1:
                pos = random.randint(0, len(net) - 1)
                net.pop(pos)
        if net:
            population.append(net)
    
    best_network = None
    best_size = float('inf')
    
    for net in population:
        if _is_correct(net):
            if len(net) < best_size:
                best_size = len(net)
                best_network = net
    
    if best_network is None:
        best_network = seed
        best_size = len(seed)
    
    pop_size = 16
    generation = 0
    
    while time.time() - start_time < timeout:
        generation += 1
        
        valid_population = []
        for net in population:
            if _is_correct(net):
                valid_population.append(net)
                if len(net) < best_size:
                    best_size = len(net)
                    best_network = net
        
        if not valid_population:
            population = [seed]
            continue
        
        valid_population.sort(key=lambda x: len(x))
        
        next_gen = []
        
        for net in valid_population[:4]:
            next_gen.append(list(net))
        
        while len(next_gen) < pop_size:
            p1_idx = random.choices(range(len(valid_population)),
                                   weights=[1/(i+1) for i in range(len(valid_population))])[0]
            p2_idx = random.choices(range(len(valid_population)),
                                   weights=[1/(i+1) for i in range(len(valid_population))])[0]
            
            p1 = valid_population[p1_idx]
            p2 = valid_population[p2_idx]
            
            child = _crossover(p1, p2)
            child = _mutation(child, mutation_rate=0.25)
            
            if len(child) <= best_size + 10:
                next_gen.append(child)
        
        population = next_gen
        
        if best_size <= 70:
            break
    
    if best_network:
        optimized = _remove_redundant_comparators(best_network)
        if _is_correct(optimized) and len(optimized) < len(best_network):
            best_network = optimized
    
    return best_network
# EVOLVE-BLOCK-END
