"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

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
    # For boolean inputs, we can check all 2^N possibilities
    # But 2^18 = 262144, which is manageable
    for mask in range(1 << N):
        inputs = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, inputs)
        # Check if result is sorted (non-decreasing)
        for i in range(N - 1):
            if result[i] > result[i + 1]:
                return False
    return True

def _depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    # Build dependency graph: a comparator (i,j) depends on any previous comparator
    # that touches wire i or j
    wire_last_step = [0] * N
    max_depth = 0
    for i, j in network:
        d = max(wire_last_step[i], wire_last_step[j]) + 1
        max_depth = max(max_depth, d)
        wire_last_step[i] = d
        wire_last_step[j] = d
    return max_depth

def _bitonic_merge(n):
    """Generate a bitonic merge network for n wires (n must be power of 2).
    Returns list of comparators."""
    if n <= 1:
        return []
    # Bitonic merge: for k in 1..log2(n), for i in 0..n-k-1, compare (i, i+k)
    # Actually, standard bitonic merge:
    # Step 1: compare (0,n/2), (1,n/2+1), ...
    # Step 2: compare (0,n/4), (n/4, n/2), ... etc.
    # Let's use the recursive definition.
    # Bitonic merge of n elements:
    # 1. Split into two halves of n/2
    # 2. Recursively merge each half (but they're already bitonic? No.)
    # Actually, bitonic merge: given a bitonic sequence of length n (power of 2),
    # we can sort it in log2(n) stages.
    # Stage k (k=1..log2(n)): for i in 0..n-2^k-1, compare (i, i+2^k)
    # Wait, that's not quite right either.
    
    # Standard bitonic merge for n elements (n power of 2):
    # For k = log2(n) down to 1:
    #   For i = 0 to n - 2^k - 1:
    #     Compare (i, i + 2^k)
    
    import math
    comps = []
    if n <= 1:
        return comps
    logn = int(math.log2(n))
    for k in range(logn, 0, -1):
        step = 1 << k
        for i in range(0, n - step):
            comps.append((i, i + step))
    return comps

def _bitonic_sort(n):
    """Generate a bitonic sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    # Bitonic sort: 
    # 1. Recursively sort first half and second half (one ascending, one descending)
    # 2. Bitonic merge
    # For simplicity, let's just use the known structure.
    # Actually, let's build it recursively.
    
    # A simpler approach: use the known bitonic sort network structure.
    # For n = 2^k, the bitonic sort network has k stages.
    # But implementing it correctly is complex. Let's use a different approach.
    
    # Let's just generate a valid sorting network using a simpler method.
    # For now, let's use odd-even merge or just return a known good network.
    
    # Actually, let's implement a basic bitonic sort:
    # Stage 1: Sort pairs (0,1), (2,3), ... ascending, and (n-2,n-1), (n-4,n-3), ... 
    # This is getting complicated. Let's use a different constructive heuristic.
    
    # For the GA, we just need valid parents. Let's use a simple valid network:
    # The seed odd-even transposition network is valid.
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _random_valid_network():
    """Generate a random valid sorting network using a constructive approach."""
    # Start with the seed network and add random comparators
    base = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    # Add some random comparators
    network = list(base)
    for _ in range(random.randint(0, 20)):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        network.append((i, j))
    return network

def _crossover(parent1, parent2):
    """Crossover two networks: take a prefix from one and suffix from the other."""
    if not parent1 or not parent2:
        return list(parent1) if parent1 else list(parent2)
    cut = random.randint(0, min(len(parent1), len(parent2)))
    child = list(parent1[:cut]) + list(parent2[cut:])
    return child

def _mutation(network, rate=0.1):
    """Mutate a network by adding, removing, or swapping comparators."""
    network = list(network)
    # Add a random comparator
    if random.random() < rate:
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(network))
        network.insert(pos, (i, j))
    # Remove a random comparator
    if random.random() < rate and network:
        pos = random.randint(0, len(network) - 1)
        network.pop(pos)
    # Swap two comparators
    if random.random() < rate and len(network) >= 2:
        i = random.randint(0, len(network) - 2)
        j = random.randint(i + 1, len(network) - 1)
        network[i], network[j] = network[j], network[i]
    return network

def construct():
    """Construct a sorting network using a genetic algorithm."""
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for safety
    
    # Initialize population
    population = []
    # Add the seed network
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    population.append(seed)
    
    # Add some random valid networks
    for _ in range(19):
        population.append(_random_valid_network())
    
    best_network = seed
    best_size = len(seed)
    
    # Verify seed is correct
    if _is_correct(seed):
        best_size = len(seed)
    
    generation = 0
    pop_size = 20
    
    while time.time() - start_time < timeout:
        generation += 1
        
        # Evaluate fitness
        scored = []
        for net in population:
            if len(net) < best_size:
                if _is_correct(net):
                    scored.append((len(net), net))
                    best_size = len(net)
                    best_network = net
                else:
                    # Penalize invalid networks
                    scored.append((len(net) + 1000, net))
            else:
                scored.append((len(net), net))
        
        # Sort by fitness
        scored.sort(key=lambda x: x[0])
        population = [net for _, net in scored]
        
        # Create next generation
        next_gen = []
        # Elitism: keep the best few
        for net in population[:5]:
            next_gen.append(list(net))
        
        # Crossover and mutation
        while len(next_gen) < pop_size:
            p1 = random.choice(population[:10])
            p2 = random.choice(population[:10])
            child = _crossover(p1, p2)
            child = _mutation(child, rate=0.15)
            next_gen.append(child)
        
        population = next_gen
        
        # Early exit if we found a very good network
        if best_size <= 70:
            break
    
    return best_network
# EVOLVE-BLOCK-END
