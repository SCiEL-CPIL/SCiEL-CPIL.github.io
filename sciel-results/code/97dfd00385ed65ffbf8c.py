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
    for mask in range(1 << N):
        inputs = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, inputs)
        for i in range(N - 1):
            if result[i] > result[i + 1]:
                return False
    return True

def _prune_network(network):
    """Attempt to remove redundant comparators from a valid network."""
    if not network:
        return network
    
    # Work on a copy
    current_net = list(network)
    changed = True
    while changed:
        changed = False
        # Try removing each comparator
        # To speed up, we can check in reverse order or random order
        # Removing a comparator that was just added is less likely to be useful than removing old ones,
        # but we want to find the minimal set.
        # We iterate through indices in reverse to avoid index shifting issues if we remove
        # or we just create a new list.
        
        for i in range(len(current_net) - 1, -1, -1):
            # Try removing current_net[i]
            test_net = current_net[:i] + current_net[i+1:]
            if _is_correct(test_net):
                current_net = test_net
                changed = True
                # Restart the loop from the end? Or just continue?
                # Continuing is fine, as we are removing one at a time.
                # However, removing one might make another redundant.
                # The `while changed` loop handles this.
                
    return current_net

def _batcher_bitonic(n, reverse=False):
    """Generate a Batcher's Bitonic Sort network for n wires."""
    if n <= 1:
        return []
    
    # Recursively sort halves
    n1 = n // 2
    n2 = n - n1
    
    # Sort first half ascending
    net1 = _batcher_bitonic(n1, reverse=False)
    # Sort second half descending (if we want ascending final result)
    # If reverse is True, we want the final result to be descending.
    # Standard Bitonic Sort:
    # 1. Split into two halves.
    # 2. Sort first half Ascending.
    # 3. Sort second half Descending.
    # 4. Bitonic Merge.
    
    # If we want the final output to be Ascending (reverse=False):
    #   Left half Ascending, Right half Descending -> Merge -> Ascending.
    # If we want the final output to be Descending (reverse=True):
    #   Left half Descending, Right half Ascending -> Merge -> Descending.
    
    if not reverse:
        net1 = _batcher_bitonic(n1, reverse=False)
        # Map indices for net2: they are in the second half [n1, n)
        net2_raw = _batcher_bitonic(n2, reverse=True)
        net2 = [(i + n1, j + n1) for i, j in net2_raw]
    else:
        net1 = _batcher_bitonic(n1, reverse=True)
        net2_raw = _batcher_bitonic(n2, reverse=False)
        net2 = [(i + n1, j + n1) for i, j in net2_raw]
    
    # Bitonic Merge
    # The merge network for n elements:
    # Stage 1: Compare (i, n-1-i) for i in 0..n/2-1
    # Stage 2: Recursively merge the two halves.
    
    merge_net = []
    m = n // 2
    # First stage of merge
    for i in range(m):
        merge_net.append((i, n - 1 - i))
    
    # Recursively merge the two resulting bitonic halves
    # The first half (0..m-1) is now bitonic, second half (m..n-1) is bitonic.
    # We need to sort them.
    # Note: The recursive merge calls assume the input to the merge is bitonic.
    # The standard recursive bitonic merge function:
    # merge(n):
    #   if n == 1: return
    #   compare (i, n-1-i) for i in 0..n/2-1
    #   merge(n/2) on first half
    #   merge(n/2) on second half
    
    # So we call the merge logic recursively.
    # But _batcher_bitonic includes the sorting of halves.
    # We need a separate _bitonic_merge function.
    
    def _bitonic_merge(k, offset=0):
        if k <= 1:
            return []
        comps = []
        half = k // 2
        # Compare (i, k-1-i)
        for i in range(half):
            comps.append((offset + i, offset + k - 1 - i))
        
        # Recursively merge the two halves
        # First half: offset to offset + half
        # Second half: offset + half to offset + k
        comps.extend(_bitonic_merge(half, offset))
        comps.extend(_bitonic_merge(k - half, offset + half))
        return comps

    merge_net = _bitonic_merge(n)
    
    return net1 + net2 + merge_net

def _random_valid_network():
    """Generate a random valid sorting network using a constructive approach."""
    # Use Bitonic Sort as a base, then add some random noise?
    # Or just return the Bitonic network.
    return _batcher_bitonic(N)

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
    """Construct a sorting network using a genetic algorithm with pruning."""
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for safety
    
    # Generate initial valid networks
    # 1. Seed (Odd-Even Transposition)
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    # 2. Bitonic Sort
    bitonic = _batcher_bitonic(N)
    
    # Prune them to get locally optimal versions
    pruned_seed = _prune_network(seed)
    pruned_bitonic = _prune_network(bitonic)
    
    # Select the best pruned network as the initial best
    if len(pruned_seed) < len(pruned_bitonic):
        best_network = pruned_seed
    else:
        best_network = pruned_bitonic
        
    best_size = len(best_network)
    
    # Initialize population with the best and some variants
    population = [list(best_network)]
    
    # Add some other valid networks (unpruned or slightly mutated)
    population.append(list(seed))
    population.append(list(bitonic))
    
    # Add some random mutations of the best
    for _ in range(10):
        mutated = _mutation(best_network, rate=0.2)
        if _is_correct(mutated):
            population.append(mutated)
        else:
            # If invalid, just keep the best
            population.append(list(best_network))
            
    pop_size = len(population)
    
    generation = 0
    while time.time() - start_time < timeout:
        generation += 1
        
        # Evaluate fitness
        scored = []
        for net in population:
            if _is_correct(net):
                scored.append((len(net), net))
                if len(net) < best_size:
                    best_size = len(net)
                    best_network = list(net)
            else:
                # Penalize invalid networks heavily
                scored.append((len(net) + 1000, net))
        
        # Sort by fitness
        scored.sort(key=lambda x: x[0])
        population = [net for _, net in scored]
        
        # Prune the top few candidates to potentially find smaller valid networks
        # This is expensive, so do it only for the best few
        for i in range(min(5, len(population))):
            pruned = _prune_network(population[i])
            if len(pruned) < len(population[i]):
                population[i] = pruned
                if len(pruned) < best_size:
                    best_size = len(pruned)
                    best_network = list(pruned)
        
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
            
            # Check if valid, if not, try to repair or keep parent
            if _is_correct(child):
                next_gen.append(child)
            else:
                # If invalid, we can try to prune a valid parent or just keep parent
                next_gen.append(list(p1))
        
        population = next_gen
        
        # Early exit if we found a very good network
        if best_size <= 70:
            break
            
    return best_network
# EVOLVE-BLOCK-END
