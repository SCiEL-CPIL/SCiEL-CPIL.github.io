"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START
# Precompute bit masks for fast verification
_ALL_MASKS = np.arange(1 << N, dtype=np.int64)
_WIRE_BITS = np.array([(_ALL_MASKS >> i) & 1 for i in range(N)], dtype=np.int8)  # shape (N, 2^N)

def _verify_network_bitparallel(net):
    """Verify that a network sorts all 2^N Boolean inputs using bit-parallel ops."""
    if not net:
        return N <= 1
    # Start with all wire values as columns
    wires = _WIRE_BITS.copy()  # shape (N, 2^N)
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min goes to i, max goes to j
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    # Check if all wires are sorted: wire[k] <= wire[k+1] for all k
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True

def _build_initial_network():
    """Build an initial sorting network using Odd-Even Transposition Sort."""
    net = []
    for phase in range(N):
        if phase % 2 == 0:
            # Even phase: compare (0,1), (2,3), ...
            for i in range(0, N - 1, 2):
                net.append((i, i + 1))
        else:
            # Odd phase: compare (1,2), (3,4), ...
            for i in range(1, N - 1, 2):
                net.append((i, i + 1))
    return net

def _random_comparator():
    """Generate a random valid comparator pair (i, j) with i < j."""
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    return (i, j)

def _ga_optimize(initial_net, max_time=80.0, pop_size=20, tournament_k=3):
    """
    Genetic Algorithm to optimize the sorting network.
    """
    start_time = time.time()
    
    # Initialize population
    # Start with the initial network, then add some mutated versions
    population = [list(initial_net)]
    for _ in range(pop_size - 1):
        # Create mutated versions of the initial network
        mutant = list(initial_net)
        # Randomly remove some comparators (might break validity, so we verify)
        # For initialization, let's just use the initial network to ensure validity
        # Or we can try to remove random ones and keep if valid
        for _ in range(random.randint(0, 5)):
            if len(mutant) > 0:
                idx = random.randint(0, len(mutant) - 1)
                candidate = mutant[:idx] + mutant[idx+1:]
                if _verify_network_bitparallel(candidate):
                    mutant = candidate
        population.append(mutant)
    
    # Ensure all initial population members are valid
    valid_population = []
    for ind in population:
        if _verify_network_bitparallel(ind):
            valid_population.append(ind)
        else:
            valid_population.append(list(initial_net)) # Fallback to known valid
    
    best_net = min(valid_population, key=len)
    best_len = len(best_net)
    
    while time.time() - start_time < max_time:
        # Selection: Tournament selection
        offspring = []
        for _ in range(pop_size):
            # Select parents
            parents = []
            for _ in range(2):
                # Tournament
                candidates = random.sample(valid_population, min(tournament_k, len(valid_population)))
                parent = min(candidates, key=len)
                parents.append(parent)
            
            # Crossover: Uniform crossover
            # We need to handle the fact that the length might change if we just splice.
            # Simple approach: Take a random prefix from P1 and suffix from P2.
            # This preserves the sequential nature.
            p1, p2 = parents
            if len(p1) > 0 and len(p2) > 0:
                # Choose crossover point
                # We want to combine the "early" part of one and "late" part of another
                # But the "early" part of a sorting network is not necessarily interchangeable.
                # However, if both are valid, a splice might be valid.
                # Let's try uniform crossover on the list of comparators.
                # Since the lists might have different lengths, we pad or truncate?
                # No, we just take a random split point in P1 and P2.
                split1 = random.randint(0, len(p1))
                split2 = random.randint(0, len(p2))
                child = p1[:split1] + p2[split2:]
            else:
                child = list(p1) if p1 else list(p2)
            
            # Mutation
            # 1. Deletion (most important for minimization)
            if len(child) > 0 and random.random() < 0.7:
                idx = random.randint(0, len(child) - 1)
                candidate = child[:idx] + child[idx+1:]
                if _verify_network_bitparallel(candidate):
                    child = candidate
            # 2. Replacement
            elif len(child) > 0 and random.random() < 0.2:
                idx = random.randint(0, len(child) - 1)
                new_comp = _random_comparator()
                candidate = list(child)
                candidate[idx] = new_comp
                if _verify_network_bitparallel(candidate):
                    child = candidate
            # 3. Insertion (to escape local minima)
            elif random.random() < 0.1:
                new_comp = _random_comparator()
                idx = random.randint(0, len(child))
                candidate = child[:idx] + [new_comp] + child[idx:]
                if _verify_network_bitparallel(candidate):
                    child = candidate
            
            # Validate child
            if _verify_network_bitparallel(child):
                offspring.append(child)
            else:
                # If invalid, keep the best parent
                offspring.append(list(p1))

        # Update population
        # Combine current population and offspring, select best
        combined = valid_population + offspring
        # Sort by length
        combined.sort(key=len)
        valid_population = combined[:pop_size]
        
        # Update best
        current_best = valid_population[0]
        if len(current_best) < best_len:
            best_len = len(current_best)
            best_net = list(current_best)
    
    return best_net

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    # Start with a correct Odd-Even Transposition network
    net = _build_initial_network()
    
    # Normalize pairs to ensure i < j (safety check)
    normalized_net = []
    for i, j in net:
        if i > j:
            i, j = j, i
        if 0 <= i < j < N:
            normalized_net.append((i, j))
    net = normalized_net
    
    # Verify it's correct
    if not _verify_network_bitparallel(net):
        # Fallback: generate a simple bubble sort network (guaranteed correct)
        net = []
        for i in range(N):
            for j in range(i + 1, N):
                net.append((i, j))
    
    # Genetic Algorithm to improve
    remaining = deadline - time.time()
    if remaining > 5.0:
        net = _ga_optimize(net, max_time=min(remaining - 5.0, 80.0))
        
    # Final safety check: ensure all pairs are valid
    final_net = []
    for i, j in net:
        if 0 <= i < j < N:
            final_net.append((i, j))
            
    return final_net
# EVOLVE-BLOCK-END
