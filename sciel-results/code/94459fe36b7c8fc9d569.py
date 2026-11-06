"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify(net):
    """Verify sorting network via zero-one principle (all 2^16 inputs)."""
    # Use numpy for speed: apply network to all 65536 boolean inputs at once
    # Each input is a 16-bit integer. We'll process in batches.
    n_inputs = 1 << N
    # Represent each input as an array of N bits
    # For memory efficiency, process in chunks
    chunk_size = 4096
    for start in range(0, n_inputs, chunk_size):
        end = min(start + chunk_size, n_inputs)
        # Create input vectors: shape (chunk, N)
        idxs = np.arange(start, end, dtype=np.uint8)
        # Extract bits: bit i of idx corresponds to wire i
        bits = ((idxs[:, None] >> np.arange(N, dtype=np.uint8)) & 1).astype(np.uint8)
        # Apply network
        for (i, j) in net:
            a = bits[:, i]
            b = bits[:, j]
            # min goes to i, max goes to j
            bits[:, i] = np.minimum(a, b)
            bits[:, j] = np.maximum(a, b)
        # Check sorted: bits[:, k] >= bits[:, k+1] for all k
        for k in range(N - 1):
            if not np.all(bits[:, k] >= bits[:, k + 1]):
                return False
    return True


def _depth(net):
    """Compute parallel depth of the network."""
    if not net:
        return 0
    # Simple layer-based: group comparators into layers where no wire is used twice
    wire_last_layer = [-1] * N
    depth = 0
    for (i, j) in net:
        d = max(wire_last_layer[i], wire_last_layer[j]) + 1
        wire_last_layer[i] = d
        wire_last_layer[j] = d
        depth = max(depth, d)
    return depth


def _random_network(n_comp):
    """Generate a random comparator network."""
    net = []
    for _ in range(n_comp):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        net.append((i, j))
    return net


def _mutate(net, num_mutations=2):
    """Mutate a network by changing some comparators."""
    net = list(net)
    for _ in range(num_mutations):
        if not net:
            break
        idx = random.randint(0, len(net) - 1)
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        net[idx] = (i, j)
    return net


def _crossover(net1, net2):
    """Uniform crossover of two networks."""
    result = []
    for (a, b) in zip(net1, net2):
        if random.random() < 0.5:
            result.append(a)
        else:
            result.append(b)
    return result


def _local_search(net, deadline, max_evals=5000):
    """Local search: try small perturbations."""
    best = list(net)
    best_count = len(best)
    best_valid = _verify(best)
    start = time.time()
    evals = 0
    
    while time.time() < deadline and evals < max_evals:
        # Try removing a comparator
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            candidate = best[:idx] + best[idx+1:]
            evals += 1
            if _verify(candidate):
                best = candidate
                best_count = len(best)
                continue
        
        # Try changing a comparator
        idx = random.randint(0, len(best) - 1)
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = list(best)
        candidate[idx] = (i, j)
        evals += 1
        if _verify(candidate):
            best = candidate
            best_count = len(best)
    
    return best, best_valid


def _genetic_algorithm(deadline):
    """Genetic algorithm to find small sorting networks."""
    start = time.time()
    
    # Initialize population with various sizes
    pop = []
    # Start with known good network (odd-even transposition, 120 comps)
    base = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    pop.append(list(base))
    
    # Add random networks of various sizes
    for size in [100, 110, 115, 120, 125, 130]:
        net = _random_network(size)
        pop.append(net)
    
    # Keep best valid network found
    best_valid = None
    best_count = float('inf')
    
    # Check initial population
    for net in pop:
        if _verify(net):
            if len(net) < best_count:
                best_count = len(net)
                best_valid = list(net)
    
    if best_valid is None:
        # If no valid network found, return base
        return base
    
    generation = 0
    max_generations = 100
    
    while time.time() < deadline and generation < max_generations:
        generation += 1
        
        # Sort population by size (prefer smaller valid networks)
        valid_nets = [n for n in pop if _verify(n)]
        if valid_nets:
            valid_nets.sort(key=len)
            if len(valid_nets[0]) < best_count:
                best_count = len(valid_nets[0])
                best_valid = list(valid_nets[0])
        
        # Create next generation
        new_pop = []
        
        # Elitism: keep best few
        for net in (valid_nets[:3] if valid_nets else pop[:3]):
            new_pop.append(list(net))
        
        # Crossover and mutation
        while len(new_pop) < len(pop):
            if len(pop) < 2:
                break
            p1 = random.choice(pop)
            p2 = random.choice(pop)
            child = _crossover(p1, p2)
            # Random size adjustment
            if random.random() < 0.3 and child:
                if random.random() < 0.5:
                    # Add a comparator
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    child.append((i, j))
                else:
                    # Remove a comparator
                    idx = random.randint(0, len(child) - 1)
                    child.pop(idx)
            child = _mutate(child, num_mutations=1)
            new_pop.append(child)
        
        pop = new_pop
    
    return best_valid if best_valid else base


def _hill_climb(deadline):
    """Hill climbing from odd-even transposition."""
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best = list(net)
    start = time.time()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each comparator
        for idx in range(len(best)):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify(candidate):
                best = candidate
                improved = True
                break
        # Try swapping comparators
        if not improved:
            for idx in range(len(best)):
                if time.time() > deadline:
                    break
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate = list(best)
                candidate[idx] = (i, j)
                if _verify(candidate):
                    best = candidate
                    improved = True
                    break
    
    return best


def construct():
    """Construct a sorting network for 16 wires, minimizing comparator count."""
    deadline = time.time() + 70  # Safety margin before 90s limit
    
    # Start with hill climbing from odd-even transposition
    best = _hill_climb(time.time() + 30)
    
    # If time remains, try genetic algorithm
    if time.time() < deadline - 10:
        ga_result = _genetic_algorithm(deadline)
        if ga_result and len(ga_result) < len(best):
            best = ga_result
    
    # Final local search polish
    if time.time() < deadline - 5:
        best, _ = _local_search(best, deadline, max_evals=2000)
    
    # Ensure we return a valid network; fallback to odd-even if something went wrong
    if not _verify(best):
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return best

# EVOLVE-BLOCK-END
