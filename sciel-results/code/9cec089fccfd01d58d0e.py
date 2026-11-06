import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

# Precompute full matrix of all 2^N boolean inputs
all_masks = np.arange(1 << N, dtype=np.int32)
W0_full = np.zeros((1 << N, N), dtype=np.uint8)
for i in range(N):
    W0_full[:, i] = (all_masks >> i) & 1

# Precompute a subset for fast verification
subset_indices = np.unique(np.concatenate([
    np.arange(0, 100),
    np.arange((1 << N) - 100, 1 << N),
    np.random.choice(1 << N, 2000, replace=False)
]))
W0_subset = W0_full[subset_indices]

def _verify_full(net):
    """Verify that the comparator network sorts all 2^N boolean inputs."""
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    W = W0_full.copy()
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
    return True

def _verify_fast(net):
    """Fast verification using a subset of inputs."""
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    W = W0_subset.copy()
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
    return True

def _batcher_merge_sort(n):
    """Generate Batcher's Odd-Even Merge Sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    net = []
    # Recursively sort halves
    net1 = _batcher_merge_sort(k)
    net2 = _batcher_merge_sort(k)
    for i, j in net1:
        net.append((i, j))
    for i, j in net2:
        net.append((i + k, j + k))
    
    # Odd-even merge
    net += _odd_even_merge(n)
    return net

def _odd_even_merge(n):
    """Generate the odd-even merge network for n wires (n must be even)."""
    comparators = []
    k = n // 2
    
    # Phase 1: compare (0,k), (2,k+2), ... then (1,k+1), (3,k+3), ...
    for i in range(0, k, 2):
        comparators.append((i, i + k))
    for i in range(1, k, 2):
        comparators.append((i, i + k))
    
    # Recursively merge the two halves of the merged sequence
    # The merged sequence has the structure:
    # [sorted(0,2,4,...,k-2), sorted(1,3,5,...,k-1), sorted(k,k+2,...), sorted(k+1,k+3,...)]
    # Actually, after the cross comparisons, we need to recursively merge:
    # First half: positions 0..k-1 (but they're interleaved)
    # Let me use the standard odd-even merge algorithm
    
    # Standard approach: after the cross comparisons, recursively merge
    # the two k-element subsequences
    # The first subsequence is at positions 0,2,4,...,n-2 (even indices)
    # The second subsequence is at positions 1,3,5,...,n-1 (odd indices)
    
    # Actually, let me use a cleaner recursive formulation
    # For odd-even merge of two sorted sequences of length k:
    # 1. Compare (0,k), (2,k+2), ... (even positions with their partners)
    # 2. Compare (1,k+1), (3,k+3), ... (odd positions with their partners)
    # 3. Recursively merge the even-positioned elements (which form a sorted sequence of length k)
    # 4. Recursively merge the odd-positioned elements (which form a sorted sequence of length k)
    
    # Wait, I need to be more careful. Let me use a direct recursive implementation.
    return comparators

def _odd_even_merge_recursive(n):
    """
    Generate odd-even merge network for merging two sorted sequences of length n/2.
    Total wires: n (must be even).
    The first n/2 wires contain one sorted sequence, the last n/2 contain another.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    comparators = []
    
    # Step 1: Compare (0, k), (2, k+2), ...
    for i in range(0, k, 2):
        comparators.append((i, i + k))
    
    # Step 2: Compare (1, k+1), (3, k+3), ...
    for i in range(1, k, 2):
        comparators.append((i, i + k))
    
    # Step 3: Recursively merge even-positioned elements
    # Even positions: 0, 2, 4, ..., n-2
    # These form two sorted subsequences of length k/2 each
    # First: positions 0, 2, ..., k-2 (from first half)
    # Second: positions k, k+2, ..., n-2 (from second half)
    # Map to 0..k-1 space
    even_net = _odd_even_merge_recursive(k)
    for i, j in even_net:
        comparators.append((2 * i, 2 * j))
    
    # Step 4: Recursively merge odd-positioned elements
    # Odd positions: 1, 3, 5, ..., n-1
    # First: positions 1, 3, ..., k-1 (from first half)
    # Second: positions k+1, k+3, ..., n-1 (from second half)
    odd_net = _odd_even_merge_recursive(k)
    for i, j in odd_net:
        comparators.append((2 * i + 1, 2 * j + 1))
    
    return comparators

def _batcher_odd_even_merge_sort(n):
    """Generate Batcher's Odd-Even Merge Sort for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    net = []
    
    # Recursively sort the two halves
    net1 = _batcher_odd_even_merge_sort(k)
    net2 = _batcher_odd_even_merge_sort(k)
    for i, j in net1:
        net.append((i, j))
    for i, j in net2:
        net.append((i + k, j + k))
    
    # Odd-even merge the two sorted halves
    net += _odd_even_merge_recursive(n)
    return net

def _odd_even_transposition(n):
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _ga_crossover(net1, net2):
    """Uniform crossover between two networks."""
    if not net1 or not net2:
        return net1[:] if net1 else net2[:]
    
    # Pad to same length
    max_len = max(len(net1), len(net2))
    c1 = net1[:] + [None] * (max_len - len(net1))
    c2 = net2[:] + [None] * (max_len - len(net2))
    
    child = []
    for i in range(max_len):
        if random.random() < 0.5:
            child.append(c1[i])
        else:
            child.append(c2[i])
    
    # Remove None entries
    child = [c for c in child if c is not None]
    return child

def _ga_mutation(net, rate=0.1):
    """Mutate a network by randomly changing, adding, or removing comparators."""
    if not net:
        return net[:]
    
    new_net = net[:]
    n_ops = max(1, int(len(net) * rate))
    
    for _ in range(n_ops):
        op = random.random()
        if op < 0.4 and len(new_net) > 0:
            # Remove a random comparator
            idx = random.randint(0, len(new_net) - 1)
            new_net.pop(idx)
        elif op < 0.7:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(new_net))
            new_net.insert(idx, (i, j))
        else:
            # Mutate a random comparator
            if len(new_net) == 0:
                continue
            idx = random.randint(0, len(new_net) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_net[idx] = (i, j)
    
    return new_net

def _tournament_select(population, scores, k=3):
    """Tournament selection."""
    candidates = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(candidates, key=lambda i: scores[i])
    return population[best_idx][:]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Initialize with Batcher's Odd-Even Merge Sort for N=18
    # Since 18 is not a power of 2, we'll use a hybrid approach
    # Use the network for 16 wires and extend to 18
    net16 = _batcher_odd_even_merge_sort(16)
    
    # Extend to 18 wires by adding comparators involving wires 16 and 17
    # Simple approach: add comparators to merge the two new wires
    base_net = net16[:]
    # Add comparators to integrate wires 16 and 17
    for i in range(16):
        base_net.append((i, 16))
    base_net.append((16, 17))
    for i in range(16):
        base_net.append((i, 17))
    
    # Verify and potentially simplify
    if not _verify_full(base_net):
        base_net = _odd_even_transposition(N)
    
    best_net = base_net[:]
    best_count = len(best_net)
    
    # Phase 1: Greedy removal to get a good starting point
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        for idx in range(len(base_net) - 1, -1, -1):
            if time.time() > deadline - 30:
                break
            new_net = base_net[:idx] + base_net[idx+1:]
            if _verify_fast(new_net) and _verify_full(new_net):
                base_net = new_net
                best_net = base_net[:]
                best_count = len(base_net)
                changed = True
                break
    
    # Phase 2: Genetic Algorithm
    population_size = 50
    population = [best_net[:]] * population_size
    # Diversify initial population
    for i in range(1, population_size):
        population[i] = _ga_mutation(best_net, rate=0.2)
    
    best_net = best_net[:]
    best_count = len(best_net)
    no_improve_generations = 0
    max_no_improve = 20
    
    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Evaluate fitness
        scores = []
        for net in population:
            if _verify_fast(net) and _verify_full(net):
                scores.append(-len(net))  # Lower count is better, so negate
            else:
                scores.append(-10000)  # Very bad fitness
        
        # Check for improvement
        gen_best_score = max(scores)
        if gen_best_score > -best_count:
            best_count = -gen_best_score
            best_idx = scores.index(gen_best_score)
            best_net = population[best_idx][:]
            no_improve_generations = 0
        else:
            no_improve_generations += 1
        
        # Early stop if stuck
        if no_improve_generations > max_no_improve:
            break
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best individual
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        new_population.append(population[best_idx][:])
        
        # Fill rest with crossover and mutation
        while len(new_population) < population_size:
            if time.time() > deadline - 5:
                break
            
            parent1 = _tournament_select(population, scores, k=3)
            parent2 = _tournament_select(population, scores, k=3)
            
            child = _ga_crossover(parent1, parent2)
            child = _ga_mutation(child, rate=0.15)
            
            # Cap size
            if len(child) > MAX_COMPARATORS:
                child = child[:MAX_COMPARATORS]
            
            new_population.append(child)
        
        population = new_population
    
    # Final validation
    if not _verify_full(best_net):
        best_net = _odd_even_transposition(N)
    
    return best_net
