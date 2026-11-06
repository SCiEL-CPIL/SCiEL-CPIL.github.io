"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute the input matrix and expected sorted outputs once at module load time
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a smaller sample set for fast validation during search
_SAMPLE_SIZE = 4096
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def _simulate_on_comparators(comparators, input_matrix):
    """Simulate network on given input matrix using vectorized operations."""
    state = input_matrix.copy()
    for i, j in comparators:
        a = state[:, i]
        b = state[:, j]
        # min goes to i, max goes to j
        mask = a > b
        state[:, i] = np.where(mask, b, a)
        state[:, j] = np.where(mask, a, b)
    return state

def is_sorted_network_full(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses precomputed input matrix for speed.
    """
    if len(comparators) > 4096:
        return False
    
    state = _simulate_on_comparators(comparators, _INPUTS)
    return np.array_equal(state, _SORTED_INPUTS)

def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs. Not sufficient for final validation,
    but much faster for guiding search.
    """
    if len(comparators) > 4096:
        return False
    
    state = _simulate_on_comparators(comparators, _SAMPLE_INPUTS)
    return np.array_equal(state, _SAMPLE_SORTED)

# Alias for backward compatibility
def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)

def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def try_remove_comparator(current, idx):
    """Try removing comparator at index idx. Returns (success, new_network)."""
    candidate = current.copy()
    candidate.pop(idx)
    if is_sorted_network_sample(candidate):
        if is_sorted_network_full(candidate):
            return True, candidate
    return False, current

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    left = odd_even_merge_sort_network(half)
    right = odd_even_merge_sort_network(n - half)
    
    # Offset right half comparators
    right = [(i + half, j + half) for i, j in right]
    
    # Merge the two halves
    merged = left + right
    # Odd-even merge: interleave comparators between the two halves
    for k in range(1, n, 2):
        if k < n:
            merged.append((k - 1, k))
    
    return merged

def odd_even_merge_sort_network_v2(n):
    """
    Better odd-even merge sort network construction.
    """
    comparators = []
    
    def build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = build(half)
        right = build(n - half)
        right = [(i + half, j + half) for i, j in right]
        
        # Merge phase: odd-even merge
        merge_comps = []
        # We need to merge two sorted sequences of length half and n-half
        # Using the odd-even merge procedure
        for phase in range(n - 1):
            start = phase % 2
            for i in range(start, n - 1, 2):
                # Compare wire i and i+1, but only if they span the boundary
                # Actually, for odd-even merge, we compare specific pairs
                pass
        
        # Simpler: just use the standard merge network
        # For merging two sorted lists of sizes a and b, we need a+b-1 comparators
        # arranged in a specific pattern
        a = half
        b = n - half
        
        # The merge network for two sorted inputs of size a and b
        # uses a+b-1 comparators in a specific arrangement
        merge = []
        for i in range(a + b - 1):
            # This is a simplified version; real odd-even merge is more complex
            # Let's just use a simple approach
            pass
        
        return left + right + merge
    
    return build(n)

# Use a known-good small network construction
def bitonic_sort_network(n):
    """
    Generate a bitonic sort network for n wires.
    n must be a power of 2 for the standard construction.
    For n=17, we'll pad to 32 or use a hybrid approach.
    Here we return the network for the largest power of 2 <= n,
    plus additional comparators for the remaining wires.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # For non-power-of-2, we'll use a different approach
    # Let's just build it recursively with the bitonic pattern
    comparators = []
    
    def bitonic(k, direction=1):
        """Generate bitonic merge network for k wires."""
        if k == 1:
            return []
        if k == 2:
            if direction == 1:
                return [(0, 1)]
            else:
                return [(1, 0)]  # swap
        
        half = k // 2
        # Recursively sort halves in opposite directions
        left = bitonic(half, 1)
        right = bitonic(k - half, -1)
        right = [(i + half, j + half) for i, j in right]
        
        # Merge: compare pairs (i, i+half)
        merge = []
        for i in range(half):
            j = i + half
            if direction == 1:
                merge.append((i, j))
            else:
                merge.append((j, i))
        
        return left + right + merge
    
    # For n=17, we can't use pure bitonic (needs power of 2)
    # Instead, use a layered approach
    return []

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin: finish well before 90s limit
    
    # Generate a valid seed network using odd-even transposition
    def oe_transposition(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    seed_network = oe_transposition(N)
    if not is_sorted_network_full(seed_network):
        # Fallback: simple insertion sort network
        seed_network = []
        for j in range(1, N):
            for i in range(j):
                seed_network.append((i, j))
    
    best_network = list(seed_network)
    best_len = len(best_network)
    
    # Convert comparators to layers (sets of non-overlapping comparators)
    def comps_to_layers(comps):
        layers = []
        used = set()
        for c in comps:
            if c[0] not in used and c[1] not in used:
                layers.append([c])
                used.add(c[0])
                used.add(c[1])
            else:
                # Start a new layer
                layers.append([c])
                used = {c[0], c[1]}
        return layers
    
    def layers_to_comps(layers):
        comps = []
        for layer in layers:
            comps.extend(layer)
        return comps
    
    # Fitness function: penalize invalid networks and count comparators
    def fitness(comps):
        if len(comps) > 4096:
            return 1e9
        if not is_sorted_network_sample(comps):
            # Count inversions in the sample to guide toward validity
            state = _simulate_on_comparators(comps, _SAMPLE_INPUTS)
            inversions = 0
            for row in state:
                for i in range(N):
                    for j in range(i+1, N):
                        if row[i] > row[j]:
                            inversions += 1
            return len(comps) + inversions * 10
        return len(comps)
    
    # Initialize population
    pop_size = 40
    population = [list(seed_network)]
    
    # Add some random variations
    rng = random.Random(42)
    for _ in range(pop_size - 1):
        candidate = list(seed_network)
        # Randomly remove some comparators
        num_remove = rng.randint(0, min(10, len(candidate) // 2))
        for _ in range(num_remove):
            if candidate:
                idx = rng.randint(0, len(candidate) - 1)
                candidate.pop(idx)
        population.append(candidate)
    
    # Genetic algorithm
    gen = 0
    max_gens = 200
    
    while time.time() < deadline - 5.0 and gen < max_gens:
        gen += 1
        
        # Evaluate fitness
        fitnesses = [fitness(ind) for ind in population]
        
        # Update best
        min_fit = min(fitnesses)
        best_idx = fitnesses.index(min_fit)
        if fitnesses[best_idx] < best_len and is_sorted_network_full(population[best_idx]):
            best_network = list(population[best_idx])
            best_len = len(best_network)
        
        # Selection: tournament selection
        def tournament_select(pop, fits, k=3):
            indices = rng.sample(range(len(pop)), k)
            best_i = min(indices, key=lambda i: fits[i])
            return list(pop[best_i])
        
        # Crossover: layer-based crossover
        def crossover(parent1, parent2):
            layers1 = comps_to_layers(parent1)
            layers2 = comps_to_layers(parent2)
            
            # Randomly pick a crossover point in terms of layers
            if not layers1 or not layers2:
                return list(parent1)
            
            point1 = rng.randint(0, len(layers1))
            point2 = rng.randint(0, len(layers2))
            
            child_layers = layers1[:point1] + layers2[point2:]
            child = layers_to_comps(child_layers)
            
            # Ensure child is not too long
            if len(child) > len(parent1) + len(parent2) + 10:
                # Truncate
                child = child[:len(parent1) + len(parent2) + 5]
            
            return child
        
        # Mutation: add/remove/swap comparators
        def mutate(ind):
            child = list(ind)
            num_mutations = rng.randint(1, 3)
            
            for _ in range(num_mutations):
                op = rng.choice(['remove', 'add', 'swap', 'move'])
                
                if op == 'remove' and child:
                    idx = rng.randint(0, len(child) - 1)
                    child.pop(idx)
                
                elif op == 'add':
                    i = rng.randint(0, N - 2)
                    j = rng.randint(i + 1, N - 1)
                    # Insert at random position
                    pos = rng.randint(0, len(child))
                    child.insert(pos, (i, j))
                
                elif op == 'swap' and len(child) >= 2:
                    i1 = rng.randint(0, len(child) - 1)
                    i2 = rng.randint(0, len(child) - 1)
                    if i1 != i2:
                        child[i1], child[i2] = child[i2], child[i1]
                
                elif op == 'move' and len(child) >= 2:
                    i1 = rng.randint(0, len(child) - 1)
                    i2 = rng.randint(0, len(child) - 1)
                    if i1 != i2:
                        comp = child.pop(i1)
                        child.insert(i2, comp)
            
            return child
        
        # Create next generation
        next_pop = [list(best_network)]  # Elitism
        
        while len(next_pop) < pop_size:
            if time.time() > deadline - 5.0:
                break
            
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            
            if rng.random() < 0.7:  # Crossover probability
                child = crossover(p1, p2)
            else:
                child = list(p1)
            
            child = mutate(child)
            
            # Limit size
            if len(child) > 200:
                child = child[:200]
            
            next_pop.append(child)
        
        population = next_pop
    
    # Final local search on best network
    current = list(best_network)
    
    def try_remove_local(idx):
        nonlocal current, best_network, best_len
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                return True
        return False
    
    i = 0
    while i < len(current) and time.time() < deadline - 1.0:
        if try_remove_local(i):
            pass
        else:
            i += 1
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return seed_network
        
    return best_network
# EVOLVE-BLOCK-END
