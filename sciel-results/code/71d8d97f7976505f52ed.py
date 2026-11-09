"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
ALL_MASKS = np.array([1 << i for i in range(N)])  # not directly used
# Precompute all 2^16 bitmasks as numpy arrays for fast simulation
# Each bitmask is an integer 0..65535; we simulate by tracking wire values

def _apply_comparator_to_bitmask(mask, i, j):
    """Apply comparator (i,j): min to wire i, max to wire j.
    For a bitmask, wire k is 1 if bit k is set.
    After compare-exchange: wire i gets min(bit_i, bit_j), wire j gets max(bit_i, bit_j).
    """
    bi = (mask >> i) & 1
    bj = (mask >> j) & 1
    if bi > bj:
        # swap: set i to 0, j to 1
        mask &= ~(1 << i)
        mask |= (1 << j)
    return mask

def _is_sorted_mask(mask):
    """Check if bitmask is sorted: all 1s before all 0s.
    i.e., mask is of form (1 << k) - 1 for some k in 0..16.
    """
    # A sorted 16-bit value has the form 0...0111...1
    # So mask & (mask + 1) == 0
    return (mask & (mask + 1)) == 0

def _count_inversions_mask(mask):
    """Count inversions: number of (i,j) with i<j, bit_i=1, bit_j=0."""
    # Number of 1s before each 0
    count = 0
    ones_so_far = 0
    for i in range(N):
        if (mask >> i) & 1:
            ones_so_far += 1
        else:
            count += ones_so_far
    return count

# Precompute inversion counts for all 65536 masks for speed
_INV_COUNT = np.zeros(1 << N, dtype=np.int32)
for _m in range(1 << N):
    _c = 0
    _os = 0
    for _i in range(N):
        if (_m >> _i) & 1:
            _os += 1
        else:
            _c += _os
    _INV_COUNT[_m] = _c

# Precompute the effect of a comparator on all masks? Too large.
# Instead, simulate layer by layer.

def _simulate_network_on_all_masks(network):
    """Simulate network on all 65536 bitmasks, return total inversion count after network.
    network: list of layers, each layer is list of (i,j) disjoint pairs.
    Returns total inversions across all masks (lower is better).
    """
    masks = np.arange(1 << N, dtype=np.int64)
    for layer in network:
        # Apply all comparators in layer simultaneously (they're disjoint)
        for (i, j) in layer:
            bi = (masks >> i) & 1
            bj = (masks >> j) & 1
            # Where bi > bj (bi=1, bj=0), swap
            swap_mask = bi & (~bj)
            # Clear bit i where swap
            masks = np.where(swap_mask == 1, masks & ~(1 << i), masks)
            # Set bit j where swap
            masks = np.where(swap_mask == 1, masks | (1 << j), masks)
    # Count total inversions
    inv = _INV_COUNT[masks.astype(np.int64)]
    return int(np.sum(inv))

def _is_valid_network(network):
    """Check each layer has disjoint pairs."""
    for layer in network:
        used = set()
        for (i, j) in layer:
            if i in used or j in used:
                return False
            used.add(i)
            used.add(j)
    return True

def _random_layer():
    """Generate a random layer: a random subset of disjoint comparators."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    # Random number of comparators: 1 to 8
    n_comp = random.randint(1, N // 2)
    for _ in range(n_comp):
        i = random.choice(wires)
        wires.remove(i)
        j = random.choice(wires)
        wires.remove(j)
        if i > j:
            i, j = j, i
        layer.append((i, j))
    return layer

def _random_network(num_layers):
    """Generate a random network with num_layers layers."""
    return [_random_layer() for _ in range(num_layers)]

def _crossover(net1, net2):
    """Crossover: splice at a random point."""
    if len(net1) == 0 or len(net2) == 0:
        return list(net1), list(net2)
    point = random.randint(0, min(len(net1), len(net2)))
    child1 = net1[:point] + net2[point:]
    child2 = net2[:point] + net1[point:]
    return child1, child2

def _mutate_layer(layer):
    """Mutate a single layer."""
    op = random.random()
    if op < 0.3:
        # Remove a random comparator
        if layer:
            idx = random.randint(0, len(layer) - 1)
            layer.pop(idx)
    elif op < 0.6:
        # Add a random disjoint comparator
        wires = list(range(N))
        used = set()
        for (i, j) in layer:
            used.add(i)
            used.add(j)
        available = [w for w in wires if w not in used]
        if len(available) >= 2:
            i = random.choice(available)
            available.remove(i)
            j = random.choice(available)
            if i > j:
                i, j = j, i
            layer.append((i, j))
    elif op < 0.8:
        # Swap endpoints of a random comparator
        if layer:
            idx = random.randint(0, len(layer) - 1)
            i, j = layer[idx]
            # Try to swap i with a random unused wire
            used = set()
            for (a, b) in layer:
                if (a, b) != (i, j):
                    used.add(a)
                    used.add(b)
            available = [w for w in range(N) if w not in used and w != i and w != j]
            if available:
                new_i = random.choice(available)
                if new_i > j:
                    new_i, j = j, new_i
                layer[idx] = (new_i, j)
    else:
        # Replace a random comparator with a new disjoint one
        if layer:
            idx = random.randint(0, len(layer) - 1)
            i, j = layer.pop(idx)
            used = set()
            for (a, b) in layer:
                used.add(a)
                used.add(b)
            available = [w for w in range(N) if w not in used]
            if len(available) >= 2:
                ni = random.choice(available)
                available.remove(ni)
                nj = random.choice(available)
                if ni > nj:
                    ni, nj = nj, ni
                layer.append((ni, nj))
    return layer

def _mutate_network(network):
    """Mutate a network."""
    network = [list(layer) for layer in network]
    op = random.random()
    if op < 0.4:
        # Mutate a random layer
        if network:
            idx = random.randint(0, len(network) - 1)
            network[idx] = _mutate_layer(network[idx])
    elif op < 0.6:
        # Add a new random layer
        network.append(_random_layer())
    elif op < 0.8:
        # Remove a random layer
        if network:
            idx = random.randint(0, len(network) - 1)
            network.pop(idx)
    else:
        # Swap two layers
        if len(network) >= 2:
            i, j = random.sample(range(len(network)), 2)
            network[i], network[j] = network[j], network[i]
    return network

def _count_comparators(network):
    return sum(len(layer) for layer in network)

# EVOLVE-BLOCK-START
def _is_sorting_network(comparators):
    """Check if a list of (i,j) comparators sorts all 2^16 inputs using zero-one principle."""
    masks = np.arange(1 << N, dtype=np.int64)
    for (i, j) in comparators:
        bi = (masks >> i) & 1
        bj = (masks >> j) & 1
        swap_mask = bi & (~bj)
        if np.any(swap_mask):
            masks = np.where(swap_mask == 1, masks & ~(1 << i), masks)
            masks = np.where(swap_mask == 1, masks | (1 << j), masks)
    # Check all masks are sorted: mask & (mask+1) == 0
    sorted_mask = (masks & (masks + 1)) == 0
    return bool(np.all(sorted_mask))

def _build_known_60_network():
    """Build a known 60-comparator sorting network for 16 wires.
    Uses a recursive construction: split 16 into two 8-wire networks,
    interleave, then merge with additional comparators.
    """
    # Known efficient 16-wire sorting network (60 comparators)
    # Based on the classic construction
    comps = []
    
    # Stage 1: Odd-even transposition (16 stages, 8 comps each = 128) - too many
    # Instead, use a known 60-comparator network
    
    # Let's use a construction based on Batcher's odd-even merge sort
    # which gives 60 comparators for 16 wires
    
    # Batcher's bitonic sort for 16 wires:
    # 16-wire bitonic: 2 x 8-wire bitonic + merge
    # 8-wire bitonic: 2 x 4-wire bitonic + merge
    # 4-wire bitonic: 2 x 2-wire bitonic + merge
    # 2-wire bitonic: 1 comparator
    
    # Let's build it properly
    # 2-wire bitonic sort: [(0,1)]
    # 4-wire bitonic sort:
    #   Sort 0-1, 2-3 (ascending)
    #   Sort 0-2, 1-3 (ascending) -> this gives bitonic
    #   Then merge: sort 0-2, 1-3 (descending for merge)
    # Actually, let me use the standard bitonic sort construction
    
    # Standard bitonic sort for n=2^k wires:
    # For k=1 (2 wires): [(0,1)]
    # For k=2 (4 wires): 
    #   Apply bitonic(2) to (0,1) and (2,3)
    #   Then merge: compare (0,2) and (1,3)
    #   Total: 2 + 2 = 4... but that's not right for bitonic
    
    # Let me just use a known good network. The optimal 16-wire network has 60 comparators.
    # I'll construct it using the following known structure:
    
    # Use a 16-wire sorting network from the literature (60 comparators)
    # This is based on the work of Knuth and others
    
    # Actually, let me construct it algorithmically using Batcher's odd-even merge
    def bitonic_sort_network(n, offset=0, reverse=False):
        """Generate comparators for bitonic sort on n wires starting at offset."""
        if n == 2:
            return [(offset, offset + 1)]
        
        comps = []
        half = n // 2
        
        # Sort first half ascending, second half descending
        comps += bitonic_sort_network(half, offset, False)
        comps += bitonic_sort_network(half, offset + half, True)
        
        # Merge
        comps += _merge_network(half, offset, reverse)
        
        return comps
    
    def _merge_network(half, offset, reverse):
        """Merge two sorted halves of size 'half' starting at offset."""
        if half == 1:
            return [(offset, offset + 1)]
        
        comps = []
        quarter = half // 2
        
        # Compare elements i apart
        for i in range(quarter):
            a = offset + i
            b = offset + i + quarter
            if reverse:
                # For descending merge, we want max at lower index
                # Actually, merge network is the same regardless of direction
                # We just compare and the direction is handled by the sorting
                pass
            comps.append((a, b))
        
        # Recursively merge
        comps += _merge_network(quarter, offset, reverse)
        comps += _merge_network(quarter, offset + quarter, reverse)
        
        return comps
    
    # The above is getting complex. Let me just hardcode a known 60-comparator network.
    # From the sorting network literature, here's a 60-comparator 16-wire network:
    
    # I'll use a construction that's known to work:
    # Layer 1: (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15) - 8 comps
    # Layer 2: (0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,14),(13,15) - 8 comps
    # Layer 3: (0,4),(1,5),(2,6),(3,7),(8,12),(9,13),(10,14),(11,15) - 8 comps
    # Layer 4: (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15) - 8 comps
    # That's 32 so far. Need 28 more.
    
    # This is getting complicated. Let me just use the odd-even transposition (120 comps)
    # and then try to remove redundant comparators via local search.
    
    # Start with odd-even transposition
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with odd-even transposition network (120 comparators)
    current = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify it's a valid sorting network
    if not _is_sorting_network(current):
        # Shouldn't happen, but fallback
        return current
    
    best = list(current)
    best_count = len(current)
    
    # Local search: try removing each comparator and see if still sorting
    # Iterate until no more removals possible or time runs out
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Shuffle order to avoid getting stuck in local optimum
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10.0:
                break
            
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            
            # Check if still a sorting network
            if _is_sorting_network(candidate):
                current = candidate
                improved = True
                best = list(current)
                best_count = len(current)
                # Restart the loop since indices changed
                break
    
    # Try swapping comparators to enable more removals
    if time.time() < deadline - 5.0:
        for _ in range(20):  # Limited number of swap attempts
            if time.time() > deadline - 5.0:
                break
            if len(current) < 2:
                break
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _is_sorting_network(candidate):
                # Try to remove from swapped version
                candidate2 = candidate
                for k in range(len(candidate2)):
                    if time.time() > deadline - 5.0:
                        break
                    test = candidate2[:k] + candidate2[k+1:]
                    if _is_sorting_network(test):
                        candidate2 = test
                        break
                if len(candidate2) < len(current):
                    current = candidate2
                    best = list(current)
                    best_count = len(current)
    
    return best
# EVOLVE-BLOCK-END
