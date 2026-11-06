"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # seconds, leave 5s safety margin for 90s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for N=17 wires as a list of (i, j) pairs.
    Uses a hybrid approach:
    1. Start with a known good network (odd-even transposition or better).
    2. Apply local search / genetic algorithm to minimize comparator count.
    3. Verify using zero-one principle with bit-vector simulation.
    """
    start_time = time.time()
    
    # Generate initial candidate networks
    # Option 1: Odd-even transposition (136 comparators for 17 wires)
    # Option 2: Batcher's bitonic sort (more efficient)
    # Option 3: Random networks with pruning
    
    best_network = None
    best_count = float('inf')
    
    # Start with a good baseline: bitonic sort network for 17 wires
    # For non-power-of-2, we can pad to 32 or use a custom approach
    # Let's use a simpler approach: start with odd-even and improve
    
    # Baseline: odd-even transposition
    def odd_even_network(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    # Better baseline: use a known efficient construction
    # For 17 wires, we can use a merge-based approach
    # But let's start simple and improve via search
    
    # Generate multiple initial candidates
    candidates = []
    
    # Candidate 1: Odd-even transposition
    candidates.append(odd_even_network(N))
    
    # Candidate 2: Random network with sufficient depth
    random.seed(42)
    for _ in range(3):
        # Generate a random network with ~80-100 comparators
        depth = random.randint(10, 15)
        comps_per_layer = random.randint(5, 8)
        net = []
        for d in range(depth):
            # Generate non-conflicting comparators for this layer
            available = list(range(N))
            random.shuffle(available)
            used = set()
            for _ in range(comps_per_layer):
                if len(available) < 2:
                    break
                i, j = available[0], available[1]
                if i > j:
                    i, j = j, i
                if i in used or j in used:
                    continue
                net.append((i, j))
                used.add(i)
                used.add(j)
                available = [x for x in available if x not in used]
        candidates.append(net)
    
    # Verify and find the best valid candidate
    for net in candidates:
        if time.time() - start_time > DEADLINE:
            break
        if verify_network(net):
            count = len(net)
            if count < best_count:
                best_count = count
                best_network = net[:]
    
    if best_network is None:
        # Fallback: return the odd-even network (always valid)
        return odd_even_network(N)
    
    # Now try to improve via local search
    # Strategy: try removing comparators, or swapping them for better ones
    
    current = best_network[:]
    current_count = best_count
    
    # Local search: try removing one comparator at a time
    for _ in range(200):  # max iterations
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        # Try removing each comparator
        for idx in range(len(current)):
            if time.time() - start_time > DEADLINE:
                break
            
            # Remove comparator at idx
            candidate = current[:idx] + current[idx+1:]
            
            if verify_network(candidate):
                current = candidate
                current_count -= 1
                improved = True
                break  # Restart the removal loop
        
        if not improved:
            # Try swapping a comparator with a different one
            # Or try adding a comparator in a different position
            # For now, just stop if no removal works
            # Try a random mutation
            if len(current) > 0 and random.random() < 0.5:
                idx = random.randint(0, len(current) - 1)
                i, j = current[idx]
                # Try replacing with a random comparator
                new_i = random.randint(0, N - 2)
                new_j = random.randint(new_i + 1, N - 1)
                candidate = current[:]
                candidate[idx] = (new_i, new_j)
                if verify_network(candidate):
                    current = candidate
                    improved = True
            
            if not improved:
                # Try a more aggressive move: remove 2 comparators
                if len(current) >= 2:
                    idx1 = random.randint(0, len(current) - 1)
                    idx2 = random.randint(0, len(current) - 2)
                    if idx2 >= idx1:
                        idx2 += 1
                    candidate = current[:idx1] + current[idx1+1:idx2] + current[idx2+1:]
                    if verify_network(candidate):
                        current = candidate
                        current_count -= 2
                        improved = True
            
            if not improved:
                break
    
    return current


def verify_network(network):
    """
    Verify that the network sorts all inputs using the zero-one principle.
    We only need to check all 2^N Boolean inputs.
    For N=17, that's 131072 inputs.
    Use bit-vector simulation for efficiency.
    """
    n = len(network)
    if n == 0:
        # Empty network only sorts if N <= 1
        return N <= 1
    
    # For each Boolean input, check if output is sorted
    # Use numpy for speed
    # Represent each input as a 17-bit integer
    
    for mask in range(1 << N):
        # Initialize wires
        wires = [(mask >> i) & 1 for i in range(N)]
        
        # Apply each comparator
        for i, j in network:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        
        # Check if sorted (non-decreasing)
        for k in range(N - 1):
            if wires[k] > wires[k + 1]:
                return False
    
    return True


def verify_network_fast(network):
    """
    Faster verification using bit-parallelism.
    Process multiple inputs at once using numpy.
    """
    n = len(network)
    if n == 0:
        return N <= 1
    
    # Process in batches
    batch_size = 1024
    total_inputs = 1 << N
    
    for batch_start in range(0, total_inputs, batch_size):
        batch_end = min(batch_start + batch_size, total_inputs)
        batch = np.arange(batch_start, batch_end, dtype=np.uint32)
        
        # Initialize wires: shape (batch_size, N)
        wires = np.zeros((batch_end - batch_start, N), dtype=np.uint8)
        for i in range(N):
            wires[:, i] = (batch >> i) & 1
        
        # Apply each comparator
        for i, j in network:
            # For each input in batch, compare wires[i] and wires[j]
            # If wires[i] > wires[j], swap
            mask = wires[:, i] > wires[:, j]
            # Swap where mask is True
            temp = wires[mask, i].copy()
            wires[mask, i] = wires[mask, j]
            wires[mask, j] = temp
        
        # Check if sorted
        # For each input, check if all adjacent pairs are non-decreasing
        for k in range(N - 1):
            if np.any(wires[:, k] > wires[:, k + 1]):
                return False
    
    return True
# EVOLVE-BLOCK-END
