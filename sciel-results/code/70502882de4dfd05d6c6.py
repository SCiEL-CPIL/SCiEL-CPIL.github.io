"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import random

_INITIAL_BITSETS = None

def _init_bitsets():
    """
    Initialize 18 bitsets, each of length 2^N.
    bitset[k] has bit i set if the i-th input (0 to 2^N-1) has a 1 in wire k.
    """
    total_bits = 1 << N
    bitsets = [0] * N
    
    for k in range(N):
        period = 1 << (k + 1)
        half_period = 1 << k
        # Mask for one period: lower half 0s, upper half 1s
        pattern = (1 << half_period) - 1
        pattern <<= half_period
        
        # Repeat pattern to fill total_bits
        # Use doubling
        current_val = pattern
        current_len = period
        
        while current_len < total_bits:
            current_val |= (current_val << current_len)
            current_len *= 2
            
        bitsets[k] = current_val

    return bitsets

def _get_initial_bitsets():
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is None:
        _INITIAL_BITSETS = _init_bitsets()
    return _INITIAL_BITSETS

def _is_sorting_network(comparators):
    """
    Check if network sorts all 2^N Boolean inputs using bitset simulation.
    """
    if not comparators:
        return N <= 1
        
    # Get a copy of the initial bitsets
    bitsets = list(_get_initial_bitsets())
    
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        bitsets[i] = bi & bj
        bitsets[j] = bi | bj
        
    # Check if sorted:
    # For all k < l, bitsets[k] must be a subset of bitsets[l].
    # i.e., bitsets[k] & ~bitsets[l] == 0.
    # It is sufficient to check adjacent wires: k and k+1.
    for k in range(N - 1):
        if bitsets[k] & ~bitsets[k + 1]:
            return False
            
    return True

def _greedy_reduce(comparators, deadline):
    """
    Greedy reduction: try removing each comparator, keep if valid.
    """
    current = list(comparators)
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        
        # Try removing comparator at i
        reduced = current[:i] + current[i+1:]
        
        if _is_sorting_network(reduced):
            current = reduced
        else:
            i += 1
    return current

def construct():
    # Start with odd-even transposition network
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best = list(seed_network)
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    # 1. Greedy reduction
    best = _greedy_reduce(best, deadline)
    
    # 2. If time remains, try random perturbations
    # Strategy: Take current best, add a random comparator, then greedy reduce.
    # If the result is smaller, keep it.
    
    while time.time() < deadline:
        candidate = list(best)
        
        # Add 1-3 random comparators
        num_add = random.randint(1, 3)
        for _ in range(num_add):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate.append((i, j))
            
        # Reduce
        reduced = _greedy_reduce(candidate, deadline)
        
        if len(reduced) < len(best):
            best = reduced
            
    return best
# EVOLVE-BLOCK-END
