"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import random
import math

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

def _simulated_annealing(start_network, deadline):
    """
    Simulated Annealing to find a smaller sorting network.
    """
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    
    # SA Parameters
    T = 1.0
    cooling_rate = 0.995
    
    # Pre-generate some random comparators for speed?
    # Or just generate on the fly.
    
    while time.time() < deadline and T > 0.01:
        # Generate a move
        move_type = random.random()
        
        neighbor = list(current)
        
        if move_type < 0.5:
            # Try Remove
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                delta_size = -1
            else:
                continue
        elif move_type < 0.8:
            # Try Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor.append((i, j))
            delta_size = 1
        else:
            # Try Swap
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    delta_size = 0
                else:
                    continue
            else:
                continue
        
        # Check validity
        if _is_sorting_network(neighbor):
            new_size = len(neighbor)
            if new_size < best_size:
                best = neighbor
                best_size = new_size
            
            # Acceptance criterion
            if delta_size < 0:
                # Improvement
                current = neighbor
            elif delta_size == 0:
                # Neutral (Swap)
                current = neighbor
            else:
                # Worse (Add)
                # Accept with probability exp(-delta_size / T)
                if random.random() < math.exp(-delta_size / T):
                    current = neighbor
        
        # Cool down
        T *= cooling_rate
        
    return best

def construct():
    # Start with odd-even transposition network
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best = list(seed_network)
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    # 1. Greedy reduction to get a good starting point
    best = _greedy_reduce(best, deadline)
    
    # 2. Simulated Annealing
    # We can run SA multiple times if time permits, or just one long run.
    # Given the time limit, one long run with restarts is better.
    
    while time.time() < deadline:
        # Restart SA from current best
        # Or restart from seed?
        # Restarting from best is local. Restarting from seed is global.
        # Let's do a mix: 50% restart from best, 50% from seed (with random perturbation)
        
        if random.random() < 0.5:
            start_net = list(best)
        else:
            start_net = list(seed_network)
            # Perturb seed
            for _ in range(random.randint(0, 10)):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            start_net = _greedy_reduce(start_net, deadline)
        
        candidate = _simulated_annealing(start_net, deadline)
        
        if len(candidate) < len(best):
            best = candidate
            
    return best
# EVOLVE-BLOCK-END
