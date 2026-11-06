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
    Multiple passes for better reduction.
    """
    current = list(comparators)
    
    # Multiple passes to ensure thorough reduction
    for _ in range(3):
        i = 0
        while i < len(current):
            if time.time() > deadline:
                return current
            
            # Try removing comparator at i
            saved = current[i]
            current.pop(i)
            
            if _is_sorting_network(current):
                # Keep it removed
                pass
            else:
                # Restore
                current.insert(i, saved)
                i += 1
        
        # If we made progress, do another pass
        if i < len(current) and time.time() < deadline:
            continue
        break
    
    return current

def _final_polish(comparators, deadline):
    """
    Final high-precision polish: try all possible single-comparator removals
    and also try swapping adjacent comparators to find better orderings.
    """
    current = list(comparators)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing each comparator
        i = 0
        while i < len(current):
            if time.time() > deadline:
                return current
            
            saved = current[i]
            current.pop(i)
            
            if _is_sorting_network(current):
                improved = True
            else:
                current.insert(i, saved)
                i += 1
        
        # Try swapping adjacent comparators
        if time.time() > deadline:
            return current
            
        for i in range(len(current) - 1):
            if time.time() > deadline:
                return current
                
            current[i], current[i+1] = current[i+1], current[i]
            if _is_sorting_network(current):
                improved = True
            else:
                current[i], current[i+1] = current[i+1], current[i]
    
    return current

def _simulated_annealing(start_network, deadline):
    """
    Simulated Annealing to find a smaller sorting network.
    Moves: Remove, Add, Swap, Move, Replace, Slide.
    """
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    
    # SA Parameters - tuned for better exploration
    T = 3.0
    cooling_rate = 0.992  # Slower cooling for more exploration
    min_T = 0.03
    
    steps_since_improvement = 0
    max_stall_steps = 80
    stall_count = 0
    
    while time.time() < deadline and T > min_T:
        # Generate a move
        r = random.random()
        neighbor = list(current)
        move_valid = False
        delta = 0
        
        # Determine move type - refined probabilities
        # Remove: 30%, Slide: 15%, Replace: 20%, Move: 10%, Swap: 10%, Add: 15%
        if r < 0.30:
            # Try Remove
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                delta = -1
                move_valid = True
        elif r < 0.45:
            # Try Slide: Change (i, j) to (i+1, j+1) or (i-1, j-1)
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                i, j = neighbor[idx]
                # Try shifting up or down
                if random.random() < 0.5:
                    # Shift down (increase indices)
                    if i + 1 < N - 1 and j + 1 < N:
                        ni, nj = i + 1, j + 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
                else:
                    # Shift up (decrease indices)
                    if i - 1 >= 0 and j - 1 > 0:
                        ni, nj = i - 1, j - 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
        elif r < 0.65:
            # Try Replace: change the pair of a random comparator
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                # Bias towards adjacent or short distance for better exploration
                if random.random() < 0.7:
                    dist = random.randint(1, 4)
                else:
                    dist = random.randint(1, N - 1)
                a = random.randint(0, N - 1 - dist)
                b = a + dist
                neighbor[idx] = (a, b)
                delta = 0
                move_valid = True
        elif r < 0.75:
            # Try Move (move a comparator to a new position)
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    comp = neighbor.pop(idx1)
                    neighbor.insert(idx2, comp)
                    delta = 0
                    move_valid = True
        elif r < 0.85:
            # Try Swap (exchange two comparators)
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    delta = 0
                    move_valid = True
        else:
            # Try Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor.append((i, j))
            delta = 1
            move_valid = True
        
        if not move_valid:
            T *= cooling_rate
            continue

        # Check validity
        if _is_sorting_network(neighbor):
            new_size = len(neighbor)
            
            if new_size < best_size:
                best = neighbor
                best_size = new_size
                steps_since_improvement = 0
                stall_count = 0
            else:
                steps_since_improvement += 1
            
            # Acceptance criterion
            if delta < 0:
                # Improvement
                current = neighbor
            elif delta == 0:
                # Neutral - always accept to explore
                current = neighbor
            else:
                # Worse (Add) - accept with probability
                if random.random() < math.exp(-1.5 / T):
                    current = neighbor
        
        # Stall handling: If stuck for a while, kick the temperature up and perturb
        if steps_since_improvement > max_stall_steps:
            stall_count += 1
            T = max(T * 1.8, 0.8)
            steps_since_improvement = 0
            
            # Perturb current to escape local minimum
            if len(current) > 0 and random.random() < 0.5:
                # Add a few random comparators to break out
                num_perturb = random.randint(2, 5)
                for _ in range(num_perturb):
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    current.append((i, j))
                # Quick reduce to get back to valid state
                if time.time() < deadline:
                    quick_deadline = min(deadline, time.time() + 1)
                    current = _greedy_reduce(current, quick_deadline)
        
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
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # 1. Initial Greedy reduction to get a compact baseline
    best = _greedy_reduce(best, deadline)
    
    # 2. Simulated Annealing loops with diverse restarts
    sa_count = 0
    while time.time() < deadline:
        sa_count += 1
        
        # Decide starting point for SA - more diverse strategy
        r = random.random()
        if r < 0.5:
            # Local search: start from best
            start_net = list(best)
        elif r < 0.75:
            # Global search: start from seed, perturb, reduce
            start_net = list(seed_network)
            # Add some random noise to break symmetry
            num_noise = random.randint(8, 20)
            for _ in range(num_noise):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            
            # Reduce the perturbed seed to make it a reasonable starting point
            reduce_deadline = min(deadline, time.time() + 3)
            start_net = _greedy_reduce(start_net, reduce_deadline)
            
            if time.time() > deadline:
                break
        else:
            # Hybrid: start from best with random perturbations
            start_net = list(best)
            num_perturb = random.randint(3, 8)
            for _ in range(num_perturb):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            
            # Quick reduce
            reduce_deadline = min(deadline, time.time() + 2)
            start_net = _greedy_reduce(start_net, reduce_deadline)
            
            if time.time() > deadline:
                break
        
        # Run SA
        candidate = _simulated_annealing(start_net, deadline)
        
        if len(candidate) < len(best):
            best = candidate
            # Quick greedy reduce after improvement
            if time.time() < deadline:
                quick_deadline = min(deadline, time.time() + 2)
                best = _greedy_reduce(best, quick_deadline)
    
    # 3. Final polish pass
    if time.time() < deadline:
        polish_deadline = min(deadline, time.time() + 3)
        best = _final_polish(best, polish_deadline)
    
    return best
# EVOLVE-BLOCK-END
