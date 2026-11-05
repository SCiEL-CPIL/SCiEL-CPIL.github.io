"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17

def _simulate_bitmask(network, input_mask):
    """Simulate network on a single bitmask input. Returns final bitmask."""
    state = input_mask
    for i, j in network:
        bi = 1 << i
        bj = 1 << j
        if (state & bi) and not (state & bj):
            state = (state & ~bi) | bj
    return state

def _is_valid_fast(network):
    """Fast validity check using bitmask simulation for all 2^N inputs."""
    n_comps = len(network)
    if n_comps == 0:
        return N <= 1
    
    # Precompute bit masks for each wire
    # For each comparator (i, j), precompute the transformation
    # A comparator (i,j) on mask m: if bit i=1 and bit j=0, swap them
    # This is: m' = m XOR (bi XOR bj) * (m & bi) * (1 - m & bj)
    # But simpler: just do the bit ops
    
    # Generate all 2^N input masks as a numpy array
    all_inputs = np.arange(1 << N, dtype=np.int32)
    
    # Simulate each input through the network
    # For efficiency, process in batches or use the fact that we can
    # check the final state: sorted means all 1s are in the lowest positions
    # i.e., if there are k ones, the mask should be (1 << k) - 1
    
    states = all_inputs.copy()
    
    for i, j in network:
        bi = 1 << i
        bj = 1 << j
        # For each state: if bit i is set and bit j is not, swap
        # bit_i_set = (state >> i) & 1
        # bit_j_set = (state >> j) & 1
        # swap if bit_i_set=1 and bit_j_set=0
        # new_state = state XOR bi XOR bj (when swap happens)
        # swap_mask = ((state & bi) != 0) & ((state & bj) == 0)
        
        bit_i_set = (states & bi) != 0
        bit_j_set = (states & bj) != 0
        swap_mask = bit_i_set & ~bit_j_set
        states = np.where(swap_mask, states ^ bi ^ bj, states)
    
    # Check if all final states are sorted: mask should be (1<<k)-1 for some k
    # A mask is "sorted" (all 1s in lowest positions) iff:
    # mask & (mask + 1) == 0  (this is true for 0, 1, 3, 7, 15, ... i.e., 2^k - 1)
    # Actually: mask is of form 2^k - 1 iff mask & (mask + 1) == 0
    # But we also need to verify the count is preserved, which it is by construction
    
    # Check: for each state, is it of the form 2^k - 1?
    # mask & (mask + 1) == 0
    checks = (states & (states + 1)) == 0
    return bool(np.all(checks))

def _depth(network):
    """Compute parallel depth of network."""
    wire_depth = [0] * N
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth) if wire_depth else 0

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _try_remove_comparators(network, deadline):
    """Try to remove redundant comparators from the network."""
    current = list(network)
    
    # Try removing each comparator in random order, repeatedly
    while True:
        if time.time() > deadline:
            break
        
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        found_removal = False
        for idx in indices:
            if time.time() > deadline:
                break
            test_network = current[:idx] + current[idx+1:]
            if _is_valid_fast(test_network):
                current = test_network
                found_removal = True
                break
        
        if not found_removal:
            break
    
    return current

def _local_search(network, deadline):
    """Local search: try small perturbations."""
    current = list(network)
    best = list(current)
    best_count = len(current)
    
    while time.time() < deadline:
        trial = list(best)
        
        # Random perturbation: remove 1-2 comparators
        num_removals = random.randint(1, 2)
        indices_to_remove = random.sample(range(len(trial)), min(num_removals, len(trial)))
        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            trial.pop(idx)
        
        # Optionally add a random comparator
        if random.random() < 0.4 and len(trial) < 150:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            trial.append((i, j))
        
        if len(trial) < best_count and _is_valid_fast(trial):
            best = trial
            best_count = len(trial)
    
    return best

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Start with odd-even transposition
    network = _construct_odd_even_transposition()
    best_network = list(network)
    best_count = len(network)
    
    # Systematic removal
    improved = _try_remove_comparators(network, deadline)
    if len(improved) < best_count:
        best_network = improved
        best_count = len(improved)
    
    # Local search if time remains
    if time.time() < deadline - 5:
        search_deadline = deadline - 2
        improved2 = _local_search(best_network, search_deadline)
        if len(improved2) < best_count:
            best_network = improved2
            best_count = len(improved2)
    
    return best_network
