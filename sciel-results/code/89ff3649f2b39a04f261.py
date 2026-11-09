"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

# EVOLVE-BLOCK-START
def _build_seed_network():
    """
    Build a seed sorting network using odd-even transposition.
    For N=17, this gives 136 comparators.
    """
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network

def _verify_sorting_network(network, n_wires):
    """
    Verify that the network sorts all 2^n_wires boolean inputs using the zero-one principle.
    Uses Python big integers as bitmasks for speed.
    """
    if not hasattr(_verify_sorting_network, '_initial_masks_cache'):
        _verify_sorting_network._initial_masks_cache = {}
    
    if n_wires not in _verify_sorting_network._initial_masks_cache:
        total_inputs = 1 << n_wires
        masks = []
        for w in range(n_wires):
            m = 0
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _verify_sorting_network._initial_masks_cache[n_wires] = masks
    
    masks = _verify_sorting_network._initial_masks_cache[n_wires][:]
    
    for i, j in network:
        if i >= j:
            return False
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    
    for w in range(n_wires - 1):
        if masks[w] & ~masks[w + 1]:
            return False
            
    return True

def _greedy_remove(network, start_time, deadline, direction='right'):
    """
    Perform greedy removal of comparators.
    direction: 'right' = try removing from end to start, 'left' = start to end
    Returns (final_network, removed_count)
    """
    removed = 0
    changed = True
    max_iterations = 1000 
    iterations = 0
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        if direction == 'right':
            indices = range(len(network) - 1, -1, -1)
        else:
            indices = range(len(network))
            
        for i in indices:
            if time.time() > deadline:
                return network, removed
            new_net = network[:i] + network[i+1:]
            if _verify_sorting_network(new_net, N):
                network = new_net
                removed += 1
                changed = True
                break
    return network, removed

def _local_search_step(network, start_time, deadline):
    """
    Perform a local search step involving moves and swaps.
    """
    best_net = list(network)
    best_size = len(network)
    
    attempts = 0
    max_attempts = 20 # Reduced to allow more frequent checks
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        # 1. Try moving a comparator to a random new position
        if len(best_net) > 1:
            i = random.randint(0, len(best_net) - 1)
            comp = best_net[i]
            
            # Remove it
            temp_net = best_net[:i] + best_net[i+1:]
            
            # Insert at random position
            j = random.randint(0, len(temp_net))
            new_net = temp_net[:j] + [comp] + temp_net[j:]
            
            # Quick verify
            if _verify_sorting_network(new_net, N):
                # Try to clean up
                cleaned_net, _ = _greedy_remove(new_net, start_time, deadline, direction='right')
                if len(cleaned_net) < best_size:
                    best_net = cleaned_net
                    best_size = len(cleaned_net)
        
        # 2. Try swapping two comparators
        if len(best_net) > 1:
            i = random.randint(0, len(best_net) - 2)
            new_net = best_net[:]
            new_net[i], new_net[i+1] = new_net[i+1], new_net[i]
            
            if _verify_sorting_network(new_net, N):
                cleaned_net, _ = _greedy_remove(new_net, start_time, deadline, direction='right')
                if len(cleaned_net) < best_size:
                    best_net = cleaned_net
                    best_size = len(cleaned_net)
                    
        # 3. Try replacing a comparator with a random one
        if len(best_net) > 0:
            idx = random.randint(0, len(best_net) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_net = best_net[:]
            new_net[idx] = (i, j)
            
            if _verify_sorting_network(new_net, N):
                cleaned_net, _ = _greedy_remove(new_net, start_time, deadline, direction='right')
                if len(cleaned_net) < best_size:
                    best_net = cleaned_net
                    best_size = len(cleaned_net)

    return best_net

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 45.0
    
    network = _build_seed_network()
    best_network = list(network)
    best_size = len(network)
    
    # Phase 1: Initial Greedy Removal
    network, _ = _greedy_remove(network, start_time, deadline, direction='right')
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    # Phase 2: Local Search Loop
    # We run local search steps until time runs out
    # Each step includes a greedy cleanup, so it's expensive but effective
    while time.time() < deadline - 2:
        # Perform a local search step
        net = _local_search_step(best_network, start_time, deadline)
        
        if len(net) < best_size:
            best_size = len(net)
            best_network = list(net)
        else:
            # If no improvement, we might be stuck. 
            # We can try a "kick" by adding a random comparator and then cleaning up?
            # Or just continue.
            pass

    return best_network

# EVOLVE-BLOCK-END
