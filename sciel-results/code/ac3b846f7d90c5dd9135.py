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

def _try_add_and_remove_two(network, start_time, deadline):
    """
    Try to add a random comparator, then remove two comparators.
    If successful, the network size decreases by 1.
    Returns the new network if improved, else None.
    """
    if time.time() > deadline:
        return None
        
    # Pick a random position to insert
    insert_idx = random.randint(0, len(network))
    # Pick a random comparator (i, j)
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    
    candidate = network[:insert_idx] + [(i, j)] + network[insert_idx:]
    
    if not _verify_sorting_network(candidate, N):
        return None
        
    # Now try to remove two comparators from this candidate
    # We do a limited greedy search for 2 removals
    best_net = candidate
    best_size = len(candidate)
    
    # Try to remove one
    removed_one = False
    for k in range(len(best_net) - 1, -1, -1):
        if time.time() > deadline:
            return None
        test_net = best_net[:k] + best_net[k+1:]
        if _verify_sorting_network(test_net, N):
            best_net = test_net
            best_size -= 1
            removed_one = True
            break
            
    if not removed_one:
        return None
        
    # Try to remove a second one from the result
    for k in range(len(best_net) - 1, -1, -1):
        if time.time() > deadline:
            return None
        test_net = best_net[:k] + best_net[k+1:]
        if _verify_sorting_network(test_net, N):
            best_net = test_net
            best_size -= 1
            break
            
    if best_size < len(network):
        return best_net
    return None

def _local_search_step(network, start_time, deadline):
    """
    Perform a limited local search: try swapping, moving, or 2-for-1.
    Returns the best network found within the time limit.
    """
    best_net = list(network)
    best_size = len(network)
    
    attempts = 0
    max_attempts = 20 # Reduced to allow more outer loop iterations
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        # Strategy 1: Random Swap of two comparators
        if len(best_net) > 1 and random.random() < 0.5:
            i = random.randint(0, len(best_net) - 1)
            j = random.randint(0, len(best_net) - 1)
            if i != j:
                new_net = best_net[:]
                new_net[i], new_net[j] = new_net[j], new_net[i]
                # Check if swap allows removal of 1
                if _verify_sorting_network(new_net, N):
                    # Try to remove one
                    for k in range(len(new_net) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        test_net = new_net[:k] + new_net[k+1:]
                        if _verify_sorting_network(test_net, N):
                            new_net = test_net
                            break
                    if len(new_net) < best_size:
                        best_net = new_net
                        best_size = len(new_net)
        else:
            # Strategy 2: 2-for-1 (Add then remove 2)
            candidate = _try_add_and_remove_two(best_net, start_time, deadline)
            if candidate and len(candidate) < best_size:
                best_net = candidate
                best_size = len(candidate)
                
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
    
    # Phase 1: Greedy removal from right
    network, _ = _greedy_remove(network, start_time, deadline, direction='right')
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    # Phase 2: Greedy removal from left
    if time.time() < deadline - 5:
        network2, _ = _greedy_remove(list(best_network), start_time, deadline, direction='left')
        if len(network2) < best_size:
            best_size = len(network2)
            best_network = list(network2)
    
    # Phase 3: Iterative Local Search with 2-for-1 moves
    while time.time() < deadline - 2:
        net = _local_search_step(best_network, start_time, deadline)
        
        if len(net) < best_size:
            best_size = len(net)
            best_network = list(net)
            
            # If improved, do a quick greedy cleanup
            if time.time() < deadline - 2:
                cleaned_net, _ = _greedy_remove(best_network, start_time, deadline, direction='right')
                if len(cleaned_net) < best_size:
                    best_size = len(cleaned_net)
                    best_network = list(cleaned_net)
        else:
            # If stuck, we might want to randomize a bit more or break
            # The loop condition handles termination
            pass

    return best_network

# EVOLVE-BLOCK-END
