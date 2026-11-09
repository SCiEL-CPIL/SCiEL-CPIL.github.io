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
            # Optimization: Use bit manipulation to generate masks faster if possible, 
            # but the loop is only 17 * 131072 ~ 2M ops once. 
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
    
    # Check sortedness: 0s then 1s. 
    # This means for any wire w, if it has a 1, all wires w+1... must have 1s.
    # Equivalently, masks[w] is a subset of masks[w+1].
    # masks[w] & ~masks[w+1] must be 0.
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
    # To prevent infinite loops or excessive time, limit total iterations in this call
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
            # Quick check: if comparator is redundant? No, we must verify.
            new_net = network[:i] + network[i+1:]
            if _verify_sorting_network(new_net, N):
                network = new_net
                removed += 1
                changed = True
                break
    return network, removed

def _local_search_step(network, start_time, deadline):
    """
    Perform a limited local search: try swapping adjacent or moving a comparator.
    Returns the best network found within the time limit.
    """
    best_net = list(network)
    best_size = len(network)
    
    # Try a few random swaps and moves
    # We limit the number of attempts to ensure we return quickly
    attempts = 0
    max_attempts = 50
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        # Move 1: Swap adjacent
        if len(best_net) > 1:
            i = random.randint(0, len(best_net) - 2)
            new_net = best_net[:]
            new_net[i], new_net[i+1] = new_net[i+1], new_net[i]
            
            # Try to remove from this new state
            # We do a shallow greedy remove here to save time
            # Instead of full greedy, just try removing a few random comparators
            candidate = list(new_net)
            
            # Try to remove one random comparator
            if len(candidate) > 0:
                rem_idx = random.randint(0, len(candidate) - 1)
                test_net = candidate[:rem_idx] + candidate[rem_idx+1:]
                if _verify_sorting_network(test_net, N):
                    candidate = test_net
            
            if len(candidate) < best_size:
                best_net = candidate
                best_size = len(candidate)
        
        # Move 2: Replace a comparator with a random one (if it still sorts)
        if len(best_net) > 0:
            idx = random.randint(0, len(best_net) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_net = best_net[:]
            new_net[idx] = (i, j)
            
            if _verify_sorting_network(new_net, N):
                # Try to remove one random comparator
                if len(new_net) > 0:
                    rem_idx = random.randint(0, len(new_net) - 1)
                    test_net = new_net[:rem_idx] + new_net[rem_idx+1:]
                    if _verify_sorting_network(test_net, N):
                        new_net = test_net
                
                if len(new_net) < best_size:
                    best_net = new_net
                    best_size = len(new_net)
    
    return best_net

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    # Set a strict deadline to ensure we return well before the 60s/90s limit.
    # 45s leaves 15s margin for overhead/return.
    deadline = start_time + 45.0
    
    # Start with seed network
    network = _build_seed_network()
    best_network = list(network)
    best_size = len(network)
    
    # Phase 1: Greedy removal from right
    network, _ = _greedy_remove(network, start_time, deadline, direction='right')
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    # Phase 2: Greedy removal from left on the result
    if time.time() < deadline - 5:
        network2, _ = _greedy_remove(list(best_network), start_time, deadline, direction='left')
        if len(network2) < best_size:
            best_size = len(network2)
            best_network = list(network2)
    
    # Phase 3: Local Search
    # Use remaining time for local search
    while time.time() < deadline - 2:
        # Perform a local search step
        net = _local_search_step(best_network, start_time, deadline)
        
        if len(net) < best_size:
            best_size = len(net)
            best_network = list(net)
            
            # If we improved, do a greedy cleanup on the new best
            if time.time() < deadline - 2:
                cleaned_net, _ = _greedy_remove(best_network, start_time, deadline, direction='right')
                if len(cleaned_net) < best_size:
                    best_size = len(cleaned_net)
                    best_network = list(cleaned_net)
        else:
            # If no improvement in a step, maybe wait a bit or break if stuck?
            # For now, just continue, but limit total steps to avoid infinite loop
            # The loop condition `time.time() < deadline` handles termination.
            pass

    return best_network

# EVOLVE-BLOCK-END
