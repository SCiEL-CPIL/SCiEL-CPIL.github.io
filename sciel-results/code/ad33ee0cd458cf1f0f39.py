import time
import random

N = 17
TOTAL_INPUTS = 1 << N

def _build_initial_masks():
    masks = [0] * N
    for m in range(TOTAL_INPUTS):
        for k in range(N):
            if (m >> k) & 1:
                masks[k] |= (1 << m)
    return masks

def _is_sorted(masks):
    for k in range(N - 1):
        if (masks[k] & ~masks[k + 1]) != 0:
            return False
    return True

def _compute_states(network, initial_masks):
    states = [initial_masks]
    current = initial_masks
    for i, j in network:
        mi = current[i]
        mj = current[j]
        new_m = current[:]
        new_m[i] = mi & mj
        new_m[j] = mi | mj
        current = new_m
        states.append(current)
    return states

def _backward_greedy(network, initial_masks):
    """
    Attempts to remove comparators from the end of the network.
    Returns the modified network.
    """
    if not network:
        return network
    
    states = _compute_states(network, initial_masks)
    C = len(network)
    
    # Check if last comparator is redundant
    # i.e., is state before last comparator (states[C-1]) already sorted?
    while C > 0:
        if _is_sorted(states[C-1]):
            # Remove last comparator
            network.pop()
            C -= 1
            # No need to recompute all states, just the last one changed?
            # Actually, removing the last comparator means the new final state is states[C-1] (old index C-1, new index C-1).
            # The states list is now too long by 1. We can just pop the last state.
            states.pop()
        else:
            break
            
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # 1. Seed
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    initial_masks = _build_initial_masks()
    
    # 2. Initial Backward Greedy
    network = _backward_greedy(network, initial_masks)
    best_network = network[:]
    best_len = len(network)
    
    # 3. Local Search
    # We will try to improve by swapping or inserting comparators.
    # After each mutation, we run backward greedy.
    
    # To make search more effective, we can also try "forward" deletion checks
    # but backward is much faster.
    
    for _ in range(2000):
        if time.time() >= deadline - 5:
            break
        
        current_net = best_network[:]
        
        # Move: Swap two random comparators
        if len(current_net) >= 2:
            k1 = random.randint(0, len(current_net) - 1)
            k2 = random.randint(0, len(current_net) - 1)
            if k1 != k2:
                current_net[k1], current_net[k2] = current_net[k2], current_net[k1]
            else:
                continue
        else:
            # If only 1 comparator, try inserting one
            pos = random.randint(0, len(current_net))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            current_net.insert(pos, (i, j))
            
        # Check validity
        states = _compute_states(current_net, initial_masks)
        if not _is_sorted(states[-1]):
            continue
            
        # Run Backward Greedy
        current_net = _backward_greedy(current_net, initial_masks)
        
        if len(current_net) < best_len:
            best_network = current_net
            best_len = len(current_net)
            
    # Final verification
    states = _compute_states(best_network, initial_masks)
    if _is_sorted(states[-1]):
        return best_network
    else:
        # Fallback
        network = []
        for phase in range(N):
            start = phase % 2
            for i in range(start, N - 1, 2):
                network.append((i, i + 1))
        return network

