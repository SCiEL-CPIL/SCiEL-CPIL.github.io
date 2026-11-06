import time
import random

N = 17
TOTAL_INPUTS = 1 << N  # 131072

def _build_initial_masks():
    """
    Builds bitmask representation for the 0-1 principle.
    mask[k] is an integer where bit m is set iff input m has a 1 on wire k.
    Input m is represented by its binary representation of m (0 to 2^N-1).
    """
    masks = [0] * N
    for m in range(TOTAL_INPUTS):
        for k in range(N):
            if (m >> k) & 1:
                masks[k] |= (1 << m)
    return masks

def _is_sorted(masks):
    """
    Checks if the output masks represent a sorted network.
    For a sorted network, if wire k has a 1, wire k+1 must also have a 1.
    So mask[k] & ~mask[k+1] must be 0.
    """
    for k in range(N - 1):
        if (masks[k] & ~masks[k + 1]) != 0:
            return False
    return True

def _simulate_tail(masks_before, tail_comparators):
    """
    Simulates the effect of 'tail_comparators' on 'masks_before'.
    Returns the final masks.
    """
    masks = masks_before[:]
    for i, j in tail_comparators:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    return masks

def _build_states(network, initial_masks):
    """
    Pre-computes the state (masks) after each comparator in the network.
    """
    states = [initial_masks]
    current_masks = initial_masks
    for comp in network:
        i, j = comp
        mi = current_masks[i]
        mj = current_masks[j]
        new_masks = current_masks[:]
        new_masks[i] = mi & mj
        new_masks[j] = mi | mj
        current_masks = new_masks
        states.append(current_masks)
    return states

def _greedy_delete(network, initial_masks, deadline):
    """
    Attempts to delete redundant comparators from the network.
    Returns the modified network and its states.
    """
    states = _build_states(network, initial_masks)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for k in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            
            if k == len(network) - 1:
                if _is_sorted(states[k]):
                    network.pop(k)
                    states.pop()
                    changed = True
                    break
            else:
                tail = network[k+1:]
                final_masks = _simulate_tail(states[k], tail)
                if _is_sorted(final_masks):
                    network.pop(k)
                    states = _build_states(network, initial_masks)
                    changed = True
                    break
    return network, states

def construct():
    start_time = time.time()
    # Safety margin: 80s deadline to avoid 90s kill.
    deadline = start_time + 80.0
    
    # 1. Generate initial network: Odd-Even Transposition
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    # 2. Pre-compute initial masks
    initial_masks = _build_initial_masks()
    
    # 3. Initial greedy deletion
    network, _ = _greedy_delete(network, initial_masks, deadline)
    best_network = list(network)
    best_len = len(network)
    
    # 4. Local Search
    # We will perform random insertions, swaps, and moves, followed by greedy deletion.
    # To speed up, we might try to limit the scope of search or use heuristics.
    
    # Helper to check if a network is valid and get its length after deletion
    def _evaluate(net):
        if time.time() >= deadline - 2:
            return None, None
        states = _build_states(net, initial_masks)
        if not _is_sorted(states[-1]):
            return None, None
        # Greedy delete
        net_copy = list(net)
        net_copy, _ = _greedy_delete(net_copy, initial_masks, deadline - 2)
        return net_copy, len(net_copy)

    # Random seed for reproducibility
    random.seed(42)
    
    steps = 0
    while time.time() < deadline - 5:
        steps += 1
        current_net = list(best_network)
        
        # Choose a move
        move_type = random.randint(0, 3)
        
        if move_type == 0:
            # Insert a random comparator
            pos = random.randint(0, len(current_net))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            current_net.insert(pos, (i, j))
        elif move_type == 1:
            # Swap two comparators
            if len(current_net) >= 2:
                k1 = random.randint(0, len(current_net) - 1)
                k2 = random.randint(0, len(current_net) - 1)
                if k1 != k2:
                    current_net[k1], current_net[k2] = current_net[k2], current_net[k1]
        elif move_type == 2:
            # Move a comparator to a new position
            if len(current_net) >= 1:
                k1 = random.randint(0, len(current_net) - 1)
                comp = current_net.pop(k1)
                k2 = random.randint(0, len(current_net))
                current_net.insert(k2, comp)
        elif move_type == 3:
            # Change a comparator's wires
            if len(current_net) >= 1:
                k1 = random.randint(0, len(current_net) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                current_net[k1] = (i, j)
        
        # Evaluate
        new_net, new_len = _evaluate(current_net)
        
        if new_net is not None and new_len < best_len:
            best_network = new_net
            best_len = new_len
            
    return best_network

