"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

_INITIAL_MASKS_CACHE = {}

def _get_initial_masks(n_wires):
    """
    Get or generate the initial bitmask representations for all 2^n inputs.
    mask[w] is an integer where bit k is 1 if input k has a 1 on wire w.
    """
    if n_wires not in _INITIAL_MASKS_CACHE:
        total_inputs = 1 << n_wires
        masks = []
        for w in range(n_wires):
            m = 0
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _INITIAL_MASKS_CACHE[n_wires] = masks
    return _INITIAL_MASKS_CACHE[n_wires]

def _verify_sorting_network(network, n_wires):
    """
    Verify that the network sorts all 2^n_wires boolean inputs using the zero-one principle.
    """
    if not network:
        return n_wires <= 1
        
    init_masks = _get_initial_masks(n_wires)
    masks = init_masks[:]
    
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

def _greedy_remove(network, deadline, direction='right'):
    """
    Perform greedy removal of comparators.
    """
    changed = True
    while changed:
        changed = False
        if direction == 'right':
            indices = range(len(network) - 1, -1, -1)
        else:
            indices = range(len(network))
            
        for i in indices:
            if time.time() > deadline:
                return network
            
            new_net = network[:i] + network[i+1:]
            if _verify_sorting_network(new_net, N):
                network = new_net
                changed = True
                break
    return network

def _local_search_step(network, deadline):
    """
    Perform a local search step.
    """
    best_net = list(network)
    best_size = len(best_net)
    
    attempts = 0
    max_attempts = 50
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        candidate = list(best_net)
        
        # Strategy 1: Swap adjacent disjoint comparators
        if len(candidate) > 1 and random.random() < 0.2:
            i = random.randint(0, len(candidate) - 2)
            c1, c2 = candidate[i], candidate[i+1]
            if len(set(c1) & set(c2)) == 0:
                candidate[i], candidate[i+1] = c2, c1
                if _verify_sorting_network(candidate, N):
                    cleaned = _greedy_remove(candidate, deadline, 'right')
                    if len(cleaned) < best_size:
                        best_net = cleaned
                        best_size = len(cleaned)
                        continue
        
        # Strategy 2: Try to remove a random comparator
        if len(candidate) > 0:
            rem_idx = random.randint(0, len(candidate) - 1)
            test_net = candidate[:rem_idx] + candidate[rem_idx+1:]
            if _verify_sorting_network(test_net, N):
                best_net = test_net
                best_size = len(best_net)
                continue

        # Strategy 3: Replace a comparator with a random one (rarely works, but high impact)
        if len(candidate) > 0 and random.random() < 0.1:
            idx = random.randint(0, len(candidate) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate[idx] = (i, j)
            
            if _verify_sorting_network(candidate, N):
                cleaned = _greedy_remove(candidate, deadline, 'right')
                if len(cleaned) < best_size:
                    best_net = cleaned
                    best_size = len(best_net)
                    continue

    return best_net

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    network = _build_seed_network()
    best_network = list(network)
    best_size = len(network)
    
    # Phase 1: Initial Greedy Removal
    network = _greedy_remove(network, deadline, 'right')
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
        
    network = _greedy_remove(best_network, deadline, 'left')
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)

    # Phase 2: Local Search
    while time.time() < deadline - 1.0:
        net = _local_search_step(best_network, deadline)
        
        if len(net) < best_size:
            best_size = len(net)
            best_network = list(net)
            
            if time.time() < deadline - 1.0:
                cleaned_net = _greedy_remove(best_network, deadline, 'right')
                if len(cleaned_net) < best_size:
                    best_size = len(cleaned_net)
                    best_network = list(cleaned_net)
                    
    return best_network
