"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

# EVOLVE-BLOCK-START
import time
import random

N = 17

def _get_initial_masks(n_wires):
    if not hasattr(_get_initial_masks, '_cache'):
        _get_initial_masks._cache = {}
    if n_wires not in _get_initial_masks._cache:
        total_inputs = 1 << n_wires
        masks = []
        for w in range(n_wires):
            m = 0
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _get_initial_masks._cache[n_wires] = masks
    return _get_initial_masks._cache[n_wires][:]

def _verify_sorting_network(network, n_wires):
    masks = _get_initial_masks(n_wires)
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
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network

def _greedy_remove(network, deadline):
    removed = 0
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network, removed
            new_net = network[:i] + network[i+1:]
            if _verify_sorting_network(new_net, N):
                network = new_net
                removed += 1
                changed = True
                break
    return network, removed

def _local_search_step(network, deadline):
    best_net = list(network)
    best_size = len(network)
    
    while time.time() < deadline:
        new_net = best_net[:]
        idx = random.randint(0, len(new_net) - 1)
        i, j = new_net[idx]
        
        move_type = random.randint(0, 2)
        if move_type == 0:
            new_i = random.randint(0, j - 1)
            new_net[idx] = (new_i, j)
        elif move_type == 1:
            src = idx
            dst = random.randint(0, len(new_net) - 1)
            comp = new_net[src]
            del new_net[src]
            if dst > src: dst -= 1
            new_net.insert(dst, comp)
        else:
            new_j = random.randint(i + 1, N - 1)
            new_net[idx] = (i, new_j)
            
        if not _verify_sorting_network(new_net, N):
            continue
            
        cleaned_net, _ = _greedy_remove(new_net, deadline)
        
        if len(cleaned_net) < best_size:
            best_net = cleaned_net
            best_size = len(best_net)
            
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    network = _build_seed_network()
    best_network = list(network)
    best_size = len(network)
    
    network, _ = _greedy_remove(network, deadline)
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
        
    while time.time() < deadline - 1:
        net = _local_search_step(best_network, deadline)
        if len(net) < best_size:
            best_size = len(net)
            best_network = list(net)
            
    return best_network

# EVOLVE-BLOCK-END
