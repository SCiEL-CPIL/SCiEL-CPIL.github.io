"""Optimized sorting network minimizer using big-integer bitwise verification."""
import time
import random

N = 17
_DEADLINE = 85  # seconds, safety margin before 90s kill
_INITIAL_WIRES = None

def _get_initial_wires():
    global _INITIAL_WIRES
    if _INITIAL_WIRES is None:
        wires = []
        for k in range(N):
            val = 0
            for m in range(1 << N):
                if (m >> k) & 1:
                    val |= (1 << m)
            wires.append(val)
        _INITIAL_WIRES = wires
    return _INITIAL_WIRES

def _verify_network(network):
    """Verify sorting network using big-integer bitwise operations."""
    wires = _get_initial_wires()[:]
    
    for i, j in network:
        wi = wires[i]
        wj = wires[j]
        wires[i] = wi & wj
        wires[j] = wi | wj
    
    for k in range(N - 1):
        if wires[k] & ~wires[k + 1]:
            return False
    return True

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _greedy_remove(network, start_time, deadline_margin=2):
    """Perform greedy removal passes on the network."""
    best = list(network)
    best_len = len(best)
    
    # Multiple passes with different directions
    for direction in ['forward', 'reverse', 'forward', 'reverse']:
        if time.time() - start_time > _DEADLINE - deadline_margin:
            break
        
        if direction == 'forward':
            i = 0
            while i < len(network):
                if time.time() - start_time > _DEADLINE - deadline_margin:
                    break
                candidate = network[:i] + network[i+1:]
                if _verify_network(candidate):
                    network = candidate
                    if len(network) < best_len:
                        best = list(network)
                        best_len = len(network)
                    # Do not increment i, check same position again
                else:
                    i += 1
        else:
            i = len(network) - 1
            while i >= 0:
                if time.time() - start_time > _DEADLINE - deadline_margin:
                    break
                candidate = network[:i] + network[i+1:]
                if _verify_network(candidate):
                    network = candidate
                    if len(network) < best_len:
                        best = list(network)
                        best_len = len(network)
                    # Do not decrement i, check same position again
                else:
                    i -= 1
    
    return best

def _random_perturb(network, start_time, deadline_margin=2):
    """Randomly perturb the network and try to improve."""
    best = list(network)
    best_len = len(network)
    
    while time.time() - start_time < _DEADLINE - deadline_margin:
        # Random move: remove, swap, or insert
        move_type = random.choice(['remove', 'swap', 'insert'])
        
        if move_type == 'remove' and len(network) > 1:
            idx = random.randint(0, len(network) - 1)
            candidate = network[:idx] + network[idx+1:]
        elif move_type == 'swap' and len(network) > 1:
            i, j = random.sample(range(len(network)), 2)
            candidate = network[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move_type == 'insert':
            # Insert a random comparator
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            idx = random.randint(0, len(network))
            candidate = network[:idx] + [(i, j)] + network[idx:]
        else:
            continue
            
        if _verify_network(candidate):
            # Accept if it doesn't increase size
            if len(candidate) <= len(best):
                best = candidate
                best_len = len(candidate)
                # After a successful move, do a quick greedy removal
                best = _greedy_remove(best, start_time, deadline_margin=1)
                best_len = len(best)

    return best

def construct():
    start_time = time.time()
    
    # Start with seed
    network = _seed_network()
    
    # Initial greedy removal
    network = _greedy_remove(network, start_time)
    
    # Random perturbation / Hill Climbing
    network = _random_perturb(network, start_time)
    
    # Final greedy removal
    network = _greedy_remove(network, start_time)
    
    # Final verification
    if not _verify_network(network):
        return _seed_network()
        
    return network
