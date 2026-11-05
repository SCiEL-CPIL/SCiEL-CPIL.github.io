"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Use bit-parallel zero-one verification for fast local search.
2. Start from a good baseline (odd-even transposition, 120 comparators).
3. Aggressive greedy removal to get down quickly.
4. Iterated Local Search:
   - Try to remove a comparator (randomized order).
   - If stuck, perform a "kick-start": insert a random comparator, then greedily remove.
   - Accept improvements or equal-size different structures to explore.
5. Keep the best network found within the time budget.
"""

import time
import random

N = 16
TOTAL_INPUTS = 1 << N  # 65536

# Pre-compute initial states for all 16 wires
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            # Pattern: 0...0 1...1 0...0 1...1 ...
            period_val = ((1 << half) - 1) << half
            num_periods = TOTAL_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def odd_even_network():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def try_remove_all(net, randomize=False):
    """Try to remove comparators until stuck. Optionally randomize removal order."""
    changed = True
    while changed:
        changed = False
        indices = list(range(len(net)))
        if randomize:
            random.shuffle(indices)
        else:
            indices.reverse()
        for idx in indices:
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break
    return net

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    random.seed(42)
    
    # Start from odd-even transposition
    net = odd_even_network()
    best_net = list(net)
    best_count = len(net)
    
    # Phase 1: Greedy removal until stuck
    net = try_remove_all(net, randomize=False)
    best_net = list(net)
    best_count = len(net)
    
    # Phase 2: Iterated Local Search
    while time.time() < deadline:
        # 1. Try to remove any comparator (Randomized Greedy)
        removed = False
        indices = list(range(len(net)))
        random.shuffle(indices)
        for idx in indices:
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                removed = True
                if len(net) < best_count:
                    best_count = len(net)
                    best_net = list(net)
                break
        
        if removed:
            continue
        
        # 2. If stuck, Kick-start with Insertion
        # Insert a random comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(net))
        candidate = net[:pos] + [(i, j)] + net[pos:]
        
        # Greedy remove from candidate (Randomized)
        candidate = try_remove_all(candidate, randomize=True)
        
        if len(candidate) < len(net):
            net = candidate
            if len(net) < best_count:
                best_count = len(net)
                best_net = list(net)
        elif len(candidate) == len(net):
            # Accept if different to explore the space
            if candidate != net:
                net = candidate

    return best_net
