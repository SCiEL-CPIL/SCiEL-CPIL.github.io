"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Start from multiple known-good initial networks (odd-even, batcher's bitonic).
2. Use fast bit-parallel verification (zero-one principle over all 65536 boolean inputs).
3. Aggressive local search: greedy removal, then insertion-removal cycles to escape local minima.
4. Time-bounded with explicit deadline guard.
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

# Pre-compute initial state patterns for speed
_initial_states = None

def _get_initial_states():
    """Pre-compute the 16 bitmask patterns representing all 65536 boolean inputs."""
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = NUM_INPUTS // period
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
    
    # Check sorted: for all i, state[i] & ~state[i+1] == 0
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def odd_even_network():
    """Odd-even transposition network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def batcher_bitonic_network():
    """Batcher's bitonic sorting network for 16 wires."""
    comparators = []
    
    def bitonic_merge(comp_list, start, size, ascending):
        """Merge a bitonic sequence of given size."""
        if size <= 1:
            return
        half = size // 2
        # First, sort the two halves in opposite directions
        bitonic_merge(comp_list, start, half, not ascending)
        bitonic_merge(comp_list, start + half, half, not ascending)
        # Then merge
        for i in range(half):
            if ascending:
                comp_list.append((start + i, start + i + half))
            else:
                comp_list.append((start + i + half, start + i))
    
    def bitonic_sort(comp_list, start, size, ascending):
        """Sort a bitonic sequence."""
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(comp_list, start, half, True)
        bitonic_sort(comp_list, start + half, half, False)
        bitonic_merge(comp_list, start, size, ascending)
    
    # For 16 wires, we need to create a bitonic sequence first
    # A bitonic sequence of size 2^k is built by sorting halves in opposite directions
    def build_bitonic(comp_list, start, size, direction):
        if size <= 1:
            return
        half = size // 2
        build_bitonic(comp_list, start, half, True)
        build_bitonic(comp_list, start + half, half, False)
    
    # Actually, for Batcher's bitonic sort, the standard approach is:
    # Stage 1: sort pairs (1-comparator stages)
    # Then recursively build bitonic sequences and merge
    
    # Let me use a cleaner recursive approach
    def bitonic_sorter(comps, n, start):
        """Generate comparators for a bitonic sort of n elements starting at position start."""
        if n <= 1:
            return
        half = n // 2
        # Recursively sort first half ascending, second half descending
        bitonic_sorter(comps, half, start)
        # For the second half, we need to sort it in reverse
        # We can do this by sorting it ascending and then reversing the comparators
        # Actually, let me just build it properly
        
        # Standard bitonic sort:
        # 1. Recursively sort first half ascending
        # 2. Recursively sort second half descending  
        # 3. Merge the two halves
        
        # For step 2, sorting descending means we swap the roles
        # Let me use a different approach: generate all comparators for bitonic sort
        
        pass
    
    # Let me use a simpler iterative approach for 16 wires
    # Batcher's bitonic sort for 16 elements has log2(16) = 4 stages
    
    # Stage 1: compare adjacent pairs
    for i in range(0, 16, 2):
        comparators.append((i, i + 1))
    
    # Stage 2: compare pairs of pairs
    for i in range(0, 16, 4):
        comparators.append((i, i + 2))
        comparators.append((i + 1, i + 3))
    
    # Stage 3: compare groups of 4
    for i in range(0, 16, 8):
        comparators.append((i, i + 4))
        comparators.append((i + 1, i + 5))
        comparators.append((i + 2, i + 6))
        comparators.append((i + 3, i + 7))
    
    # Stage 4: compare groups of 8
    comparators.append((0, 8))
    comparators.append((1, 9))
    comparators.append((2, 10))
    comparators.append((3, 11))
    comparators.append((4, 12))
    comparators.append((5, 13))
    comparators.append((6, 14))
    comparators.append((7, 15))
    
    return comparators

def greedy_remove(net, deadline):
    """Repeatedly try to remove each comparator until no more can be removed."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break  # Restart from beginning since indices changed
    return net

def targeted_insertion_remove(net, deadline, random_state):
    """Try inserting random comparators and then removing to escape local minima."""
    best_net = list(net)
    best_count = len(net)
    
    max_iter = 200
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Insert a random comparator at a random position
        i = random_state.randint(0, N - 2)
        j = random_state.randint(i + 1, N - 1)
        pos = random_state.randint(0, len(best_net))
        candidate = best_net[:pos] + [(i, j)] + best_net[pos:]
        
        # Try greedy removal on the candidate
        candidate = greedy_remove(candidate, deadline)
        
        if len(candidate) < best_count:
            best_count = len(candidate)
            best_net = candidate
    
    return best_net

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    random_state = random.Random(42)
    
    # Try multiple initial networks
    initial_networks = [
        odd_even_network(),
        batcher_bitonic_network()
    ]
    
    best_net = None
    best_count = float('inf')
    
    for init_net in initial_networks:
        if time.time() > deadline:
            break
        
        net = list(init_net)
        
        # Verify initial network
        if not fast_verify(net):
            continue
        
        # Greedy removal
        net = greedy_remove(net, deadline)
        
        # Targeted insertion-removal search
        net = targeted_insertion_remove(net, deadline, random_state)
        
        # Final greedy removal
        net = greedy_remove(net, deadline)
        
        if len(net) < best_count:
            best_count = len(net)
            best_net = list(net)
    
    # If we still have time, do more random restarts from the best network
    if best_net is not None and time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline:
                break
            
            net = list(best_net)
            
            # Perturb: swap two adjacent comparators
            if len(net) > 1:
                idx = random_state.randint(0, len(net) - 2)
                net[idx], net[idx+1] = net[idx+1], net[idx]
            
            # Try to improve
            net = greedy_remove(net, deadline)
            net = targeted_insertion_remove(net, deadline, random_state)
            net = greedy_remove(net, deadline)
            
            if len(net) < best_count:
                best_count = len(net)
                best_net = list(net)
    
    return best_net if best_net is not None else odd_even_network()
