"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Start from multiple known constructions (odd-even, bitonic, shell, merge) and aggressively optimize.
2. Use local search with a focus on structural changes: swap, move, insert/remove.
3. Use the zero-one principle for verification with bit-parallel simulation for speed.
4. Incorporate a Max-SAT-like iterative constraint satisfaction approach:
   - Identify "critical" comparators that are essential for sorting certain inputs.
   - Try to replace them with different comparators that might enable further reductions.
   - Use a greedy removal pass after each modification to minimize the network.

The key insight from the inspiration is to use a more sophisticated local search that
doesn't just remove comparators but also swaps and moves them, and uses multiple
starting points. The current program already has some of this, but we can improve
the search strategy and add more structural modifications.

We'll also add a "constraint-based" search where we identify which comparators are
necessary for specific inputs and try to find alternative configurations that satisfy
the same constraints with fewer comparators.
"""

import time
import random
import math

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            # For input j (0..65535), wire i has value (j >> i) & 1
            # Build a 65536-bit integer where bit j = (j >> i) & 1
            # Pattern: period 2^(i+1), lower half 0, upper half 1
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
    
    # Unrolled check for N=16 to reduce loop overhead
    if (state[0] & ~state[1]) or (state[1] & ~state[2]) or (state[2] & ~state[3]) or (state[3] & ~state[4]) or \
       (state[4] & ~state[5]) or (state[5] & ~state[6]) or (state[6] & ~state[7]) or (state[7] & ~state[8]) or \
       (state[8] & ~state[9]) or (state[9] & ~state[10]) or (state[10] & ~state[11]) or (state[11] & ~state[12]) or \
       (state[12] & ~state[13]) or (state[13] & ~state[14]) or (state[14] & ~state[15]):
        return False
    return True

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires (n must be power of 2).
    Returns a list of (i, j) comparator pairs.
    """
    if n <= 1:
        return []
    
    comparators = []
    
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    
    bitonic_sort(n, 0, True)
    return comparators

def odd_even_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def shell_sort_network():
    """Shell sort sorting network with standard gaps."""
    comparators = []
    gaps = [8, 4, 2, 1]
    for gap in gaps:
        for i in range(gap, N):
            comparators.append((i - gap, i))
    return comparators

def merge_sort_network(n):
    """Merge sort sorting network."""
    if n <= 1:
        return []
    comparators = []
    
    def merge_sort(size, start):
        if size <= 1:
            return
        half = size // 2
        merge_sort(half, start)
        merge_sort(half, start + half)
        merge(half, start)
    
    def merge(size, start):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            comparators.append((a, b))
        merge(size // 2, start)
        merge(size // 2, start + size)
    
    merge_sort(n, 0)
    return comparators

def compute_states(net):
    """
    Returns (states, final_state) where states[k] is the state of all wires
    before comparator k is applied. states[0] is the initial state.
    """
    states = [None] * (len(net) + 1)
    current_state = _get_initial_states()[:]
    states[0] = current_state
    
    for k, (i, j) in enumerate(net):
        next_state = current_state[:]
        min_val = current_state[i] & current_state[j]
        max_val = current_state[i] | current_state[j]
        next_state[i] = min_val
        next_state[j] = max_val
        states[k+1] = next_state
        current_state = next_state
        
    return states, current_state

def remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns (new_net, final_state).
    """
    if not net:
        return [], _get_initial_states()[:]
        
    states, final_state = compute_states(net)
    to_remove = set()
    
    for k, (i, j) in enumerate(net):
        # Check if wire i <= wire j for all inputs before this comparator
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
    if not to_remove:
        return net, final_state
        
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net, final_state

def is_sorted_state(state):
    """Check if the final state is sorted (non-decreasing)."""
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def greedy_remove_fast(net, deadline):
    """
    Fast greedy removal: try removing each comparator; if removal keeps
    the network valid, remove it. Uses incremental verification.
    Returns the reduced network.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try from the end to the beginning
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break
    return net

def greedy_remove_with_states(net, deadline):
    """
    Greedy removal that recomputes states each pass and removes all no-ops.
    More efficient than one-by-one re-verification.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Compute states and find all no-ops
        if not net:
            break
        states, final_state = compute_states(net)
        to_remove = set()
        for k, (i, j) in enumerate(net):
            if states[k][i] & ~states[k][j] == 0:
                to_remove.add(k)
        
        if to_remove:
            new_net = [net[k] for k in range(len(net)) if k not in to_remove]
            net = new_net
            changed = True
    
    return net

def identify_critical_comparators(net):
    """
    Identify comparators that are essential for sorting certain inputs.
    A comparator is critical if removing it causes the network to fail
    for at least one input.
    Returns a set of indices of critical comparators.
    """
    critical = set()
    for idx in range(len(net)):
        candidate = net[:idx] + net[idx+1:]
        if not fast_verify(candidate):
            critical.add(idx)
    return critical

def try_swap(net, idx1, idx2, deadline):
    """Try swapping two comparators."""
    if idx1 == idx2:
        return net
    candidate = list(net)
    candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
    if fast_verify(candidate):
        return greedy_remove_fast(candidate, deadline)
    return None

def try_move(net, from_idx, to_idx, deadline):
    """Try moving a comparator to a different position."""
    if from_idx == to_idx:
        return net
    comp = net[from_idx]
    candidate = net[:from_idx] + net[from_idx+1:]
    if to_idx > from_idx:
        to_idx -= 1
    candidate = candidate[:to_idx] + [comp] + candidate[to_idx:]
    if fast_verify(candidate):
        return greedy_remove_fast(candidate, deadline)
    return None

def try_insert_remove(net, i, j, pos, deadline):
    """Try inserting a comparator and then removing others."""
    candidate = net[:pos] + [(i, j)] + net[pos:]
    candidate = greedy_remove_fast(candidate, deadline)
    return candidate

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to leave 5s margin for overhead
    random.seed(42)
    
    best_net = None
    best_count = float('inf')
    
    # Generate multiple starting points
    start_nets = []
    
    # 1. Bitonic sorter
    bn = bitonic_sorter(N)
    bn_clean, bn_final = remove_redundant(bn)
    if is_sorted_state(bn_final):
        start_nets.append(bn_clean)
    
    # 2. Odd-even transposition
    oen = odd_even_network()
    oen_clean, oen_final = remove_redundant(oen)
    if is_sorted_state(oen_final):
        start_nets.append(oen_clean)
    
    # 3. Shell sort
    sn = shell_sort_network()
    sn_clean, sn_final = remove_redundant(sn)
    if is_sorted_state(sn_final):
        start_nets.append(sn_clean)
    
    # 4. Merge sort
    mn = merge_sort_network(N)
    mn_clean, mn_final = remove_redundant(mn)
    if is_sorted_state(mn_final):
        start_nets.append(mn_clean)
    
    # Process each starting network
    for net in start_nets:
        if time.time() > deadline:
            break
        
        # First, do a greedy removal pass
        net = greedy_remove_with_states(net, deadline)
        
        if len(net) < best_count:
            best_count = len(net)
            best_net = list(net)
        
        # Local search with structural modifications
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            op_type = random.random()
            
            if op_type < 0.3:
                # Swap two comparators
                if len(net) < 2:
                    continue
                idx1 = random.randint(0, len(net) - 1)
                idx2 = random.randint(0, len(net) - 1)
                candidate = try_swap(net, idx1, idx2, deadline)
                if candidate is not None and len(candidate) < len(net):
                    net = candidate
                    if len(net) < best_count:
                        best_count = len(net)
                        best_net = list(net)
            
            elif op_type < 0.6:
                # Move a comparator
                if len(net) < 2:
                    continue
                from_idx = random.randint(0, len(net) - 1)
                to_idx = random.randint(0, len(net) - 1)
                candidate = try_move(net, from_idx, to_idx, deadline)
                if candidate is not None and len(candidate) < len(net):
                    net = candidate
                    if len(net) < best_count:
                        best_count = len(net)
                        best_net = list(net)
            
            elif op_type < 0.85:
                # Insert a comparator and remove others
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(net))
                candidate = try_insert_remove(net, i, j, pos, deadline)
                if candidate is not None and len(candidate) < len(net):
                    net = candidate
                    if len(net) < best_count:
                        best_count = len(net)
                        best_net = list(net)
            
            else:
                # Critical comparator analysis: try to replace critical comparators
                if len(net) > 0:
                    critical = identify_critical_comparators(net)
                    if critical:
                        crit_idx = random.choice(list(critical))
                        old_comp = net[crit_idx]
                        # Try replacing with a different comparator
                        i = random.randint(0, N - 2)
                        j = random.randint(i + 1, N - 1)
                        candidate = net[:crit_idx] + [(i, j)] + net[crit_idx+1:]
                        candidate = greedy_remove_fast(candidate, deadline)
                        if len(candidate) < len(net):
                            net = candidate
                            if len(net) < best_count:
                                best_count = len(net)
                                best_net = list(net)
    
    if best_net is None:
        return odd_even_network()
        
    return best_net
