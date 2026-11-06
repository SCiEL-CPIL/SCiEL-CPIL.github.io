"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# EVOLVE-BLOCK-START
def _build_initial_wire_states():
    """
    Build wire states as Python ints representing all 2^N inputs simultaneously.
    wire_state[w] is an int where bit k = (k >> w) & 1 for k in 0..2^N-1.
    """
    n = N
    wire_state = []
    for w in range(n):
        period = 1 << (w + 1)
        half = 1 << w
        # Build one period: bits [half, period) are set
        period_mask = 0
        for bit_pos in range(half, period):
            period_mask |= (1 << bit_pos)
        # Repeat across all 2^n inputs
        reps = 1 << (n - w - 1)
        mask = 0
        for rep in range(reps):
            mask |= (period_mask << (rep * period))
        wire_state.append(mask)
    return wire_state

def _verify_network_bitparallel(network):
    """
    Verify network sorts all 2^N Boolean inputs using bit-parallel zero-one principle.
    Each wire's state across all inputs is a single Python int.
    """
    wire_state = _build_initial_wire_states()
    for (i, j) in network:
        a = wire_state[i]
        b = wire_state[j]
        wire_state[i] = a & b   # min for 0/1
        wire_state[j] = a | b   # max for 0/1
    # Check sorted: for each adjacent pair, no input has wire k=1 and wire k+1=0
    for k in range(N - 1):
        if (wire_state[k] & ~wire_state[k + 1]) != 0:
            return False
    return True

def _generate_bitonic_network(n):
    """
    Generate a valid bitonic sorting network for n wires (power of 2).
    Returns list of (i, j) comparators with i < j.
    """
    network = []
    
    def bitonic_sort(k, offset, direction):
        """
        k: log2(size), so size = 2^k
        offset: starting wire index
        direction: 0 for ascending, 1 for descending
        """
        if k == 1:
            # Sort 2 elements
            network.append((offset, offset + 1))
            return
        half = 1 << (k - 1)
        # Compare elements at distance 'half'
        for i in range(half):
            if direction == 0:
                network.append((offset + i, offset + i + half))
            else:
                # For descending, we want max at lower index.
                # Comparator (a, b) puts min at a, max at b.
                # To put max at lower index 'a', we need the comparator to effectively
                # swap. But we can only do (i,j) with i<j putting min at i.
                # Trick: the bitonic merge for descending uses comparators
                # (offset+i+half, offset+i) but since we need i<j, we write
                # (offset+i, offset+i+half) and the "descending" is handled by
                # the recursive structure. Actually the standard approach:
                # For direction=1, the compare is still (offset+i, offset+i+half)
                # because the bitonic property ensures correctness.
                network.append((offset + i, offset + i + half))
        # Recursively sort halves
        bitonic_sort(k - 1, offset, 0)          # left half ascending
        bitonic_sort(k - 1, offset + half, 1)   # right half descending
    
    bitonic_sort(int(__import__('math').log2(n)), 0, 0)
    return network

def _generate_batcher_network(n):
    """
    Generate Batcher odd-even merge sort network.
    """
    network = []
    
    def compare(a, b):
        if a < b:
            network.append((a, b))
        else:
            network.append((b, a))
    
    def batcher_sort(k, offset, direction):
        """
        k: log2(size)
        offset: starting index
        direction: 0 ascending, 1 descending
        """
        if k == 1:
            i = offset
            j = offset + 1
            if direction == 0:
                network.append((i, j))
            else:
                network.append((i, j))  # same, handled by merge
            return
        
        half = 1 << (k - 1)
        # Recursively sort halves in opposite directions
        batcher_sort(k - 1, offset, 0)
        batcher_sort(k - 1, offset + half, 1)
        # Merge
        batcher_merge(k, offset, direction)
    
    def batcher_merge(k, offset, direction):
        """
        Merge two sorted halves of size 2^(k-1) starting at offset.
        """
        if k == 1:
            # Merge two single elements
            i = offset
            j = offset + 1
            network.append((i, j))
            return
        
        half = 1 << (k - 1)
        # Compare elements at distance half
        for i in range(half):
            a = offset + i
            b = offset + i + half
            network.append((a, b))
        # Recursively merge
        batcher_merge(k - 1, offset, direction)
        batcher_merge(k - 1, offset + half, direction)
    
    batcher_sort(int(__import__('math').log2(n)), 0, 0)
    return network

def _generate_known_good_network():
    """
    Return a known small 16-wire sorting network.
    We'll use the bitonic network as a strong starting point.
    """
    return _generate_bitonic_network(N)

def _local_search_sa(network, deadline):
    """
    Simulated annealing local search to minimize comparator count.
    Moves: remove a comparator, add a random comparator, swap two comparators.
    Uses bit-parallel verification.
    """
    start_time = time.time()
    best = list(network)
    best_len = len(best)
    current = list(best)
    current_len = len(current)
    temp = 5.0  # initial temperature
    cooling = 0.995
    min_temp = 0.01
    
    rng = random.Random(42)
    
    while temp > min_temp:
        if time.time() > deadline - 5.0:
            break
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.4 and len(current) > 1:
            # Remove a random comparator
            idx = rng.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            delta_cost = -1
        elif move_type < 0.6:
            # Add a random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            candidate = current + [(i, j)]
            delta_cost = 1
        elif move_type < 0.85 and len(current) >= 2:
            # Swap two comparators
            i = rng.randint(0, len(current) - 1)
            j = rng.randint(0, len(current) - 1)
            if i == j:
                i = (i + 1) % len(current)
            candidate = current[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            delta_cost = 0
        else:
            # Replace a random comparator with a different one
            idx = rng.randint(0, len(current) - 1)
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            candidate = current[:]
            candidate[idx] = (i, j)
            delta_cost = 0
        
        # Verify candidate
        if not _verify_network_bitparallel(candidate):
            # Invalid: always reject (or accept with very low prob)
            temp *= cooling
            continue
        
        cand_len = len(candidate)
        cost_diff = cand_len - current_len
        
        if cost_diff <= 0:
            # Better or equal: always accept
            current = candidate
            current_len = cand_len
            if cand_len < best_len:
                best = candidate
                best_len = cand_len
        else:
            # Worse: accept with probability exp(-cost_diff / temp)
            import math
            prob = math.exp(-cost_diff / temp)
            if rng.random() < prob:
                current = candidate
                current_len = cand_len
        
        temp *= cooling
    
    # Final greedy removal pass
    improved = True
    while improved:
        if time.time() > deadline - 2.0:
            break
        improved = False
        for idx in range(len(best)):
            if time.time() > deadline - 2.0:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_bitparallel(candidate):
                best = candidate
                improved = True
                break
    
    return best

def construct():
    """
    Construct a sorting network for 16 wires.
    Uses simulated annealing with bit-parallel verification.
    """
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Start with bitonic network (80 comparators)
    network = _generate_bitonic_network(N)
    
    # Verify it's correct (should be)
    if not _verify_network_bitparallel(network):
        # Fallback to a simpler construction
        network = _generate_batcher_network(N)
    
    # Simulated annealing local search
    network = _local_search_sa(network, deadline)
    
    return network
# EVOLVE-BLOCK-END
