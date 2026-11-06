"""
Size-efficient sorting networks on 17 wires – CSP + Simulated Annealing hybrid.
Uses constraint-driven comparator selection followed by stochastic local search.
"""

import time
import random
import math

N = 17
TOTAL_INPUTS = 1 << N  # 131072

# Precompute initial state bitmasks for verification
INITIAL_STATE = [0] * N
for i in range(N):
    block = 1 << i
    val = 0
    for m in range(block, TOTAL_INPUTS, 2 * block):
        bits_to_set = (1 << block) - 1
        val |= bits_to_set << m
    INITIAL_STATE[i] = val


def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1

    state = list(INITIAL_STATE)
    st = state
    for i, j in network:
        si = st[i]
        sj = st[j]
        st[i] = si & sj
        st[j] = si | sj
    
    # Check sortedness
    for i in range(N - 1):
        if st[i] & ~st[i + 1]:
            return False
    return True


def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _remove_redundant_comparators(network, time_limit):
    """
    Try to remove each comparator if the network remains valid without it.
    Iterates repeatedly until no more can be removed or time runs out.
    """
    start_time = time.time()
    current = list(network)
    changed = True
    
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        n = len(current)
        for idx in range(n):
            if (time.time() - start_time) >= time_limit:
                break
            # Try removing comparator at idx
            # Optimization: Skip if we can't afford the check
            candidate = current[:idx] + current[idx + 1:]
            if _is_valid(candidate):
                current = candidate
                changed = True
                break # Restart loop from beginning as indices shifted
    return current


def _simulated_annealing(network, time_limit):
    """
    Simulated Annealing to minimize comparator count.
    """
    start_time = time.time()
    current_network = list(network)
    best_network = list(current_network)
    best_len = len(current_network)
    current_len = best_len

    T = 2.0
    cooling_rate = 0.999
    min_T = 0.1

    while (time.time() - start_time) < time_limit:
        if T < min_T:
            T = min_T

        move_type = random.random()

        if move_type < 0.5:
            # Delete a random comparator (High probability to reduce size)
            if len(current_network) > 10:
                idx = random.randint(0, len(current_network) - 1)
                candidate = current_network[:idx] + current_network[idx + 1:]
                new_len = current_len - 1
            else:
                continue
        elif move_type < 0.7:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(current_network))
            candidate = current_network[:idx] + [(i, j)] + current_network[idx:]
            new_len = current_len + 1
        elif move_type < 0.9:
            # Modify a comparator (change wires)
            if len(current_network) == 0:
                continue
            idx = random.randint(0, len(current_network) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = list(current_network)
            candidate[idx] = (i, j)
            new_len = current_len
        else:
            # Swap two comparators
            if len(current_network) < 2:
                continue
            idx1 = random.randint(0, len(current_network) - 1)
            idx2 = random.randint(0, len(current_network) - 1)
            if idx1 == idx2:
                continue
            candidate = list(current_network)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            new_len = current_len

        if not _is_valid(candidate):
            continue

        delta_cost = new_len - current_len
        if delta_cost < 0:
            current_network = candidate
            current_len = new_len
            if current_len < best_len:
                best_network = list(current_network)
                best_len = current_len
        else:
            if delta_cost == 0:
                if random.random() < 0.5:
                    current_network = candidate
            else:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    current_network = candidate
                    current_len = new_len

        T *= cooling_rate

    return best_network


def construct():
    """
    Construct a sorting network with as few comparators as possible.
    Uses Odd-Even Transposition seed + Redundancy Removal + SA Refinement.
    """
    start_time = time.time()
    # 60s limit, use 55s to be safe
    deadline = start_time + 55.0 

    # Phase 1: Start with a known valid network (Odd-Even Transposition)
    # This is much faster and more reliable than the previous CSP greedy approach
    network = _construct_odd_even_transposition()
    
    best_network = list(network)
    best_len = len(network)

    # Phase 2: Remove redundant comparators
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Use most of the time for redundancy removal initially as it's very effective
        network = _remove_redundant_comparators(network, remaining_time - 10)
        if len(network) < best_len:
            best_network = list(network)
            best_len = len(network)

    # Phase 3: Simulated Annealing refinement
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        network = _simulated_annealing(network, remaining_time - 1)
        if len(network) < best_len:
            best_network = list(network)
            best_len = len(network)

    # Phase 4: Final redundancy removal pass if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        final_network = _remove_redundant_comparators(best_network, remaining_time - 0.5)
        if len(final_network) < best_len:
            best_network = list(final_network)
            best_len = len(final_network)

    return best_network
