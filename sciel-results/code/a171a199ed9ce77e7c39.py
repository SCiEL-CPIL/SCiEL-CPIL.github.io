"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
NUM_WIRES = N
NUM_COMPARATORS = N * (N - 1) // 2  # 153

# Precompute all possible comparators
ALL_COMPARATORS = [(i, j) for i in range(N) for j in range(i + 1, N)]
COMP_INDEX = {c: idx for idx, c in enumerate(ALL_COMPARATORS)}

# Bit positions for fast comparator application
# For a comparator (i, j), we need to swap bits if bit_i > bit_j
# In a 64-bit integer, bit i is at position (N-1-i) for wire i
# Let's define: wire k corresponds to bit position (N-1-k) in the integer
# So wire 0 is MSB, wire N-1 is LSB
# Comparator (i, j) with i < j: wire i has higher bit position than wire j
# If bit_i = 1 and bit_j = 0, we need to swap

def _apply_comparator_fast(state, i, j):
    """Apply comparator (i, j) to state (integer representing bit vector).
    Wire k is at bit position (N-1-k).
    i < j means wire i is 'higher' (should get min), wire j is 'lower' (should get max).
    So after compare-exchange: wire i gets min, wire j gets max.
    Since we're sorting ascending: smallest value goes to wire 0.
    Wait, the task says: "Each compare-exchange writes min to wire i and max to j."
    And the network must sort every input.
    So after sorting, wire 0 has the smallest, wire N-1 has the largest.
    """
    pos_i = N - 1 - i
    pos_j = N - 1 - j
    bit_i = (state >> pos_i) & 1
    bit_j = (state >> pos_j) & 1
    if bit_i == 1 and bit_j == 0:
        # Swap: set bit_i to 0, bit_j to 1
        state ^= (1 << pos_i) | (1 << pos_j)
    return state

def _is_sorted_state(state):
    """Check if state is sorted: all 1s come before all 0s? No.
    Sorted means wire 0 has min, wire N-1 has max.
    For Boolean: sorted means all 0s are in lower wires and all 1s in higher wires?
    No: min to wire 0, max to wire j. So wire 0 gets smallest.
    For Boolean inputs, sorted means: some number of 0s then 1s.
    Wire 0 = 0, ..., wire k-1 = 0, wire k = 1, ..., wire N-1 = 1.
    So the state should be of the form: 000...0111...1
    """
    # Find the position of the first 1 (from wire 0 side)
    # Wire 0 is MSB. So we want MSBs to be 0 and LSBs to be 1.
    # State is sorted if it equals (1 << k) - 1 for some k in [0, N]
    # i.e., the set of bits that are 1 forms a contiguous block at the LSB end.
    if state == 0:
        return True
    if state == (1 << N) - 1:
        return True
    # Check if state is of the form 2^k - 1
    # This is true if state & (state + 1) == 0
    return (state & (state + 1)) == 0

def _count_inversions(state):
    """Count the number of inversions in the state.
    An inversion is a pair (i, j) with i < j where wire i has 1 and wire j has 0.
    """
    count = 0
    ones_seen = 0
    for i in range(N):
        pos = N - 1 - i
        bit = (state >> pos) & 1
        if bit == 1:
            ones_seen += 1
        else:
            count += ones_seen
    return count

def _apply_network_fast(state, network):
    """Apply the entire network to a state."""
    for i, j in network:
        pos_i = N - 1 - i
        pos_j = N - 1 - j
        bit_i = (state >> pos_i) & 1
        bit_j = (state >> pos_j) & 1
        if bit_i == 1 and bit_j == 0:
            state ^= (1 << pos_i) | (1 << pos_j)
    return state

def _verify_network(network):
    """Verify that the network sorts all 2^N Boolean inputs."""
    for state in range(1 << N):
        final_state = _apply_network_fast(state, network)
        if not _is_sorted_state(final_state):
            return False
    return True

def _remove_redundant_comparators(network):
    """Remove comparators that are redundant (don't affect the sorted output for any input)."""
    # A comparator is redundant if removing it still yields a valid sorting network.
    # We can try removing each comparator and checking validity.
    # But that's expensive. Let's do a simpler check: 
    # If a comparator is applied and the two wires already have the correct order
    # for ALL inputs that reach that point, it's redundant.
    # Simpler: try removing each one and verify.
    improved = True
    while improved:
        improved = False
        for idx in range(len(network)):
            candidate = network[:idx] + network[idx+1:]
            if _verify_network(candidate):
                network = candidate
                improved = True
                break
    return network

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s safety margin
    
    # Start with the odd-even transposition network
    base_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    best_network = _remove_redundant_comparators(base_network)
    best_len = len(best_network)
    
    if best_len <= 77:
        return best_network
    
    # Local search: try to remove more comparators
    # We'll use a more aggressive approach: try swapping, inserting, and removing
    
    rng = random.Random(42)
    
    # Try to improve by removing comparators one at a time
    # and also try reordering
    
    # First, let's try a greedy removal approach
    # For each comparator, check if it's needed
    current = list(best_network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each comparator
        for idx in range(len(current)):
            if time.time() > deadline:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate):
                current = candidate
                improved = True
                best_network = list(current)
                best_len = len(current)
                break
    
    if best_len <= 77:
        return best_network
    
    # Now try more complex moves: swap two comparators, or move a comparator
    # This is a local search in the space of comparator sequences
    
    # Use simulated annealing or hill climbing
    temperature = 1.0
    cooling = 0.99
    
    current = list(best_network)
    current_len = len(current)
    
    while time.time() < deadline:
        # Try random modifications
        move_type = rng.choice(['remove', 'swap', 'move', 'insert'])
        
        candidate = list(current)
        
        if move_type == 'remove' and len(candidate) > 0:
            idx = rng.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif move_type == 'swap' and len(candidate) > 1:
            i, j = rng.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move_type == 'move' and len(candidate) > 1:
            i, j = rng.sample(range(len(candidate)), 2)
            elem = candidate.pop(i)
            candidate.insert(j, elem)
        elif move_type == 'insert' and len(candidate) < 4096:
            # Insert a random comparator
            comp = rng.choice(ALL_COMPARATORS)
            idx = rng.randint(0, len(candidate))
            candidate.insert(idx, comp)
        
        if _verify_network(candidate):
            # Accept if shorter or same length (with some probability for same length)
            if len(candidate) < len(current):
                current = candidate
                current_len = len(current)
                if current_len < best_len:
                    best_network = list(current)
                    best_len = current_len
                    if best_len <= 77:
                        return best_network
            elif len(candidate) == len(current):
                # Accept with some probability to allow exploration
                if rng.random() < 0.1:
                    current = candidate
        else:
            # Reject invalid candidates
            pass
        
        temperature *= cooling
    
    return best_network
# EVOLVE-BLOCK-END
