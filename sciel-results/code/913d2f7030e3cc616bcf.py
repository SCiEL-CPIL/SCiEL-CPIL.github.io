"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
TOTAL_INPUTS = 1 << N  # 65536

# Precompute all inputs as bit patterns
_all_inputs = np.arange(TOTAL_INPUTS, dtype=np.uint32)
# For each wire position, extract that bit from all inputs
# bits[w] is an array of shape (65536,) with the w-th bit of each input
_bits = np.array([(_all_inputs >> w) & 1 for w in range(N)], dtype=np.uint8).T
# _bits shape: (65536, 16)

# For fast evaluation, we want to simulate the network.
# A comparator (i, j) does:
#   min_val = bits[:, i] & bits[:, j]  (AND)
#   max_val = bits[:, i] | bits[:, j]  (OR)
#   bits[:, i] = min_val
#   bits[:, j] = max_val

def _evaluate_network(net):
    """Check if the network sorts all 65536 inputs.
    Returns True if correct."""
    # Work on a copy
    state = _bits.copy()
    for (i, j) in net:
        # min goes to i, max goes to j
        new_i = state[:, i] & state[:, j]
        new_j = state[:, i] | state[:, j]
        state[:, i] = new_i
        state[:, j] = new_j
    
    # After sorting, wire 0 should have the minimum (all zeros if all zeros, etc.)
    # Actually, for a sorting network, after processing, the output should be sorted:
    # wire 0 <= wire 1 <= ... <= wire 15
    # In terms of bits: for each input, the output should have 0s followed by 1s
    # i.e., once we see a 1, all subsequent wires must be 1.
    
    # Check: for each row, the bits should be non-decreasing (0s then 1s)
    # Equivalent: no row has a 0 after a 1
    # For each wire pair (w, w+1), we need state[:, w] <= state[:, w+1]
    # i.e., state[:, w] & ~state[:, w+1] == 0 for all w
    
    for w in range(N - 1):
        # If wire w is 1 and wire w+1 is 0, that's a violation
        if np.any(state[:, w] & ~state[:, w+1]):
            return False
    return True

def _evaluate_partial(net, partial=True):
    """Evaluate a partial network, checking only the constraints that can be violated.
    For a partial network, we check if the current state is 'sort-ready' in a weak sense."""
    state = _bits.copy()
    for (i, j) in net:
        new_i = state[:, i] & state[:, j]
        new_j = state[:, i] | state[:, j]
        state[:, i] = new_i
        state[:, j] = new_j
    
    # For a partial network, we can't fully verify. But we can check
    # that no new violations are introduced relative to a baseline.
    # Actually, for search purposes, we just need to know if adding a comparator
    # helps or hurts. We'll use a simpler check: does the network reduce the
    # number of inversions?
    
    # Count inversions: for each input, count pairs (a, b) with a < b but state[a] > state[b]
    # This is expensive. Let's just return the state for now.
    return state

def _count_inversions(state):
    """Count total inversions across all inputs."""
    total = 0
    for w in range(N):
        for v in range(w + 1, N):
            # Inversion if state[:, w] > state[:, v], i.e., state[:, w] & ~state[:, v]
            total += np.sum(state[:, w] & ~state[:, v])
    return total

def _is_sorted(state):
    """Check if all inputs are sorted in the given state."""
    for w in range(N - 1):
        if np.any(state[:, w] & ~state[:, w+1]):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    """Construct a sorting network for 16 wires.
    
    Strategy: Start from the odd-even transposition network (120 comparators)
    and iteratively remove comparators that are redundant. Then try to find
    replacements with fewer comparators using local search.
    
    Deadline guard: track elapsed time and return best result before 85s.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Start with odd-even transposition network
    def _odd_even_transposition():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    current_net = _odd_even_transposition()
    best_net = list(current_net)
    best_count = len(current_net)
    
    # Verify initial network
    assert _evaluate_network(current_net), "Initial network should be correct"
    
    def _time_left():
        return DEADLINE - time.time()
    
    # Phase 1: Greedy removal of redundant comparators
    # Try removing each comparator and check if the network still sorts
    removal_order = list(range(len(current_net)))
    random.shuffle(removal_order)
    
    for idx in removal_order:
        if _time_left() < 5.0:
            break
        if idx >= len(current_net):
            continue
        
        # Try removing comparator at idx
        trial_net = current_net[:idx] + current_net[idx+1:]
        
        # Quick check: is the network still correct?
        if _evaluate_network(trial_net):
            current_net = trial_net
            if len(current_net) < best_count:
                best_net = list(current_net)
                best_count = len(current_net)
    
    # Phase 2: Local search - try replacing pairs of comparators with alternatives
    # For each pair of comparators, try swapping with other possible comparators
    # and see if we can reduce the count or maintain correctness with fewer.
    
    # Actually, let's do a more systematic local search:
    # Try adding a comparator and then removing two others.
    
    all_possible = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for iteration in range(200):
        if _time_left() < 3.0:
            break
        
        # Try to find a place where we can add one comparator and remove two
        # Randomly pick a position to insert and two to remove
        if len(current_net) < 2:
            break
        
        # Randomly choose two indices to potentially remove
        idx1, idx2 = random.sample(range(len(current_net)), 2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        # Try all possible new comparators to add
        for new_comp in all_possible:
            if _time_left() < 2.0:
                break
            
            # Build trial network: remove idx1, idx2, add new_comp at some position
            # Try adding at a random position
            remove_set = {idx1, idx2}
            base_net = [c for i, c in enumerate(current_net) if i not in remove_set]
            
            # Try inserting new_comp at various positions
            for pos in range(len(base_net) + 1):
                trial_net = base_net[:pos] + [new_comp] + base_net[pos:]
                
                if _evaluate_network(trial_net):
                    if len(trial_net) < len(current_net):
                        current_net = trial_net
                        if len(current_net) < best_count:
                            best_net = list(current_net)
                            best_count = len(current_net)
                        break
                else:
                    # If this position doesn't work, try next
                    continue
    
    # Phase 3: Try to find a known good construction
    # The reference 60-comparator network is known to exist.
    # Let's try to construct it using a bitonic sort approach.
    
    def _bitonic_sort_network():
        """Generate a bitonic sorting network for N=16."""
        comparators = []
        
        def _bitonic_merge(start, size, direction):
            """Merge a bitonic sequence of given size starting at start.
            direction: True if ascending (small to large), False if descending."""
            if size == 1:
                return
            half = size // 2
            # First, split into two halves and sort them in opposite directions
            _bitonic_merge(start, half, not direction)
            _bitonic_merge(start + half, half, direction)
            # Then compare-exchange elements that are half apart
            for i in range(half):
                a = start + i
                b = start + i + half
                if direction:
                    # Want ascending: min to a, max to b
                    comparators.append((a, b))
                else:
                    # Want descending: max to a, min to b
                    # But our comparator always puts min to first, max to second
                    # So for descending, we need to swap the order
                    comparators.append((b, a))
        
        def _bitonic_sort(start, size, direction):
            """Sort a sequence of given size in the given direction."""
            if size <= 1:
                return
            half = size // 2
            # Sort first half ascending, second half descending
            _bitonic_sort(start, half, True)
            _bitonic_sort(start + half, half, False)
            # Merge
            _bitonic_merge(start, size, direction)
        
        _bitonic_sort(0, N, True)
        return comparators
    
    bitonic_net = _bitonic_sort_network()
    if _evaluate_network(bitonic_net):
        if len(bitonic_net) < best_count:
            best_net = bitonic_net
            best_count = len(bitonic_net)
    
    # Return the best network found
    return best_net
# EVOLVE-BLOCK-END
