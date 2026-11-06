"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
TIME_LIMIT = 60.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to leave before the hard limit

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on N wires.
    
    Strategy:
    1. Start with a known-correct network (odd-even transposition, 153 comparators).
    2. Use local search to remove redundant comparators while maintaining correctness.
    3. Use simulated annealing to explore the space of networks.
    
    Returns a list of (i, j) pairs where i < j, representing compare-exchange operations.
    """
    start_time = time.time()
    
    # Generate the initial odd-even transposition network
    def odd_even_transposition():
        """Generate odd-even transposition sorting network."""
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators
    
    # Check if a network sorts all 2^N boolean inputs
    def is_sorting_network(comparators, timeout=None):
        """Check if the network sorts all boolean inputs."""
        if timeout is not None:
            if time.time() - start_time > timeout:
                return None  # timeout
        
        n = len(comparators)
        if n > MAX_COMPARATORS:
            return False
        
        # Efficient approach using bit operations on a uint32 array
        # Represent all 2^N inputs as an array of uint32 integers
        # Each integer represents the state of the 18 wires
        
        # Precompute the bit masks for each wire position
        # For a comparator (i, j), we need to swap bits i and j where bit_i=1 and bit_j=0
        
        # Create all 2^N inputs
        all_inputs = np.arange(1 << N, dtype=np.uint32)
        
        # Apply each comparator
        for (i, j) in comparators:
            mask_i = np.uint32(1) << np.uint32(i)
            mask_j = np.uint32(1) << np.uint32(j)
            
            # Find states where bit_i=1 and bit_j=0
            swap_mask = (all_inputs & mask_i) != 0
            swap_mask = swap_mask & ((all_inputs & mask_j) == 0)
            
            if np.any(swap_mask):
                # Clear both bits, then set them swapped
                # Clear bit_i: all_inputs[swap_mask] &= ~mask_i
                # Clear bit_j: all_inputs[swap_mask] &= ~mask_j
                # Set bit_j: all_inputs[swap_mask] |= mask_j
                # Set bit_i: all_inputs[swap_mask] |= mask_i (but we want to swap, so set i to 0 and j to 1)
                # Wait, we want min to wire i and max to wire j.
                # If bit_i=1 and bit_j=0, after swap: bit_i=0, bit_j=1.
                # So: clear bit_i, set bit_j.
                all_inputs[swap_mask] = all_inputs[swap_mask] & (np.uint32(-1) ^ mask_i)
                all_inputs[swap_mask] = all_inputs[swap_mask] | mask_j
        
        # Check if all states are sorted
        # A state is sorted if for all k < l, if bit_k=1 then bit_l=1
        # Equivalently, the state is of the form 0...01...1
        # We can check this by verifying that the state equals its sorted version
        # Or more efficiently: for each state, check that it has the form 0^a 1^b
        
        # For a sorted state with k ones, the value is (2^k - 1) << (N - k)
        # Let's check: for each state, compute the number of ones (popcount)
        # and verify that the state matches the canonical sorted form.
        
        # Popcount for uint32
        # We can use a lookup table or bit manipulation
        # For N=18, the max value is 2^18 - 1, so we only need 18 bits
        
        # Let's use a different check: a state is sorted iff
        # for all i, if bit_i = 1 then for all j > i, bit_j = 1
        # Equivalently: if we look at the sequence of bits from LSB to MSB,
        # once we see a 0, all higher bits must be 0? No, that's wrong.
        # Wire 0 is LSB, wire N-1 is MSB.
        # Sorted means wire 0 has min value, wire N-1 has max value.
        # So if wire i has 1 and wire j has 0 with i < j, that's unsorted.
        # In our representation, bit i corresponds to wire i.
        # So sorted means: if bit_i = 1 and i < j, then bit_j must be 1.
        # Equivalently: the bits are non-decreasing from wire 0 to wire N-1.
        # So the state looks like 0...01...1 where 0s are on the left (lower wires) and 1s on the right (higher wires).
        
        # Let's verify: for each state, check that it is of the form (2^k - 1) << (N - k) for some k.
        # This is equivalent to: the state has the form 00...011...1 with k ones at the top.
        
        # We can check this by:
        # 1. Count the number of ones: k
        # 2. The expected value is ((1 << k) - 1) << (N - k)
        # 3. Check if state == expected_value
        
        # But computing popcount for 2^18 values might be slow.
        # Alternative: a state is sorted iff (state & (state - 1)) == 0? No, that's for powers of 2.
        
        # Let's use a simpler check:
        # A state is sorted iff for all i from 0 to N-2:
        #   if bit_i = 1 then bit_{i+1} = 1
        # This is equivalent to: no pattern 10 in adjacent bits (from low to high).
        # Or: (state >> i) & 1 == 1 implies (state >> (i+1)) & 1 == 1
        
        # Let's check this vectorized:
        # For each i, check: ((states >> i) & 1) <= ((states >> (i+1)) & 1)
        
        # Actually, let's just check: state is sorted iff
        # state & (state - 1) ... no.
        
        # Let's use the property: a number is of the form 0...01...1 iff
        # it is equal to (1 << k) - 1 shifted, or more simply:
        # Let's define: a state is sorted iff for all i < j, bit_i <= bit_j.
        # This is equivalent to: the state is a "suffix" of ones.
        
        # Efficient check:
        # Let's compute: for each state, let k = number of trailing zeros? No.
        
        # Actually, the simplest vectorized check:
        # A state is sorted iff (state | (state - 1)) == (1 << N) - 1? No.
        
        # Let's just check: for i in range(N-1):
        #   if bit_i == 1 and bit_{i+1} == 0, then unsorted.
        
        # We can do this with a loop over i, but that's N-1 = 17 iterations.
        # Or we can use a clever bit trick.
        
        # Trick: a state is sorted iff state & (state - 1) == 0 is false for most.
        
        # Let's just do the adjacent check:
        unsorted = np.zeros(1 << N, dtype=bool)
        for i in range(N - 1):
            mask_i = np.uint32(1) << np.uint32(i)
            mask_j = np.uint32(1) << np.uint32(i + 1)
            # Check if bit_i = 1 and bit_{i+1} = 0
            bad = (all_inputs & mask_i) != 0
            bad = bad & ((all_inputs & mask_j) == 0)
            unsorted = unsorted | bad
        
        return not np.any(unsorted)
    
    # Start with the odd-even transposition network
    network = odd_even_transposition()
    
    # Verify it's correct (it should be)
    if not is_sorting_network(network):
        # Shouldn't happen, but just in case
        return network
    
    best_network = list(network)
    best_score = len(best_network)
    
    # Local search: try removing each comparator and check if still correct
    # Also try swapping adjacent comparators, etc.
    
    def try_remove(comparator_index):
        """Try removing the comparator at the given index."""
        new_network = network[:comparator_index] + network[comparator_index + 1:]
        if is_sorting_network(new_network):
            return new_network
        return None
    
    def try_swap(comparator_index):
        """Try swapping the comparator at the given index with the next one."""
        if comparator_index >= len(network) - 1:
            return None
        new_network = network.copy()
        new_network[comparator_index], new_network[comparator_index + 1] = \
            new_network[comparator_index + 1], new_network[comparator_index]
        if is_sorting_network(new_network):
            return new_network
        return None
    
    # Simulated annealing
    current_network = list(network)
    current_score = len(current_network)
    best_network = list(current_network)
    best_score = current_score
    
    temperature = 100.0
    cooling_rate = 0.95
    min_temperature = 0.1
    
    random.seed(42)
    
    while temperature > min_temperature:
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Try removing a random comparator
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 1)
                new_network = current_network[:idx] + current_network[idx + 1:]
                if is_sorting_network(new_network):
                    new_score = len(new_network)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_network = new_network
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_network = list(current_network)
        elif move_type < 0.85:
            # Try swapping two adjacent comparators
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 2)
                new_network = current_network.copy()
                new_network[idx], new_network[idx + 1] = \
                    new_network[idx + 1], new_network[idx]
                if is_sorting_network(new_network):
                    new_score = len(new_network)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_network = new_network
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_network = list(current_network)
        else:
            # Try moving a comparator to a random position
            if len(current_network) > 2:
                idx1 = random.randint(0, len(current_network) - 1)
                idx2 = random.randint(0, len(current_network) - 1)
                if idx1 != idx2:
                    new_network = current_network.copy()
                    comp = new_network.pop(idx1)
                    new_network.insert(idx2, comp)
                    if is_sorting_network(new_network):
                        new_score = len(new_network)
                        delta = new_score - current_score
                        if delta < 0 or random.random() < np.exp(-delta / temperature):
                            current_network = new_network
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_network = list(current_network)
        
        temperature *= cooling_rate
    
    # Final: try to remove more comparators from the best network found
    current_network = list(best_network)
    changed = True
    while changed and time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        changed = False
        for idx in range(len(current_network)):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            new_network = current_network[:idx] + current_network[idx + 1:]
            if is_sorting_network(new_network):
                current_network = new_network
                changed = True
                break
    
    return current_network
# EVOLVE-BLOCK-END
