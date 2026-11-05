import numpy as np
import time
import random

N = 17

def _generate_bitonic_sort_16():
    """
    Generates a Bitonic Sorting Network for 16 wires.
    Returns a list of (i, j) comparators.
    """
    comps = []
    
    def bitonic_merge(start, size, direction):
        # direction: 1 for ascending, -1 for descending
        if size == 1:
            return
        half = size // 2
        # Recursively merge two halves
        bitonic_merge(start, half, 1)
        bitonic_merge(start + half, half, -1)
        
        # Compare and swap
        for i in range(start, start + half):
            j = i + half
            if direction == 1:
                comps.append((i, j))
            else:
                comps.append((j, i))
                
    # Bitonic Sort for 16 wires
    # 1. Create bitonic sequences
    # 2. Merge them
    
    # Standard Bitonic Sort construction
    # Step 1: Generate bitonic sequences of size 2, 4, 8, 16
    # Actually, the recursive function above is for merging two sorted halves into a bitonic sequence?
    # No, standard Bitonic Sort:
    # 1. Sort first half ascending, second half descending.
    # 2. Merge them.
    
    # Let's implement the standard recursive Bitonic Sort for N=16.
    
    def sort_bitonic(start, size, direction):
        if size <= 1:
            return
        half = size // 2
        # Sort first half ascending, second half descending (or vice versa)
        sort_bitonic(start, half, 1)
        sort_bitonic(start + half, half, -1)
        # Merge
        for i in range(start, start + half):
            j = i + half
            if direction == 1:
                comps.append((i, j))
            else:
                comps.append((j, i))
                
    # To sort 16 wires ascending:
    # sort_bitonic(0, 16, 1)
    # Note: This assumes the input is arbitrary. Bitonic sort handles arbitrary input by first creating bitonic sequences.
    # The recursive call above creates a bitonic sequence from two sorted halves.
    # But the initial halves are not sorted.
    # Standard Bitonic Sort requires the input to be a bitonic sequence? No, it sorts any sequence.
    # The base case is size 1.
    # For size 2: sort_bitonic(0, 2, 1) -> sort(0,1,1), sort(1,1,-1), merge(0,1).
    # This works.
    
    sort_bitonic(0, 16, 1)
    return comps

def _generate_insertion_network_17():
    """
    Generates a network to insert wire 16 into the sorted wires 0-15.
    Uses a sequential insertion strategy which can be parallelized in depth but uses 16 comparators.
    """
    comps = []
    # Compare wire 16 with 0, then 1, ..., 15
    # This effectively bubbles wire 16 to its correct position.
    for i in range(16):
        comps.append((i, 16))
    return comps

def _simulate_network(network, inputs):
    """
    Simulates the network on a batch of boolean inputs.
    inputs: numpy array of shape (num_inputs, N)
    Returns: boolean array of shape (num_inputs,) indicating if sorted.
    """
    # Copy inputs to avoid modifying original
    state = inputs.copy()
    
    for i, j in network:
        # Perform compare-exchange
        # min to i, max to j
        # Since inputs are boolean, min is AND, max is OR
        # state[:, i] = state[:, i] & state[:, j]
        # state[:, j] = state[:, i] | state[:, j]
        # But we need to use the original values for both
        a = state[:, i]
        b = state[:, j]
        state[:, i] = a & b
        state[:, j] = a | b
        
    # Check if sorted
    # A boolean array is sorted if it is non-decreasing
    # i.e., no 1 is followed by a 0
    # sorted if all(state[:, i] <= state[:, i+1] for i in range(N-1))
    # In numpy: np.all(state[:, :-1] <= state[:, 1:], axis=1)
    return np.all(state[:, :-1] <= state[:, 1:], axis=1)

def _is_sorted_all(network):
    """
    Checks if the network sorts all 2^17 boolean inputs.
    """
    # Generate all 2^17 inputs
    # 131072 inputs
    # We can generate them in chunks or all at once if memory allows.
    # 131072 * 17 bytes = ~2MB. Fine.
    
    # Generate all binary numbers from 0 to 2^17 - 1
    max_val = 1 << N
    # Use numpy to generate bits
    # np.arange(max_val)
    # Then extract bits
    # This might be slow to generate all at once in Python loops, but numpy is fast.
    
    # Efficient way:
    # Create an array of indices 0 to 2^N - 1
    indices = np.arange(max_val, dtype=np.uint8)
    # Extract bits: (indices >> bit_pos) & 1
    # We want shape (max_val, N)
    bit_positions = np.arange(N - 1, -1, -1) # MSB to LSB? Doesn't matter for sorting check, just needs to cover all combinations.
    # Let's use LSB to MSB or vice versa.
    # inputs = (indices[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1
    inputs = (indices[:, np.newaxis] >> np.arange(N)[np.newaxis, :]) & 1
    # inputs is (131072, 17)
    
    return _simulate_network(network, inputs).all()

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s safety margin
    
    # 1. Generate Seed Network
    # Bitonic Sort for 16 wires
    seed_16 = _generate_bitonic_sort_16()
    # Insertion for 17th wire
    insertion = _generate_insertion_network_17()
    
    network = seed_16 + insertion
    
    # Optional: Verify seed is correct (should be)
    # if not _is_sorted_all(network):
    #     raise ValueError("Seed network is invalid")
    
    best_network = network
    best_count = len(network)
    
    # 2. Local Search
    # We will try to remove comparators.
    # To speed up verification, we use a random sample of inputs for intermediate checks.
    sample_size = 1000
    # Pre-generate sample inputs
    sample_indices = np.random.choice(1 << N, sample_size, replace=False)
    sample_inputs = (sample_indices[:, np.newaxis] >> np.arange(N)[np.newaxis, :]) & 1
    
    # Shuffle the network to try removing random ones
    # We'll maintain a list of indices of comparators to try removing
    # Or just iterate through the network and try removing each one?
    # Random order is better to avoid bias.
    
    # Create a list of indices to try
    indices = list(range(len(network)))
    random.shuffle(indices)
    
    # We will iterate through the network multiple times if time permits
    while time.time() < deadline:
        # Try to remove each comparator in random order
        # We need to be careful: removing one might make another removable.
        # So we iterate through the current list of comparators.
        
        # To handle dynamic list size, we use a while loop or a list of indices
        # Let's use a list of indices to try, and if one is removed, we don't try the next index immediately?
        # Simpler: Iterate through the network, try to remove. If removed, restart the iteration?
        # That might be slow.
        
        # Better: Try to remove a random comparator from the current network.
        if not network:
            break
            
        # Pick a random comparator to remove
        # But we should try to remove "redundant" ones.
        # A random choice is a valid heuristic.
        
        # To avoid re-checking the same removed comparator, we can keep a list of "active" comparators.
        # But the network structure changes.
        
        # Let's use a "greedy" approach:
        # Iterate through the network from the end to the beginning?
        # Or random.
        
        # Let's pick a random index
        idx = random.randint(0, len(network) - 1)
        
        # Try removing network[idx]
        candidate = network[:idx] + network[idx+1:]
        
        # Quick check on sample
        if _simulate_network(candidate, sample_inputs).all():
            # Full check on all inputs
            if _is_sorted_all(candidate):
                # Accept
                network = candidate
                best_network = network
                best_count = len(network)
                # Restart random selection? No, just continue
            else:
                # Revert
                pass
        else:
            # Revert
            pass
            
        # If we have time, we can also try to add comparators? No, we want to minimize.
        # We can also try to swap comparators to reduce depth? Score is count, not depth.
        
    return best_network

# EVOLVE-BLOCK-START
def construct():
    # Use the local search strategy defined above
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # 1. Generate Seed Network
    seed_16 = _generate_bitonic_sort_16()
    insertion = _generate_insertion_network_17()
    
    network = seed_16 + insertion
    
    best_network = network
    best_count = len(network)
    
    sample_size = 2000
    sample_indices = np.random.choice(1 << N, sample_size, replace=False)
    sample_inputs = (sample_indices[:, np.newaxis] >> np.arange(N)[np.newaxis, :]) & 1
    
    # Randomly shuffle the order of removal attempts
    # We'll use a list of indices to try, but since the list shrinks, we need to be careful.
    # We'll just pick random indices from the current length.
    
    while time.time() < deadline:
        if not network:
            break
            
        idx = random.randint(0, len(network) - 1)
        
        candidate = network[:idx] + network[idx+1:]
        
        # Quick check
        if _simulate_network(candidate, sample_inputs).all():
            # Full check
            if _is_sorted_all(candidate):
                network = candidate
                best_network = network
                best_count = len(network)
            else:
                pass
        else:
            pass
            
    return best_network
# EVOLVE-BLOCK-END
