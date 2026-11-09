"""Size-efficient sorting networks on 17 wires."""
import time
import random
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def _apply_comparator(state, i, j):
    """Apply a single compare-exchange to a numpy array state."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _apply_network(state, network):
    """Apply the full network to a state."""
    for i, j in network:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _is_sorted(state):
    """Check if state is sorted in non-decreasing order."""
    for k in range(N - 1):
        if state[k] > state[k + 1]:
            return False
    return True

def _verify_network(network):
    """
    Verify a sorting network using the zero-one principle.
    Returns True if the network sorts all 2^17 binary inputs.
    """
    # Use numpy for vectorized verification
    # Generate all 2^17 binary inputs
    # We can do this efficiently by iterating through all combinations
    # but 131072 is manageable if we're careful
    
    # Actually, let's use a more efficient approach:
    # For each binary input, apply the network and check if sorted.
    
    # We'll iterate through all 2^N inputs
    # To speed up, we can skip inputs that are already sorted (all 0s or all 1s)
    # but we still need to check them for correctness.
    
    # Let's use a simple loop but with early termination on failure
    for mask in range(1 << N):
        # Create the binary input
        state = np.zeros(N, dtype=np.int8)
        for bit in range(N):
            if (mask >> bit) & 1:
                state[bit] = 1
        
        # Apply the network
        for i, j in network:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        
        # Check if sorted
        for k in range(N - 1):
            if state[k] > state[k + 1]:
                return False
    
    return True

def _verify_network_fast(network):
    """
    Faster verification using bit-level operations.
    Since we're only dealing with binary inputs, we can optimize.
    """
    # For binary inputs, the network must map each input to its sorted form.
    # The sorted form of a binary input with k ones is [0]*(N-k) + [1]*k
    
    # We'll iterate through all 2^N inputs
    # Precompute the network as a list of (i, j) pairs
    
    # To speed up, let's use a different approach:
    # Track the positions of 1s through the network.
    # But that's complex. Let's just do the direct simulation but optimized.
    
    # Actually, for 131072 inputs and up to ~100 comparators, 
    # that's about 13 million operations, which should be fast enough in Python
    # if we minimize overhead.
    
    # Let's try a vectorized approach using numpy
    # Generate all inputs at once
    # But 131072 x 17 is about 2.2 million elements, which is fine
    
    # Create all inputs as a 2D array
    # Input[i, j] = bit j of number i
    inputs = np.zeros((1 << N, N), dtype=np.int8)
    for bit in range(N):
        # For bit position 'bit', the pattern is 0 for first half, 1 for second half
        # repeated every 2^(bit+1)
        period = 1 << (bit + 1)
        half = 1 << bit
        # Create pattern: [0]*half + [1]*half repeated
        # This is a bit complex to vectorize directly
        # Let's just fill it in
        for val in range(1 << N):
            if (val >> bit) & 1:
                inputs[val, bit] = 1
    
    # Apply the network to all inputs simultaneously
    for i, j in network:
        # For each input, if inputs[:, i] > inputs[:, j], swap
        # Since they're binary, this is: if inputs[:, i] == 1 and inputs[:, j] == 0
        swap_mask = (inputs[:, i] == 1) & (inputs[:, j] == 0)
        inputs[swap_mask, i] = 0
        inputs[swap_mask, j] = 1
    
    # Check if all rows are sorted
    # A row is sorted if it's non-decreasing
    # For binary, this means all 0s come before all 1s
    # So for each row, once we see a 1, we can't see a 0 again
    # Equivalently, for each row, the sequence should be of the form 0...01...1
    
    # Check: for each row, find the first 1. All elements before must be 0,
    # and all elements from that point must be 1.
    # Or simpler: check that no row has a 0 after a 1.
    
    # For each adjacent pair of columns, check that there's no 0 followed by 1 in reverse
    # i.e., for each row, inputs[row, k] <= inputs[row, k+1] for all k
    
    # Vectorized check:
    # For each k from 0 to N-2, check that inputs[:, k] <= inputs[:, k+1]
    # This is equivalent to: no row has inputs[:, k] == 1 and inputs[:, k+1] == 0
    for k in range(N - 1):
        bad = (inputs[:, k] == 1) & (inputs[:, k + 1] == 0)
        if np.any(bad):
            return False
    
    return True

def _depth_of_network(network):
    """Calculate the parallel depth of the network."""
    # Depth is the length of the longest path of dependent comparators
    # Two comparators (i,j) and (k,l) are dependent if they share a wire
    # We can compute this by topological sort / longest path
    
    if not network:
        return 0
    
    # Build a graph where nodes are comparators, edges indicate dependency
    # Comparator a depends on comparator b if they share a wire and b comes before a
    
    # For simplicity, we'll compute depth by tracking the earliest time each wire can be free
    # Actually, let's just do a simple simulation:
    # depth[wire] = the step at which the wire was last used
    # When we place a comparator (i, j), its depth is max(depth[i], depth[j]) + 1
    # Then we update depth[i] and depth[j] to this new depth
    
    wire_depth = [0] * N
    max_depth = 0
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
        if d > max_depth:
            max_depth = d
    
    return max_depth

def _odd_even_transposition_network(n):
    """Generate the odd-even transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _akl_network(n):
    """
    Try to generate a smaller network using a heuristic approach.
    This is a placeholder for more sophisticated generation.
    """
    # Start with odd-even transposition and try to remove redundant comparators
    network = _odd_even_transposition_network(n)
    
    # Try removing each comparator and checking if the network still works
    # This is expensive, so we'll do it selectively
    
    improved = True
    while improved:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            if len(candidate) == 0:
                continue
            # Verify the candidate
            if _verify_network_fast(candidate):
                network = candidate
                improved = True
                break  # Restart the loop since the list changed
    
    return network

def _randomized_search(deadline):
    """
    Use randomized search to find a smaller sorting network.
    """
    # Start with the odd-even transposition network
    current_network = _odd_even_transposition_network(N)
    best_network = list(current_network)
    best_size = len(current_network)
    
    # Random seed for reproducibility
    random.seed(42)
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Try a random mutation
        if random.random() < 0.5:
            # Remove a random comparator
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                candidate = current_network[:idx] + current_network[idx+1:]
            else:
                continue
        else:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            # Insert at a random position
            idx = random.randint(0, len(current_network))
            candidate = current_network[:idx] + [(i, j)] + current_network[idx:]
        
        # Verify the candidate
        if _verify_network_fast(candidate):
            current_network = candidate
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_network = list(candidate)
    
    return best_network

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    # Set a deadline 5 seconds before the 90s limit to be safe
    deadline = time.time() + 85
    
    # Start with the odd-even transposition network
    # This uses 17*16/2 = 136 comparators
    current_network = _odd_even_transposition_network(N)
    best_network = list(current_network)
    best_size = len(current_network)
    
    # Try to improve using greedy removal
    # First, try removing each comparator in reverse order
    for idx in range(len(current_network) - 1, -1, -1):
        if time.time() > deadline:
            break
        
        candidate = current_network[:idx] + current_network[idx+1:]
        if _verify_network_fast(candidate):
            current_network = candidate
            best_size = len(current_network)
            best_network = list(current_network)
    
    # Try randomized search with remaining time
    if time.time() < deadline - 5:
        improved_network = _randomized_search(deadline - 5)
        if len(improved_network) < best_size:
            best_network = improved_network
            best_size = len(best_network)
    
    return best_network
# EVOLVE-BLOCK-END
