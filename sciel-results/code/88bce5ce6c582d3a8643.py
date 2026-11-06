"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 18 wires using a hybrid approach:
    1. Start with odd-even transposition (baseline).
    2. Apply local search to reduce comparator count by removing redundant comparators
       and swapping where possible, verified against all 2^18 Boolean inputs.
    
    For efficiency within 60s, we use a simplified local search:
    - Try removing each comparator and check if network still sorts all inputs.
    - Try swapping adjacent comparators to enable further removals.
    """
    import numpy as np
    import itertools
    
    # Generate initial network: odd-even transposition
    # This gives 18*9 = 162 comparators? Let's compute properly.
    # Phase 0: (0,1), (2,3), ..., (16,17) -> 9 comparators
    # Phase 1: (1,2), (3,4), ..., (15,16) -> 8 comparators
    # ... alternating
    # Total for N phases: 9 + 8 + 9 + 8 + ... for 18 phases
    # Even phases (0,2,4,...,16): 9 phases * 9 = 81
    # Odd phases (1,3,5,...,17): 9 phases * 8 = 72
    # Total = 153 comparators
    
    def build_network(comparators):
        """Build the network as a list of (i, j) pairs."""
        return list(comparators)
    
    def verify_network(comparators, N=18):
        """Verify that the sorting network sorts all 2^N Boolean inputs."""
        n = N
        # We'll simulate on all 2^n inputs.
        # For performance, use numpy vectorization.
        
        # Generate all 2^n inputs as an array of shape (2^n, n)
        # This can be large for n=18: 262144 x 18 = ~4.7M elements, manageable.
        
        # Create all possible binary inputs
        # Use bit manipulation to generate all 2^n vectors
        num_inputs = 1 << n
        # For each input index from 0 to 2^n - 1, create the bit vector
        # We'll process in chunks to manage memory if needed, but 262144*18 should be fine.
        
        # Create the full input array
        # Input matrix: each row is one of the 2^n inputs
        # input_array[k, i] = bit i of k
        
        # Efficient way using numpy
        # bits = np.arange(num_inputs)[:, None] >> np.arange(n)[::-1]
        # But this reverses bit order. Let's think about wire ordering.
        # Wire i corresponds to bit position i.
        # input_array[k, i] = (k >> i) & 1
        
        # Let's generate it properly
        indices = np.arange(num_inputs)
        # For each wire i, the value is (indices >> i) & 1
        wires = np.array([((indices >> i) & 1).astype(np.uint8) for i in range(n)]).T
        # wires shape: (num_inputs, n)
        
        # Now apply each comparator
        for (i, j) in comparators:
            # For each input, compare wires i and j
            # min goes to i, max goes to j
            wi = wires[:, i]
            wj = wires[:, j]
            
            # min(wi, wj) and max(wi, wj)
            mi = np.minimum(wi, wj)
            mj = np.maximum(wi, wj)
            
            wires[:, i] = mi
            wires[:, j] = mj
        
        # Check if sorted: for each input, wires should be non-increasing
        # (since min goes to lower index, so wire 0 has smallest, wire n-1 has largest)
        # Wait: min goes to wire i (lower index), max to wire j (higher index)
        # So after sorting, wire 0 <= wire 1 <= ... <= wire n-1
        # For binary inputs, this means all 0s come before all 1s.
        
        # Check: for each row, verify it's sorted
        # A binary sequence is sorted iff there's no 1 followed by a 0
        # Check: wires[:, 1:] <= wires[:, :-1] is false... 
        # Actually: sorted means wires[:, i] <= wires[:, i+1] for all i
        # For binary: no occurrence of [1, 0] adjacent
        
        # Check: all(wires[:, 1:] >= wires[:, :-1])
        sorted_mask = np.all(wires[:, 1:] >= wires[:, :-1], axis=1)
        return np.all(sorted_mask)
    
    # Start with odd-even transposition
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    
    # Initial network has 153 comparators
    # Now try to reduce by removing redundant comparators
    
    # Verify initial network (should pass)
    # For speed, we'll do a simpler check during optimization
    
    def verify_fast(comparators, N=18):
        """Fast verification using the zero-one principle: just check all 2^N Boolean inputs."""
        n = N
        num_inputs = 1 << n
        indices = np.arange(num_inputs)
        wires = np.array([((indices >> i) & 1).astype(np.uint8) for i in range(n)]).T
        
        for (i, j) in comparators:
            wi = wires[:, i]
            wj = wires[:, j]
            wires[:, i] = np.minimum(wi, wj)
            wires[:, j] = np.maximum(wi, wj)
        
        return np.all(wires[:, 1:] >= wires[:, :-1], axis=1)
    
    # Local search: try removing each comparator
    improved = True
    while improved:
        improved = False
        for idx in range(len(network)):
            # Try removing comparator at idx
            reduced = network[:idx] + network[idx+1:]
            if verify_fast(reduced):
                network = reduced
                improved = True
                break  # Restart the loop since indices changed
    
    return network
# EVOLVE-BLOCK-END
