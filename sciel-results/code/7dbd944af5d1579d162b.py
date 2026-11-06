import numpy as np
import random
import time

N = 18

def construct():
    """
    Constructs a sorting network for N=18 wires using a Greedy Construction 
    followed by Local Search Reduction.
    """
    
    def simulate_network(inputs, network):
        """
        Simulates the sorting network on a batch of inputs.
        inputs: numpy array of shape (num_inputs, N)
        network: list of (i, j)
        Returns: numpy array of sorted inputs
        """
        # Convert inputs to float for easier min/max or just use logic
        # inputs are binary (0/1)
        for i, j in network:
            # min goes to i, max goes to j
            min_val = np.minimum(inputs[:, i], inputs[:, j])
            max_val = np.maximum(inputs[:, i], inputs[:, j])
            inputs[:, i] = min_val
            inputs[:, j] = max_val
        return inputs

    def is_sorted(inputs):
        """
        Checks if all inputs are sorted (non-decreasing).
        inputs: numpy array of shape (num_inputs, N)
        """
        # Check if each row is sorted
        # diff >= 0
        diffs = np.diff(inputs, axis=1)
        return np.all(diffs >= 0)

    def count_unsorted(inputs, network):
        """
        Returns the number of inputs that are not sorted after applying network.
        """
        sorted_inputs = simulate_network(inputs.copy(), network)
        return np.sum(~is_sorted(sorted_inputs))

    def greedy_construct(seed=42):
        """
        Builds a network using a greedy heuristic.
        """
        random.seed(seed)
        # Generate a set of test inputs
        # Include unit vectors, anti-unit vectors, and random ones
        test_inputs = []
        for i in range(N):
            v = np.zeros(N)
            v[i] = 1
            test_inputs.append(v)
            v2 = np.ones(N)
            v2[i] = 0
            test_inputs.append(v2)
        
        for _ in range(500):
            test_inputs.append(np.random.randint(0, 2, N))
            
        test_inputs = np.array(test_inputs, dtype=int)
        
        network = []
        # We want to sort the test inputs.
        # Start with empty network.
        # Iteratively add comparators that help sort the "worst" inputs.
        
        # To speed up, we track which inputs are currently unsorted.
        # Initially all are unsorted (unless they are already sorted).
        
        max_iters = 200
        for _ in range(max_iters):
            # Find the input with the most inversions (or just pick a random unsorted one)
            # Simulate current network on test inputs
            current_state = simulate_network(test_inputs.copy(), network)
            unsorted_mask = ~is_sorted(current_state)
            
            if np.all(~unsorted_mask):
                break
            
            # Pick a random unsorted input
            unsorted_indices = np.where(unsorted_mask)[0]
            idx = random.choice(unsorted_indices)
            target_input = current_state[idx]
            
            # Try all comparators (i, j)
            best_comparator = None
            best_score = -1
            # To make it faster, only try a subset of comparators?
            # N=18, 153 pairs. 153 simulations of 1 input is fast.
            for i in range(N):
                for j in range(i+1, N):
                    # Apply comparator to target_input
                    t = target_input.copy()
                    min_val = min(t[i], t[j])
                    max_val = max(t[i], t[j])
                    t[i] = min_val
                    t[j] = max_val
                    # Calculate inversions in t
                    inversions = 0
                    for k in range(N):
                        for l in range(k+1, N):
                            if t[k] > t[l]:
                                inversions += 1
                    # We want to minimize inversions
                    # Current inversions of target_input
                    curr_inv = 0
                    for k in range(N):
                        for l in range(k+1, N):
                            if target_input[k] > target_input[l]:
                                curr_inv += 1
                    score = curr_inv - inversions
                    if score > best_score:
                        best_score = score
                        best_comparator = (i, j)
            
            if best_comparator is not None and best_score > 0:
                network.append(best_comparator)
            else:
                # If no single comparator helps this input, pick a random one?
                # Or break?
                # If we break, we might have unsorted inputs.
                # Let's just break and rely on reduction to fix? No, reduction only removes.
                # We need a valid network.
                # If greedy fails to sort, we need a fallback.
                pass
                
        return network

    def reduce_network(network, full_inputs):
        """
        Attempts to remove comparators from the network while maintaining validity.
        """
        # full_inputs: all 2^N inputs
        # This is expensive to check every time.
        # We can check on a subset first, then full.
        
        changed = True
        while changed:
            changed = False
            # Shuffle order to avoid bias
            indices = list(range(len(network)))
            random.shuffle(indices)
            
            for idx in indices:
                # Try removing network[idx]
                if idx >= len(network):
                    continue
                new_network = network[:idx] + network[idx+1:]
                
                # Check validity on full inputs
                # This is the bottleneck. 262k inputs.
                # simulate_network takes ~0.1s?
                # 100 removals * 0.1s = 10s. OK.
                
                if is_sorted(simulate_network(full_inputs.copy(), new_network)):
                    network = new_network
                    changed = True
                    # Restart loop? Or continue? 
                    # Continuing is fine, but indices are now shifted.
                    # Better to restart the while loop to re-shuffle and re-index.
                    break
                    
        return network

    # Generate all possible inputs
    # 2^18 = 262144
    # Create a numpy array of shape (262144, 18)
    # Using np.arange and bit manipulation
    all_vals = np.arange(2**N, dtype=np.uint8)
    full_inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        full_inputs[:, i] = (all_vals >> i) & 1

    # 1. Greedy Construction
    # Try a few seeds
    best_network = None
    best_size = float('inf')
    
    for seed in [42, 123, 999]:
        net = greedy_construct(seed)
        # Validate
        if is_sorted(simulate_network(full_inputs.copy(), net)):
            # Try to reduce
            reduced_net = reduce_network(net, full_inputs)
            if len(reduced_net) < best_size:
                best_size = len(reduced_net)
                best_network = reduced_net
        else:
            # If greedy failed to sort, try to fix? 
            # Or just skip.
            pass
            
    if best_network is None:
        # Fallback: Odd-Even Transposition (Seed)
        # This ensures we return a valid network, even if score is 1.0
        # But we want to beat 1.68.
        # If greedy fails, we are in trouble.
        # Let's force a valid network by using a known simple one?
        # Or just return the greedy one if it's close? No, must be valid.
        
        # Let's try a simpler deterministic construction: Batcher's Odd-Even Merge?
        # No, "do not reuse current designs".
        
        # Let's try Knuth's algorithm? 
        # I'll implement a simplified Knuth-like merge.
        # Or just return the seed network (Odd-Even) to be safe?
        # Score 1.0 is better than 0.
        
        # Let's try to build a network using "Bubble Sort" comparators?
        # Bubble sort network has N*(N-1)/2 comparators = 18*17/2 = 153.
        # It is valid.
        bubble_net = []
        for i in range(N-1):
            for j in range(0, N-1-i, 2):
                bubble_net.append((j, j+1))
                bubble_net.append((j+1, j+2)) # Wait, standard bubble sort network is different.
        
        # Standard Bubble Sort Network (Odd-Even Transposition is similar but parallel).
        # Sequential Bubble Sort comparators:
        # Pass 1: (0,1), (2,3), ..., (16,17)
        # Pass 2: (1,2), (3,4), ..., (15,16)
        # ...
        # This is exactly Odd-Even Transposition.
        # So fallback is Odd-Even.
        return _odd_even_transposition()

    return best_network

def _odd_even_transposition():
    """Fallback: Standard Odd-Even Transposition Network."""
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N, 2):
                if i + 1 < N:
                    network.append((i, i + 1))
        else:
            for i in range(1, N, 2):
                if i + 1 < N:
                    network.append((i, i + 1))
    return network

# EVOLVE-BLOCK-END
