"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sorting network for N=17 wires using simulated annealing with
    vectorized verification over all 131,072 Boolean inputs.
    
    Strategy:
    1. Start from odd-even transposition network (136 comparators).
    2. Use simulated annealing to minimize comparator count.
    3. Moves: remove a comparator, add a comparator, swap two comparators.
    4. Verify correctness using the zero-one principle with NumPy vectorization.
    """
    deadline = time.time() + 85  # Leave 5s safety margin for 90s limit
    
    # Initialize with odd-even transposition network
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Precompute all Boolean inputs as a (131072, 17) uint8 array
    # Each row is one Boolean input
    all_inputs = np.zeros((NUM_BOOL_INPUTS, N), dtype=np.uint8)
    for i in range(NUM_BOOL_INPUTS):
        for j in range(N):
            all_inputs[i, j] = (i >> j) & 1
    
    def verify(net):
        """
        Verify that the network sorts all 131,072 Boolean inputs.
        Returns True if valid, False otherwise.
        """
        # Process each comparator in sequence
        # For each comparator (i, j): wire i gets min, wire j gets max
        # In terms of Boolean values: 
        #   new_i = old_i & old_j  (min of two bits)
        #   new_j = old_i | old_j  (max of two bits)
        
        states = all_inputs.copy()  # (131072, 17)
        
        for (i, j) in net:
            # min(i,j) goes to wire i, max(i,j) goes to wire j
            # For bits: min = AND, max = OR
            new_i = states[:, i] & states[:, j]
            new_j = states[:, i] | states[:, j]
            states[:, i] = new_i
            states[:, j] = new_j
        
        # Check if output is sorted: for each input, wire 0 <= wire 1 <= ... <= wire N-1
        # For bits, this means once we see a 1, all subsequent must be 1
        # i.e., no pattern of 1 followed by 0
        # Check: for each adjacent pair (k, k+1), states[:,k] <= states[:,k+1]
        # This is equivalent to: NOT (states[:,k] == 1 AND states[:,k+1] == 0)
        
        for k in range(N - 1):
            # If any row has 1 at position k and 0 at position k+1, it's not sorted
            if np.any((states[:, k] == 1) & (states[:, k+1] == 0)):
                return False
        return True
    
    def verify_fast(net):
        """
        Faster verification using early termination and batch processing.
        """
        states = all_inputs.copy()
        
        for (i, j) in net:
            new_i = states[:, i] & states[:, j]
            new_j = states[:, i] | states[:, j]
            states[:, i] = new_i
            states[:, j] = new_j
        
        # Check sortedness: for each adjacent pair, no 1 followed by 0
        # Vectorized check
        for k in range(N - 1):
            if np.any((states[:, k] == 1) & (states[:, k+1] == 0)):
                return False
        return True
    
    # Check initial network
    if not verify_fast(network):
        # Should not happen, but fallback
        return network
    
    best_network = list(network)
    best_cost = len(network)
    current_cost = best_cost
    
    # Simulated annealing parameters
    temp = 5.0
    cooling_rate = 0.995
    min_temp = 0.01
    
    random.seed(42)
    
    while time.time() < deadline and temp > min_temp:
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a random comparator
            if len(network) > 0:
                idx = random.randint(0, len(network) - 1)
                removed = network.pop(idx)
                new_cost = len(network)
                # Check validity
                if verify_fast(network):
                    # Accept
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_network = list(network)
                else:
                    # Reject, restore
                    network.insert(idx, removed)
        elif move_type < 0.7:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_comp = (i, j)
            network.append(new_comp)
            new_cost = len(network)
            # Check validity
            if verify_fast(network):
                # Accept with probability based on cost increase
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_network = list(network)
                else:
                    network.pop()
            else:
                network.pop()
        else:
            # Swap two random comparators
            if len(network) >= 2:
                idx1 = random.randint(0, len(network) - 1)
                idx2 = random.randint(0, len(network) - 1)
                if idx1 != idx2:
                    network[idx1], network[idx2] = network[idx2], network[idx1]
                    if verify_fast(network):
                        # Cost unchanged, always accept
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_network = list(network)
                    else:
                        # Revert
                        network[idx1], network[idx2] = network[idx2], network[idx1]
        
        temp *= cooling_rate
    
    # Final cleanup: try to remove any redundant comparators
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(best_network) - 1, -1, -1):
            removed = best_network.pop(idx)
            if verify_fast(best_network):
                changed = True
                best_cost = len(best_network)
            else:
                best_network.insert(idx, removed)
    
    return best_network
# EVOLVE-BLOCK-END
