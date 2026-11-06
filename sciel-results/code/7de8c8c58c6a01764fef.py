"""Improved program: search for a smaller sorting network on 18 wires."""
import time
import random
import numpy as np

N = 18
NUM_BOOL_INPUTS = 1 << N  # 262144

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # --- Helper functions ---
    
    def simulate_network(network, inputs):
        """
        Simulate the network on a batch of boolean inputs.
        inputs: shape (B, N) boolean array
        network: list of (i, j) comparators
        Returns: shape (B, N) boolean array of outputs
        """
        state = inputs.copy()
        for i, j in network:
            # min goes to i, max goes to j
            # For booleans: min = AND, max = OR
            state[:, i] = state[:, i] & state[:, j]
            state[:, j] = state[:, i] | state[:, j]
        return state

    def is_sorted(state):
        """Check if each row of state is sorted (non-decreasing)."""
        # A boolean row is sorted if it's of the form 000...0111...1
        # i.e., once a 1 appears, all subsequent must be 1
        # Equivalent: no 1 followed by 0
        # state[:, :-1] <= state[:, 1:]
        return np.all(state[:, :-1] <= state[:, 1:])

    def is_valid_network(network):
        """Check if the network sorts all boolean inputs."""
        # Generate all 2^N boolean inputs
        # Use bit manipulation for efficiency
        inputs = np.zeros((NUM_BOOL_INPUTS, N), dtype=np.uint8)
        for i in range(N):
            inputs[:, i] = (np.arange(NUM_BOOL_INPUTS, dtype=np.uint32) >> i) & 1
        
        outputs = simulate_network(network, inputs)
        return is_sorted(outputs)

    def count_comparators(network):
        return len(network)

    # --- Seed: Odd-even transposition network ---
    def seed_network():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    # --- Search: Greedy removal with random restarts ---
    
    best_network = seed_network()
    best_count = count_comparators(best_network)
    
    # First, verify the seed is valid
    if not is_valid_network(best_network):
        # Fallback: just return the seed as is (shouldn't happen)
        return best_network

    # Try to improve via greedy removal
    def greedy_remove(network, max_rounds=100, time_limit=None):
        """
        Try to remove redundant comparators one by one.
        Returns the smallest valid network found.
        """
        current = list(network)
        current_count = len(current)
        
        for round_num in range(max_rounds):
            if time_limit and time.time() > time_limit:
                break
            
            # Shuffle the order of comparators to try
            indices = list(range(len(current)))
            random.shuffle(indices)
            
            improved = False
            for idx in indices:
                if time_limit and time.time() > time_limit:
                    break
                
                # Try removing comparator at idx
                if idx >= len(current):
                    continue
                    
                candidate = current[:idx] + current[idx+1:]
                
                # Check validity
                if is_valid_network(candidate):
                    current = candidate
                    current_count -= 1
                    improved = True
                    break  # Restart the scan since indices shifted
            
            if not improved:
                break
        
        return current

    # Run greedy removal from the seed
    result = greedy_remove(best_network, max_rounds=50, time_limit=deadline - 5.0)
    if len(result) < best_count and is_valid_network(result):
        best_network = result
        best_count = len(best_network)

    # Try random mutations: swap, insert, remove
    def random_mutation(network):
        """Apply a random mutation to the network."""
        net = list(network)
        r = random.random()
        
        if r < 0.5:
            # Remove a random comparator
            if len(net) > 0:
                idx = random.randint(0, len(net) - 1)
                net.pop(idx)
        elif r < 0.8:
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            net.insert(pos, (i, j))
        else:
            # Swap two comparators
            if len(net) >= 2:
                i = random.randint(0, len(net) - 2)
                j = random.randint(i + 1, len(net) - 1)
                net[i], net[j] = net[j], net[i]
        
        return net

    # Simulated annealing-like search
    current = list(best_network)
    current_count = len(current)
    temperature = 1.0
    cooling = 0.999
    
    step = 0
    while time.time() < deadline - 5.0:
        step += 1
        
        # Mutate
        candidate = random_mutation(current)
        candidate_count = len(candidate)
        
        # Check validity
        if not is_valid_network(candidate):
            # Reject invalid
            temperature *= cooling
            continue
        
        # Accept if better or with probability based on temperature
        if candidate_count < current_count:
            current = candidate
            current_count = candidate_count
            if current_count < best_count:
                best_network = list(current)
                best_count = current_count
        else:
            # Accept worse with probability exp(-delta/T)
            delta = candidate_count - current_count
            if delta > 0 and temperature > 0.01:
                if random.random() < np.exp(-delta / temperature):
                    current = candidate
                    current_count = candidate_count
        
        temperature *= cooling
        
        # Periodic greedy cleanup
        if step % 100 == 0:
            cleaned = greedy_remove(current, max_rounds=10, time_limit=deadline - 5.0)
            if len(cleaned) < len(current):
                current = cleaned
                current_count = len(current)
                if current_count < best_count:
                    best_network = list(current)
                    best_count = current_count

    # Final greedy cleanup
    final_result = greedy_remove(best_network, max_rounds=30, time_limit=deadline - 1.0)
    if len(final_result) < len(best_network) and is_valid_network(final_result):
        best_network = final_result

    return best_network
# EVOLVE-BLOCK-END
