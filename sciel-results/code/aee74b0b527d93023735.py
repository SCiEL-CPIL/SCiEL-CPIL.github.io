"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute all Boolean inputs as bitmasks
    all_inputs = np.arange(1 << N, dtype=np.int64)
    
    # Sorted values: bitmask x is sorted (0s then 1s) iff x = 2^N - 2^k for k in [0, N]
    sorted_values = np.array([(1 << N) - (1 << k) for k in range(N + 1)], dtype=np.int64)
    sorted_set = set(sorted_values.tolist())
    
    def is_sorted(inputs):
        """Check if all inputs in the array are sorted (bitmask form)."""
        # Fast path: use set membership
        unique_vals = np.unique(inputs)
        return all(v in sorted_set for v in unique_vals)
    
    def apply_network_fast(inputs, comparators):
        """Apply a list of comparators to an array of bitmasks - optimized."""
        for (i, j) in comparators:
            # Extract bits i and j from all inputs
            bit_i = (inputs >> i) & 1
            bit_j = (inputs >> j) & 1
            # Where bit_j=1 and bit_i=0, we need to swap
            swap_mask = bit_j & (~bit_i)
            # Apply swap
            inputs = inputs ^ (swap_mask << i) ^ (swap_mask << j)
        return inputs
    
    def verify_network(network, inputs=None):
        """Verify if the network sorts all inputs."""
        if not network:
            return False
        if inputs is None:
            inputs = all_inputs.copy()
        else:
            inputs = inputs.copy()
        result = apply_network_fast(inputs, network)
        return is_sorted(result)
    
    def verify_network_subset(network, subset_size=4096):
        """Quick verification on a subset of inputs."""
        if not network:
            return False
        test_inputs = all_inputs[:subset_size].copy()
        result = apply_network_fast(test_inputs, network)
        return is_sorted(result)

    # Start with odd-even transposition network
    initial_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(initial_network)
    best_count = len(best_network)
    
    # Verify initial network
    if not verify_network(best_network):
        return best_network

    rng = random.Random(42)
    
    def greedy_remove_multi_pass(network, max_passes=3):
        """Try to remove comparators with multiple passes and different orderings."""
        net = list(network)
        
        for pass_num in range(max_passes):
            if time.time() - start_time > DEADLINE - 3:
                break
            
            # Create different orderings for each pass
            if pass_num == 0:
                # Forward order
                indices = list(range(len(net)))
            elif pass_num == 1:
                # Reverse order
                indices = list(range(len(net) - 1, -1, -1))
            else:
                # Random order
                indices = list(range(len(net)))
                rng.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > DEADLINE - 3:
                    break
                # If the index is out of bounds (because we removed things), skip
                if idx >= len(net):
                    continue
                
                # Try removing net[idx]
                candidate = net[:idx] + net[idx+1:]
                
                # Quick check on a subset first to save time on invalid candidates
                if verify_network_subset(candidate, 2048):
                    # Full check
                    if verify_network(candidate):
                        net = candidate
        
        return net

    def perturb_network(network, rng):
        """Apply a random perturbation to escape local minima."""
        net = list(network)
        if len(net) == 0:
            return net
        
        perturbation_type = rng.choice(['swap', 'insert', 'reverse', 'move'])
        
        if perturbation_type == 'swap':
            # Swap two adjacent elements
            if len(net) > 1:
                i = rng.randint(0, len(net) - 2)
                net[i], net[i+1] = net[i+1], net[i]
        
        elif perturbation_type == 'insert':
            # Insert a random comparator
            if len(net) < MAX_COMPARATORS:
                # Choose a random wire pair that's likely useful
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                pos = rng.randint(0, len(net))
                net.insert(pos, (i, j))
        
        elif perturbation_type == 'reverse':
            # Reverse a random segment
            if len(net) > 2:
                i = rng.randint(0, len(net) - 3)
                j = rng.randint(i + 1, len(net) - 1)
                net[i:j+1] = reversed(net[i:j+1])
        
        elif perturbation_type == 'move':
            # Move an element to a random position
            if len(net) > 2:
                i = rng.randint(0, len(net) - 1)
                j = rng.randint(0, len(net) - 1)
                if i != j:
                    elem = net.pop(i)
                    net.insert(j, elem)
        
        return net

    # Phase 1: Initial Greedy Removal with multiple passes
    best_network = greedy_remove_multi_pass(best_network, max_passes=5)
    best_count = len(best_network)
    
    # Phase 2: Iterative perturbation and greedy removal
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() - start_time < DEADLINE - 8:
        # Create a perturbed version of the best network
        perturbed = perturb_network(best_network, rng)
        
        # Check if perturbed is valid (use subset for speed)
        if not verify_network_subset(perturbed, 4096):
            no_improvement_count += 1
            continue
        
        # Try to remove comparators from this valid perturbed network
        reduced = greedy_remove_multi_pass(perturbed, max_passes=2)
        
        if len(reduced) < best_count:
            best_network = reduced
            best_count = len(best_network)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # If stuck for too long, try a more aggressive reset
        if no_improvement_count > max_no_improvement:
            # Do a few more aggressive perturbations
            for _ in range(10):
                if time.time() - start_time > DEADLINE - 8:
                    break
                perturbed = perturb_network(best_network, rng)
                if verify_network_subset(perturbed, 2048):
                    reduced = greedy_remove_multi_pass(perturbed, max_passes=1)
                    if len(reduced) < best_count:
                        best_network = reduced
                        best_count = len(best_network)
                        no_improvement_count = 0
                        break
            no_improvement_count = 0
    
    # Final polish: one more greedy pass on the best network
    if time.time() - start_time < DEADLINE - 2:
        final_network = greedy_remove_multi_pass(best_network, max_passes=3)
        if len(final_network) < best_count:
            best_network = final_network
            best_count = len(best_network)

    return best_network
# EVOLVE-BLOCK-END
