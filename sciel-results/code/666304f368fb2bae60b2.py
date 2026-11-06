import time
import numpy as np
import random

N = 18
NUM_INPUTS = 1 << N

def _generate_inputs():
    """Generate all 2^N boolean inputs as a numpy array."""
    # Create a pattern for all numbers from 0 to 2^N - 1
    # We need the bits of each number.
    # np.arange(2**N) gives the numbers.
    # We can use bit operations.
    inputs = np.arange(NUM_INPUTS, dtype=np.uint8)
    # Expand to binary representation
    # This is a bit tricky to do efficiently without loops in pure numpy for arbitrary N, 
    # but for N=18, we can do:
    bits = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        bits[:, i] = (inputs >> i) & 1
    return bits

# Pre-compute inputs globally to save time in construct
INPUTS = None

def _verify_network(network):
    """
    Verify if the network sorts all boolean inputs.
    Uses vectorized operations.
    """
    global INPUTS
    if INPUTS is None:
        INPUTS = _generate_inputs()
    
    # Copy inputs
    Y = INPUTS.copy()
    
    # Apply comparators
    for i, j in network:
        # Min goes to i, Max goes to j
        min_val = np.minimum(Y[:, i], Y[:, j])
        max_val = np.maximum(Y[:, i], Y[:, j])
        Y[:, i] = min_val
        Y[:, j] = max_val
        
    # Check if sorted
    # Y[:, k] <= Y[:, k+1] for all k
    # We can check all adjacent pairs
    is_sorted = np.all(Y[:, :-1] <= Y[:, 1:], axis=1)
    return np.all(is_sorted)

def _odd_even_transposition_network():
    """Generate the seed network."""
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network

def _greedy_remove(network):
    """Remove redundant comparators."""
    improved = True
    while improved:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            test_network = network[:idx] + network[idx+1:]
            if _verify_network(test_network):
                network = test_network
                improved = True
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # 1. Initialize
    network = _odd_even_transposition_network()
    
    # 2. Initial Greedy Removal
    network = _greedy_remove(network)
    
    best_network = network
    best_size = len(network)
    
    # 3. Local Search with Swaps and Add-Remove
    # We will alternate between trying to remove, swapping, and adding-then-removing.
    
    # Pre-compute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Random seed for reproducibility in local search steps if needed, but random is fine
    random.seed(42)
    
    while (time.time() - start_time) < deadline:
        current_network = best_network
        current_size = best_size
        improved_in_round = False
        
        # Phase A: Try Greedy Removal again (might be enabled by previous swaps)
        # We do a quick pass, not full while loop, to save time for search
        for idx in range(len(current_network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            test_network = current_network[:idx] + current_network[idx+1:]
            if _verify_network(test_network):
                current_network = test_network
                current_size -= 1
                improved_in_round = True
                # If we removed one, we should restart the scan or continue?
                # Continuing is fine, but indices shift. 
                # To be safe and fast, let's just mark improved and break to restart phase?
                # Or just continue. Since we are iterating backwards, removing idx doesn't affect 0..idx-1.
                # So we can continue.
        
        if current_size < best_size:
            best_network = current_network
            best_size = current_size
            # If we improved, we should probably restart the search logic from this new best
            # But let's try to do more swaps in the same "round" if time permits?
            # Actually, if we found a smaller network, the previous swaps might be invalid or suboptimal.
            # Let's reset current_network to best_network and continue the loop.
            continue

        # Phase B: Random Swaps
        # Try a few random swaps to escape local minima
        for _ in range(50): # 50 random swaps per round
            if (time.time() - start_time) > deadline - 10:
                break
            if len(current_network) < 2:
                break
            
            i, j = random.sample(range(len(current_network)), 2)
            test_network = current_network.copy()
            test_network[i], test_network[j] = test_network[j], test_network[i]
            
            if _verify_network(test_network):
                current_network = test_network
                # After a successful swap, try to remove again
                # We don't update best_size yet, as size is same.
                # But we might enable a removal.
                # Let's do a quick removal check on the swapped network
                for idx in range(len(current_network) - 1, -1, -1):
                    if (time.time() - start_time) > deadline - 10:
                        break
                    test_net2 = current_network[:idx] + current_network[idx+1:]
                    if _verify_network(test_net2):
                        current_network = test_net2
                        current_size -= 1
                        if current_size < best_size:
                            best_network = current_network
                            best_size = current_size
                        break # Found a removal, break inner loop
        
        # Phase C: Add-Remove (Net Reduction)
        # If we haven't improved size in this round, try adding a comparator and removing two.
        # This is expensive, so do it less frequently or with limited attempts.
        if (time.time() - start_time) > deadline - 20:
            break
            
        # Try a limited number of Add-Remove attempts
        for _ in range(10):
            if (time.time() - start_time) > deadline - 20:
                break
            
            # Pick a random pair to add
            add_pair = random.choice(all_pairs)
            if add_pair in current_network:
                continue
                
            test_network = current_network + [add_pair]
            
            # Try to remove two comparators from this extended network
            # We want to end up with size current_size - 1
            # So we remove 2 from size current_size + 1.
            
            # Optimization: Only try removing pairs that are "likely" to be redundant.
            # But for simplicity, let's try random pairs of indices to remove.
            # There are ~100 comparators. Pairs are ~5000. Too many for 10 attempts.
            
            # Better: Try removing the added pair and one other? No, that's just removal of one other.
            # We want to remove 2 *existing* ones, keeping the added one?
            # Or remove 1 existing and the added one? That's just removal of 1 existing (which we already tried in Phase A).
            # So we must remove 2 existing ones and keep the added one?
            # If we keep the added one, we are swapping 2 existing for 1 new? No, size changes from K to K+1-2 = K-1.
            # This is a "2-for-1" exchange.
            
            # Strategy:
            # 1. Add pair P.
            # 2. Try to remove pair (A, B) from original network.
            # 3. Check if (original - A - B + P) is valid.
            
            # To limit checks, we can pick random A and B.
            if len(current_network) < 2:
                continue
                
            idx_a, idx_b = random.sample(range(len(current_network)), 2)
            
            # Construct candidate: remove A and B, add P
            # Note: indices in current_network
            # We need to be careful with indices if we modify the list.
            # Let's create the list explicitly.
            candidate = []
            for k, comp in enumerate(current_network):
                if k != idx_a and k != idx_b:
                    candidate.append(comp)
            candidate.append(add_pair)
            
            if _verify_network(candidate):
                current_network = candidate
                current_size -= 1
                if current_size < best_size:
                    best_network = current_network
                    best_size = current_size
                    break # Restart with new best

    return best_network

