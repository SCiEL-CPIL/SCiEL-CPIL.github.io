import numpy as np
import time
import random

N = 16
MAX_INPUTS = 2 ** N
DEADLINE = 85.0  # Safety margin for 90s limit

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # 1. Generate the baseline Odd-Even Transposition Network (120 comparators)
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
                
    # 2. Prepare Bit-Parallel Verification Data
    inputs = np.arange(MAX_INPUTS, dtype=np.uint64)
    W = np.empty((N, MAX_INPUTS), dtype=np.uint8)
    for w in range(N):
        W[w] = (inputs >> w) & 1

    def apply_comparator(W_state, i, j):
        a = W_state[i].copy()
        b = W_state[j].copy()
        W_state[i] = np.minimum(a, b)
        W_state[j] = np.maximum(a, b)

    def is_sorted(W_state):
        return np.all(W_state[:-1] <= W_state[1:])

    def verify_network(net):
        if not net:
            return is_sorted(W)
        current_W = W.copy()
        for comp in net:
            apply_comparator(current_W, comp[0], comp[1])
        return is_sorted(current_W)

    # 3. Structural Change: Start with a different, more compact baseline.
    # The Odd-Even Transposition network is 120 comparators.
    # The Bitonic Sorting Network for 16 wires is 5 * 16 / 2 = 40? No.
    # Bitonic sort for N=2^k uses (k*(k+1)/2) * N/2 comparators.
    # For N=16 (k=4): (4*5/2) * 8 = 10 * 8 = 80 comparators.
    # This is a much better starting point than 120.
    # Let's implement a standard Bitonic Sort Network generator.
    
    def generate_bitonic_network(n):
        # n must be a power of 2
        comps = []
        def bitonic_merge(n, start, direction):
            # n is the size of the bitonic sequence
            if n <= 1:
                return
            half = n // 2
            # Recursively sort the two halves in opposite directions
            bitonic_merge(half, start, direction)
            bitonic_merge(half, start + half, not direction)
            
            # Compare elements that are half apart
            for i in range(half):
                # direction True means ascending (min at lower index)
                if direction:
                    comps.append((start + i, start + i + half))
                else:
                    comps.append((start + i + half, start + i))
            
        def bitonic_sort(n, start, direction):
            if n <= 1:
                return
            half = n // 2
            # Sort first half ascending, second half descending (to form bitonic)
            bitonic_sort(half, start, True)
            bitonic_sort(half, start + half, False)
            # Merge the bitonic sequence
            bitonic_merge(n, start, direction)

        bitonic_sort(n, 0, True)
        return comps

    bitonic_net = generate_bitonic_network(N)
    
    # Verify bitonic network
    if verify_network(bitonic_net):
        current_network = bitonic_net
    else:
        # Fallback to odd-even if bitonic generation failed
        current_network = network

    # 4. Local Search: Greedy Deletion with Randomized Restarts
    # We will perform multiple passes of greedy deletion.
    # To escape local minima, we will occasionally shuffle the network order
    # (preserving the multiset of comparators) and try to delete again.
    # However, shuffling a valid sorting network does not necessarily keep it valid,
    # but we only delete if the result is valid.
    # Actually, a better heuristic for sorting networks is to try to delete 
    # comparators that are "redundant". 
    # A simple and effective strategy is:
    # 1. Try to delete every comparator from the end to the start.
    # 2. If a deletion is valid, keep it.
    # 3. Repeat until no more deletions are possible.
    # 4. To escape local minima, we can try a "simulated annealing" like approach
    #    or just random restarts with different initial shuffles of the *valid* network.
    #    But shuffling a valid network makes it invalid usually.
    #    Instead, we can try to *move* comparators? That's complex.
    #    
    #    Alternative: Use a different initial network.
    #    We already have Bitonic (80).
    #    We can also try the AKS network? Too complex to generate quickly.
    #    We can try a random network? Unlikely to be valid.
    #    
    #    Let's stick to Bitonic (80) as the primary baseline.
    #    The current code starts with Odd-Even (120).
    #    Starting with 80 gives us a huge head start.
    
    # Optimization: Use a faster verification for deletion.
    # Instead of re-simulating the whole network, we can use the "suffix simulation" trick.
    # But since the network is small (<=80), full re-simulation is fast enough.
    # 80 comparators * 65536 inputs * 16 wires is very fast in numpy.
    
    # Let's implement the greedy deletion loop.
    
    def optimize_network(net):
        if not verify_network(net):
            return net # Should not happen if we start with a valid one
        
        current_net = net[:]
        changed = True
        while changed:
            changed = False
            # Iterate from end to start
            for i in range(len(current_net) - 1, -1, -1):
                if time.time() > DEADLINE:
                    return current_net
                
                # Try removing current_net[i]
                test_net = current_net[:i] + current_net[i+1:]
                
                # Verify
                if verify_network(test_net):
                    current_net = test_net
                    changed = True
                    # Continue loop, indices shift
                    # We don't need to re-simulate the whole thing if we were smart,
                    # but verify_network is O(N * 2^N * N) which is fine.
                    
            if time.time() > DEADLINE:
                return current_net
        
        return current_net

    # Run optimization on the bitonic network
    optimized_bitonic = optimize_network(current_network)
    
    # If we have time, try to improve further with a different heuristic.
    # One idea: "Local Search" where we swap adjacent comparators?
    # Or try to delete from the middle?
    
    # Let's try a second pass with a slightly different order (e.g., random order of indices)
    # to see if we can delete more.
    
    if time.time() < DEADLINE - 5:
        # Try a randomized deletion order
        best_net = optimized_bitonic[:]
        
        for restart in range(10): # 10 restarts
            if time.time() > DEADLINE - 2:
                break
            
            # Start from the best found so far
            current_net = best_net[:]
            
            # Randomize the order of deletion attempts
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            
            changed = True
            while changed:
                changed = False
                for i in indices:
                    if i >= len(current_net):
                        continue
                    if time.time() > DEADLINE - 2:
                        break
                    
                    test_net = current_net[:i] + current_net[i+1:]
                    if verify_network(test_net):
                        current_net = test_net
                        changed = True
                        # Update indices? This is tricky.
                        # Simpler: just do a full pass again with a new shuffle.
                        # Or break and re-shuffle.
                        break 
                
                # If we deleted something, re-shuffle and try again
                if changed:
                    indices = list(range(len(current_net)))
                    random.shuffle(indices)
            
            if len(current_net) < len(best_net):
                best_net = current_net[:]
    
    return best_net

# EVOLVE-BLOCK-END
