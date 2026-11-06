import numpy as np
import time
import random

N = 18
SIZE = 1 << N  # 262144

def construct():
    """
    Constructs a sorting network using a bit-parallel exhaustive search strategy.
    Uses the 0-1 principle and NumPy vectorization to efficiently verify 
    sorting correctness for all 2^N inputs.
    """
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin
    
    # Initialize states for all 2^N inputs
    # states[j] is a bitstring of length SIZE representing the value of wire j
    # for all possible inputs.
    # We represent the state of all inputs as a matrix of uint8, where each row
    # is a wire and each column is an input.
    k_range = np.arange(SIZE, dtype=np.uint64)
    states = np.empty((N, SIZE), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)
    
    # Helper to check if sorted
    def _is_sorted(st):
        # Sorted if st[0] >= st[1] >= ... >= st[N-1]
        # For binary, this means no inversion where st[j] < st[j+1]
        # i.e., st[j] == 1 and st[j+1] == 0 is an inversion.
        # We want to check if there are ANY inversions.
        # Inversion exists if (st[j] & ~st[j+1]).any()
        # We can vectorize this:
        # inversions = st[:-1] & ~st[1:]
        # return not np.any(inversions)
        return not np.any(st[:-1] & ~st[1:])
    
    # Helper to apply a comparator to the global states
    # Returns the modified states
    def _apply_comparator(st, i, j):
        # min to i, max to j
        # min = a & b, max = a | b
        si = st[i].copy()
        sj = st[j].copy()
        st[i] = si & sj
        st[j] = si | sj
        return st

    # Initial network: Odd-Even Transposition Sort
    # This is a known valid sorting network, though large (N^2/2 comparators)
    # For N=18, this is 18*18/2 = 162? No, it's sum of 9+9+... 
    # Actually O(N^2) comparators.
    # Let's just generate it.
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    
    # Verify initial network
    # We need to apply all comparators to the initial states to get the final states
    # But we want to be able to remove them.
    # Strategy: 
    # 1. Start with the full network.
    # 2. Compute the "residual" effect of the network.
    # 3. Try to remove comparators from the end (or anywhere) and check if the 
    #    remaining network still sorts.
    
    # To make removal efficient, we can pre-compute the state of the wires 
    # *before* each comparator is applied.
    # Let S_k be the state of the system after applying the first k comparators.
    # S_0 is the initial input.
    # S_N is the final output.
    # If we remove comparator k (which operates on S_{k-1} to produce S_k), 
    # the new state at step k becomes S_{k-1} (instead of S_k).
    # We then need to propagate this change through the rest of the network 
    # (comparators k+1 to M) and check if the final result is sorted.
    
    # Storing all S_k is memory intensive (N * SIZE * M). 
    # M ~ 150. N=18, SIZE=262144.
    # 150 * 18 * 262144 bytes ~ 700 MB. This might be too much for some environments, 
    # but likely okay. Let's try to be smarter.
    
    # Alternative: 
    # Greedy removal from the end.
    # If we remove the last comparator, we just need to check if the state 
    # before the last comparator is already sorted.
    # If not, we try removing the second to last, etc.
    
    # Let's implement a "Backward Greedy Removal" with full re-verification 
    # only when a removal is attempted, but optimized.
    
    # Actually, a better approach for speed:
    # Keep the list of comparators.
    # Keep the current "effective" states.
    # To test removal of comparator at index k:
    # 1. Save current states.
    # 2. Remove comparator k from the list.
    # 3. Re-run the network from scratch? No, too slow.
    
    # Let's use the "Checkpoint" method.
    # Store states at every k-th step?
    
    # Given the 60s limit, let's stick to a robust Greedy Removal that 
    # re-verifies by re-applying the network from the state *before* the removed 
    # comparator.
    
    # We will store the states at each step of the network.
    # states_history[k] is the state (N x SIZE) after applying comparator k.
    # This uses ~700MB. If it fails, we fall back to simpler methods.
    
    try:
        # Pre-compute history
        history = [None] * (len(network) + 1)
        history[0] = states.copy()
        current_state = states.copy()
        
        for idx, (i, j) in enumerate(network):
            current_state = _apply_comparator(current_state, i, j)
            history[idx + 1] = current_state.copy()
            
        # Now, greedy removal from the end
        # We iterate from the last comparator to the first.
        # We maintain a "working" network and a "working" history?
        # No, we can just check if the state before a comparator is sufficient 
        # to lead to a sorted output given the subsequent comparators.
        
        # Actually, the simplest correct greedy removal:
        # 1. Start with the full network.
        # 2. For each comparator from last to first:
        #    a. Temporarily remove it.
        #    b. Check if the resulting network sorts.
        #    c. If yes, keep it removed. If no, restore it.
        
        # How to check efficiently?
        # If we remove comparator k, the state at step k becomes history[k] (state before comp k).
        # We then apply comparators k+1 ... M-1 to history[k] and check if sorted.
        
        # We can pre-compute the "suffix networks"?
        # Suffix k is the network consisting of comparators k, k+1, ..., M-1.
        # We want to know if Suffix(k+1) applied to history[k] results in a sorted state.
        
        # Let's just do it on the fly. 
        # M is ~150. 
        # For each removal attempt, we apply ~M/2 comparators on average.
        # 150 * 75 * (2 * 262144 ops) ... 
        # 150 * 75 * 500k ops = 5.6 billion ops. 
        # NumPy can do ~100M-1G ops/sec. So ~5-50 seconds. 
        # This is feasible.
        
        current_network = network[:]
        # We need to be careful with the history if we remove items.
        # If we remove item k, the indices of subsequent items in the original 
        # history don't change, but their position in the network changes.
        
        # Let's use a different representation for the removal phase:
        # We keep the original history.
        # We maintain a list of "active" comparators.
        # Initially all active.
        
        # This is getting complicated. Let's simplify.
        # We will perform "Local Search" on the network structure directly.
        
        # 1. Start with Odd-Even Transposition.
        # 2. Verify it is a sorting network (it is).
        # 3. Attempt to remove comparators.
        
        # To make removal fast, we will use a "Forward-Backward" check.
        # But let's try a simpler heuristic first: 
        # "Randomized Greedy Removal"
        
        # We will try to remove comparators in a random order.
        # For each removal, we re-verify the entire network.
        # Re-verification cost: M * N * SIZE / 64 (if using bit ops) or M * N * SIZE (byte ops).
        # With NumPy byte ops: M * 18 * 262144.
        # If M=100, 100 * 18 * 262144 ~ 470 million byte ops.
        # NumPy is fast. ~0.5s per verification.
        # We can do ~100 verifications in 50s.
        
        # Let's implement a loop that tries to remove random comparators.
        
        # Reset current_state to initial
        current_state = states.copy()
        
        def verify_network(net):
            """Apply network to initial states and check if sorted."""
            st = states.copy()
            for i, j in net:
                si = st[i].copy()
                sj = st[j].copy()
                st[i] = si & sj
                st[j] = si | sj
            return _is_sorted(st)
        
        # Initial check
        if not verify_network(current_network):
            # Should not happen
            return current_network
            
        best_network = current_network[:]
        best_len = len(current_network)
        
        # Optimization: 
        # Instead of re-verifying from scratch every time, we can use the 
        # "prefix/suffix" property if we only remove from the end.
        # But random removal is more powerful.
        
        # Let's try a mix: 
        # 1. Aggressive removal from the end (using suffix logic if possible, 
        #    or just re-verification).
        # 2. Random local search.
        
        # Given the time constraint, re-verification is the safest bet.
        
        # We will iterate through multiple passes.
        # In each pass, we try to remove each comparator (from end to start).
        # If removal keeps it sorted, we remove it.
        # We repeat until no more removals are possible.
        
        # To speed up, we can stop early if we find a removal.
        
        removed_any = True
        while removed_any and time.time() < deadline - 5:
            removed_any = False
            # Iterate from end to start to avoid index shifting issues
            for idx in range(len(current_network) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                # Try to remove current_network[idx]
                # Save current network
                old_net = current_network[:]
                current_network.pop(idx)
                
                # Verify
                if verify_network(current_network):
                    # Success
                    removed_any = True
                    if len(current_network) < best_len:
                        best_network = current_network[:]
                        best_len = len(current_network)
                else:
                    # Failure, restore
                    current_network = old_net
                    
    except MemoryError:
        # Fallback to simpler method if history allocation fails
        pass
        
    return best_network

# Keep the seed function name if required, though construct is the main entry
def _seed_network():
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network
