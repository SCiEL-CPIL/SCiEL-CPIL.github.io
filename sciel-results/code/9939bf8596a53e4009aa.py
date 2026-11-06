import numpy as np
import time
import random

N = 17

# EVOLVE-BLOCK-START

def _batcher_odd_merge(n):
    """
    Generate Batcher's Odd-Merge sorting network for n inputs.
    Returns a list of (i, j) pairs.
    """
    if n == 1:
        return []
    
    comps = []
    
    def odd_merge(n):
        if n == 1:
            return []
        if n % 2 == 1:
            # Recursively sort odd and even indices
            # Odd indices: 0, 2, 4...
            # Even indices: 1, 3, 5...
            # This is tricky to map directly to indices without a helper.
            # Standard recursive definition:
            # 1. Sort odd indices (0, 2, ...) and even indices (1, 3, ...) recursively.
            # 2. Merge the results.
            # For n=17:
            # Odd indices (0-based 0, 2, 4, 6, 8, 10, 12, 14, 16) -> 9 elements.
            # Even indices (0-based 1, 3, 5, 7, 9, 11, 13, 15) -> 8 elements.
            
            # We need a helper that generates comparators for a specific subset of wires.
            pass
        else:
            pass
        return comps

    # This recursive implementation is complex to get right with index mapping.
    # Alternative: Use a known smaller network or a simpler construction.
    # Let's use the "0-1 principle" to verify, but start with a known good network.
    # Batcher's Merge is well-documented.
    
    # Let's implement a simpler recursive Batcher merge.
    # f(n) = comparators for n inputs.
    # f(1) = 0
    # f(2) = 1
    # f(n) for n even: 
    #   Split into two halves n/2.
    #   Sort first half, sort second half.
    #   Merge them.
    # Merge of two sorted lists of size m:
    #   If m=1, 1 comparator.
    #   If m>1:
    #     Odd merge of first halves, even merge of second halves?
    
    # Let's use a library-like implementation if possible, or hardcode for N=17?
    # No, we need to generate it.
    
    # I will implement a correct Batcher Odd-Merge.
    
    def merge(comps, offset, n1, n2):
        """
        Merge two sorted networks of sizes n1 and n2, starting at offset.
        n1 is the size of the first list (wires offset to offset+n1-1).
        n2 is the size of the second list (wires offset+n1 to offset+n1+n2-1).
        Assumes n1 == n2 for standard merge, but Odd-Merge handles odd sizes.
        """
        if n1 == 0 or n2 == 0:
            return
        if n1 == 1 and n2 == 1:
            comps.append((offset, offset + 1))
            return
        
        # Standard Batcher Merge assumes equal sizes. 
        # Odd-Merge is for odd total sizes.
        # If total N is odd:
        #   Split into N1 = N//2, N2 = N - N1.
        #   Sort first N1, Sort last N2.
        #   Merge them.
        
        # This function assumes the inputs to merge are already sorted internally.
        # The merge network for two sorted lists of size m is:
        # 1. Merge odd positions of the two lists.
        # 2. Merge even positions of the two lists.
        
        # For equal sizes m:
        #   Odd merge: merge wires (offset, offset+m), (offset+1, offset+m+1) ... ?
        #   Actually, the "Odd-Merge" step for equal size m:
        #   Compare (offset, offset+m), (offset+1, offset+m+1) is not right.
        
        # Let's look up Batcher's Odd-Merge algorithm precisely.
        # 1. If n is odd:
        #    a. Recursively sort the odd-indexed inputs (0, 2, ...) and even-indexed inputs (1, 3, ...).
        #    b. Merge the two resulting sorted lists.
        # 2. If n is even:
        #    a. Split into two equal halves.
        #    b. Recursively sort each half.
        #    c. Merge the two halves.
        
        # The Merge step (for two sorted lists of equal size m):
        #   - Compare (offset, offset+m), (offset+1, offset+m+1), ..., (offset+m-1, offset+2m-1) ? No.
        #   - The merge of two sorted lists of size m is done by:
        #     1. Merging the "odd" positions of the two lists.
        #     2. Merging the "even" positions of the two lists.
        #     This is recursive.
        
        # Let's implement a helper `merge_sort(comps, offset, size)` that sorts wires [offset, offset+size).
        
        def sort_network(comps, offset, size):
            if size <= 1:
                return
            if size % 2 == 0:
                half = size // 2
                sort_network(comps, offset, half)
                sort_network(comps, offset + half, half)
                merge_sorted(comps, offset, half)
            else:
                # Odd size
                # Split into odd and even indices
                # Odd indices in local coords: 0, 2, 4... -> global: offset, offset+2...
                # Even indices in local coords: 1, 3, 5... -> global: offset+1, offset+3...
                
                # Count of odd indices: (size+1)//2
                # Count of even indices: size//2
                n_odd = (size + 1) // 2
                n_even = size // 2
                
                # We cannot simply call sort_network on non-contiguous wires.
                # We need to map the recursive calls to the specific wires.
                
                # Helper to get wire index for local index k in a subset defined by step
                def get_wire(local_idx, start, step):
                    return offset + start + local_idx * step
                
                # This approach is getting complicated to implement recursively with index mapping.
                # 
                # Alternative: Use a known small network for N=17?
                # Or use a simpler construction: **0-1 Principle + Random Search from Scratch?**
                # No, random search from scratch is too slow.
                
                # Let's try a different seed: **Bitonic Sort**.
                # Bitonic sort for N=16 is easy. For N=17, we can pad to 32? No, too big.
                # Bitonic sort for N=17 isn't standard.
                
                # Let's go back to **Odd-Even Transposition** but optimize the search.
                # The previous code failed because it was slow.
                # If I pre-compute the inputs and use a fast sample check, I can do more iterations.
                
                # I will implement a **Greedy Removal** on the OET network, but with:
                # 1. Pre-computed full input matrix.
                # 2. Sample check (1000 random inputs) before full check.
                # 3. Order the removal attempts by "criticality" (e.g., try removing comparators that are redundant in the sample first).
                
                pass
        return comps

    # Given the complexity of implementing Batcher correctly from scratch in this context, 
    # and the risk of bugs, I will stick to OET but significantly optimize the search loop.
    # The key is speed.
    
    pass

def _generate_oet_network(n):
    """Generate Odd-Even Transposition Network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

# Pre-compute inputs for verification
def _get_inputs():
    indices = np.arange(2**N)
    inputs = ((indices[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)
    return inputs

def _check_network_fast(network, inputs, sample_indices=None):
    """
    Check if network sorts inputs.
    If sample_indices is provided, only check those rows (for quick rejection).
    """
    if sample_indices is not None:
        test_inputs = inputs[sample_indices].copy()
    else:
        test_inputs = inputs.copy()
        
    for i, j in network:
        min_val = np.minimum(test_inputs[:, i], test_inputs[:, j])
        max_val = np.maximum(test_inputs[:, i], test_inputs[:, j])
        test_inputs[:, i] = min_val
        test_inputs[:, j] = max_val
        
    diffs = np.diff(test_inputs, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def construct():
    """
    Construct a sorting network for 17 wires.
    Uses OET as seed, optimized greedy removal.
    """
    start_time = time.time()
    time_limit = 55.0
    
    # Generate OET
    network = _generate_oet_network(N)
    best_network = network.copy()
    best_size = len(network)
    
    # Pre-compute all inputs
    all_inputs = _get_inputs()
    
    # Pre-compute sample indices for fast check (e.g., 5% of inputs)
    sample_size = 2000
    sample_indices = np.random.choice(2**N, sample_size, replace=False)
    
    # Local Search / Greedy Removal
    # We iterate through the network and try to remove each comparator.
    # If removal works (passes sample AND full check), we keep it removed.
    # We repeat this process until no more comparators can be removed.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        current_len = len(network)
        for idx in range(current_len - 1, -1, -1): # Remove from end to start to avoid index shifting issues? 
            # Actually, removing from the end is safer for index management if we modify the list in place.
            # But we need to check if the *current* network minus this comparator is valid.
            
            # Try removing network[idx]
            if len(network) <= 1:
                break
                
            new_network = network[:idx] + network[idx+1:]
            
            # Fast check
            if _check_network_fast(new_network, all_inputs, sample_indices):
                # Full check
                if _check_network_fast(new_network, all_inputs, None):
                    network = new_network
                    improved = True
                    if len(network) < best_size:
                        best_size = len(network)
                        best_network = network.copy()
                    # Break to restart the loop from the end? 
                    # Or continue? If we continue, the indices shift.
                    # It's safer to break and restart the for loop.
                    break
        
        # If we made an improvement, we might want to do a random shuffle to escape local minima?
        # For now, stick to greedy.
        
        # If no improvement in a full pass, try a random move?
        if not improved:
            # Try random moves for a bit
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                idx = random.randint(0, len(network) - 1)
                comp = network[idx]
                new_network = network[:idx] + network[idx+1:]
                new_idx = random.randint(0, len(new_network))
                new_network = new_network[:new_idx] + [comp] + new_network[new_idx:]
                
                if _check_network_fast(new_network, all_inputs, sample_indices):
                    if _check_network_fast(new_network, all_inputs, None):
                        network = new_network
                        # Size didn't change, but structure did
                        # Don't update best_size, but keep exploring
                        break
                        
    return best_network

# EVOLVE-BLOCK-END
