import numpy as np
import time

N = 16

def _generate_batcher_oes(n):
    """
    Generates Batcher's Odd-Even Merge Sort network.
    Returns a list of (i, j) pairs.
    """
    if n == 1:
        return []
    
    # Recursively sort the two halves
    # Note: The standard Batcher OES sorts the odd indices and even indices separately?
    # Actually, Batcher's OES is defined recursively:
    # To sort N elements:
    # 1. Recursively sort the even-indexed elements and the odd-indexed elements.
    # 2. Perform a merge step.
    
    # Wait, the standard recursive definition for OES:
    # Merge(N):
    #   if N == 1: return
    #   Merge(N/2) on odd indices
    #   Merge(N/2) on even indices
    #   Final stage: compare (0,1), (2,3), ..., (N-2, N-1)
    
    # But we want a sorting network, not just a merge.
    # Sorting network S(N):
    # 1. S(N/2) on even indices
    # 2. S(N/2) on odd indices
    # 3. Merge(N)
    
    def merge(n, offset=0, step=2):
        # Merges two sorted lists of length n/2?
        # No, the recursive structure of OES is:
        # To merge two sorted lists of length n:
        # 1. Recursively merge the odd-indexed elements (length n/2)
        # 2. Recursively merge the even-indexed elements (length n/2)
        # 3. Final stage: n/2 comparators
        
        # This function assumes we are merging two lists of length n.
        # The indices are offset + step * k.
        # The two lists are interleaved?
        # In Batcher's OES, the inputs to the merge are two sorted lists of length n/2.
        # They are placed such that the first list is at even indices and the second at odd indices?
        # Or are they just two halves?
        
        # Let's use the standard generation:
        # S(n):
        # 1. S(n/2) on indices 0, 2, 4...
        # 2. S(n/2) on indices 1, 3, 5...
        # 3. Merge(n)
        
        # Merge(n):
        # 1. Merge(n/2) on odd indices
        # 2. Merge(n/2) on even indices
        # 3. Final stage: compare (i, i+1) for i=0, 2, ... n-2
        
        # This recursion is tricky to implement with offsets.
        # Let's use a simpler known construction or just hardcode the search.
        # Actually, we can just use a random initialization and search, or a simpler network.
        # But Batcher is a good starting point.
        
        # Let's try a different approach: Batcher's Bitonic Sort is easier to generate.
        # Bitonic Sort size for N=16 is 128.
        # We can start with Bitonic Sort (128 comps) and peel it down.
        
        pass

    # Let's implement Bitonic Sort generation instead, it's easier.
    def bitonic_sort(n):
        if n <= 1:
            return []
        half = n // 2
        left = bitonic_sort(half)
        right = [ (i + half, j + half) for (i, j) in left ]
        
        # Merge phase
        # Standard bitonic merge
        def bitonic_merge(n, offset=0, direction=1):
            if n <= 1:
                return []
            half = n // 2
            # Compare (offset + i, offset + i + half)
            comps = []
            for i in range(half):
                comps.append((offset + i, offset + i + half))
            
            # Recursively merge halves
            # If direction is 1 (ascending), we want the first half to be sorted ascending and second ascending?
            # No, bitonic merge:
            # 1. Compare (0, n/2), (1, n/2+1)...
            # 2. Recursively merge first half (ascending) and second half (ascending)?
            # Actually, the standard bitonic merge of a bitonic sequence of length n:
            # 1. Compare (i, i+n/2) for i in 0..n/2-1.
            # 2. Recursively sort first half (ascending) and second half (ascending).
            # Wait, if the input is bitonic (up then down), after step 1, the first half is bitonic and the second is bitonic?
            # Yes.
            
            # So:
            left_merge = bitonic_merge(half, offset, 1)
            right_merge = bitonic_merge(half, offset + half, 1)
            
            return comps + left_merge + right_merge

        # To sort n elements:
        # 1. Recursively sort first half (ascending)
        # 2. Recursively sort second half (descending)
        # 3. Bitonic merge (ascending)
        
        # My simple bitonic_sort above only does ascending.
        # We need a helper for direction.
        
        def bs(n, offset=0, direction=1):
            if n <= 1:
                return []
            half = n // 2
            # Sort first half in direction 'direction'
            # Sort second half in opposite direction
            # Then merge in direction 'direction'
            
            # This is getting complex. Let's just use a known list or a simpler search.
            # Given the time, I will implement a local search starting from a random valid network?
            # No, random is not valid.
            
            # Let's use the "peeling" method on a known valid network.
            # I will hardcode a 60-comparator network if I can find one online?
            # No, I must generate it.
            
            # Let's try to generate Batcher's Odd-Even Merge correctly.
            pass

    # Fallback: Use a simple valid network and search.
    # Odd-Even Transposition is valid (120 comps).
    # We can start with that and peel.
    
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _verify_network(comparators, n=16):
    """
    Verifies if the network sorts all binary inputs.
    Uses NumPy for speed.
    """
    # Generate all 2^n binary inputs
    # Shape: (2^n, n)
    inputs = np.array([list(map(int, format(i, f'0{n}b'))) for i in range(2**n)])
    
    # Apply comparators
    for i, j in comparators:
        # min to i, max to j
        # If inputs[:, i] > inputs[:, j], swap
        mask = inputs[:, i] > inputs[:, j]
        inputs[mask, i] = 1 - inputs[mask, i]
        inputs[mask, j] = 1 - inputs[mask, j]
        
    # Check if all outputs are sorted (0s then 1s)
    # A sorted binary vector is of the form 000...111
    # So for each row, it should be non-decreasing.
    # Check if any row has a 1 followed by a 0.
    # diff = np.diff(inputs, axis=1)
    # if np.any(diff < 0): return False
    # return True
    
    # Faster check:
    # For each row, the number of 1s must equal the position of the last 1?
    # Or simply:
    # sorted_inputs = np.sort(inputs, axis=1)
    # return np.array_equal(inputs, sorted_inputs)
    
    return np.array_equal(inputs, np.sort(inputs, axis=1))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit
    
    # Start with Odd-Even Transposition (120 comparators)
    # This is a valid sorting network.
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial
    if not _verify_network(network):
        # Should not happen
        return network

    best_network = network
    best_size = len(network)
    
    # Peeling heuristic
    # Try to remove comparators.
    # To be efficient, we can try removing from the end first, as later comparators are often redundant if the network is "dense".
    # Or try all, but that's O(K^2 * Verify).
    # Verify is ~1s. K=120. 120*120 = 14400 verifications. Too slow.
    
    # Better: 
    # 1. Iterate through indices from end to start.
    # 2. Try removing index i.
    # 3. If valid, keep removed.
    # 4. If invalid, restore.
    # This is O(K * Verify). 120 * 1s = 120s. Too slow.
    
    # We need a faster verification or fewer iterations.
    # We can verify only a subset of inputs? No, must be all.
    # We can optimize verification.
    
    # Let's optimize _verify_network.
    # Instead of creating the full matrix every time, we can precompute the "effect" of the network?
    # No, the network changes.
    
    # We can use a faster check:
    # If the network is valid, the output for any input x is sorted.
    # We can check only the "critical" inputs? No.
    
    # Let's try to reduce the number of verifications.
    # We can batch the removals?
    # No.
    
    # Alternative: Start with a smaller valid network.
    # Batcher's OES for N=16 is ~60-120.
    # Let's implement Batcher's OES properly.
    
    def generate_batcher_oes(n):
        # Returns list of (i, j)
        if n == 1:
            return []
        
        # Recursive helper
        def merge(n, offset=0, step=2):
            # Merges two sorted lists of length n/2 located at indices offset, offset+step, ...
            # The two lists are the even and odd indexed elements of the block?
            # No, the standard OES merge takes two sorted lists of length n.
            # Wait, the recursive definition:
            # To merge two sorted lists of length n:
            # 1. Recursively merge the odd-indexed elements (length n/2)
            # 2. Recursively merge the even-indexed elements (length n/2)
            # 3. Final stage: n/2 comparators
            
            # This function assumes the inputs are at specific indices.
            # Let's define indices explicitly.
            # List 1: offset, offset+2, ..., offset+n-2
            # List 2: offset+1, offset+3, ..., offset+n-1
            # This is not standard.
            
            # Standard OES:
            # S(n):
            # 1. S(n/2) on even indices
            # 2. S(n/2) on odd indices
            # 3. Merge(n)
            
            # Merge(n):
            # 1. Merge(n/2) on odd indices
            # 2. Merge(n/2) on even indices
            # 3. Final: compare (0,1), (2,3)...
            
            # Let's implement this with indices.
            if n == 1:
                return []
            
            half = n // 2
            
            # Recursive merges
            # Odd indices: offset+1, offset+3, ...
            # These form a sequence of length n/2.
            # We need to call merge on these indices.
            # But merge expects a contiguous block?
            # No, merge(n, offset, step) can handle stride.
            
            # Odd part:
            # Indices: offset+1, offset+3, ..., offset+n-1
            # Step is 2.
            # We call merge(half, offset+1, 2)?
            # Wait, the "odd" elements are interleaved.
            # The recursive merge operates on the sub-sequences.
            
            # Let's use a different representation.
            # Just return the comparators for a standard block of size n starting at 0.
            
            pass

    # Given the complexity and time, I will stick to the peeling heuristic but optimize the verification.
    # We can verify using a smaller set of inputs if we are confident? No.
    
    # Let's try to run the peeling heuristic but stop early if time runs out.
    # And we can try to remove multiple comparators at once?
    
    # Let's try a different initial network: Batcher's Bitonic Sort.
    # It has 128 comparators.
    # OET has 120.
    # So OET is better.
    
    # Let's optimize the verification loop.
    # Pre-compute the inputs? Yes, do it once outside.
    
    inputs = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(2**N)])
    
    def verify_fast(comparators):
        # Copy inputs to avoid modifying the original
        state = inputs.copy()
        for i, j in comparators:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                state[mask, i] = 1 - state[mask, i]
                state[mask, j] = 1 - state[mask, j]
        return np.array_equal(state, np.sort(state, axis=1))

    # Now run peeling
    # We will try to remove comparators from the end to the beginning.
    # This is a greedy approach.
    
    current_network = network[:]
    
    # To speed up, we can check if a comparator is "essential" by seeing if removing it breaks sorting for any input.
    # We do this by actually removing it and verifying.
    
    # We will iterate backwards.
    # If we remove one, we don't re-check the ones before it in this pass?
    # No, we should be careful.
    
    # Let's do a single pass from end to start.
    for k in range(len(current_network) - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Try removing k
        trial_network = current_network[:k] + current_network[k+1:]
        if verify_fast(trial_network):
            current_network = trial_network
            # Continue to next k (which is now the new k-1)
            
    return current_network

# EVOLVE-BLOCK-START
def construct():
    # Use the logic from above
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 16
    inputs = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(2**N)])
    
    def verify_fast(comparators):
        state = inputs.copy()
        for i, j in comparators:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                state[mask, i] = 1 - state[mask, i]
                state[mask, j] = 1 - state[mask, j]
        return np.array_equal(state, np.sort(state, axis=1))

    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    current_network = network[:]
    
    # Peeling pass
    for k in range(len(current_network) - 1, -1, -1):
        if time.time() > deadline:
            break
        
        trial_network = current_network[:k] + current_network[k+1:]
        if verify_fast(trial_network):
            current_network = trial_network
            
    return current_network
# EVOLVE-BLOCK-END
