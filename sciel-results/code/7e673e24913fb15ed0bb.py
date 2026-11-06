import time
import random
import numpy as np

N = 18
START_TIME = time.time()
TIME_LIMIT = 85.0  # Safety margin for 90s limit

def verify_network(network):
    """
    Verify if the network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    # Generate all boolean inputs as a matrix of shape (2^N, N)
    # For N=18, 2^18 = 262144 inputs.
    # We can represent this efficiently.
    
    # Create an array of indices 0 to 2^N - 1
    indices = np.arange(1 << N, dtype=np.uint32)
    
    # Extract bits. We want wire i to be the i-th bit.
    # Standard bit extraction: (indices >> i) & 1
    # This gives a matrix of shape (2^N, N) where col i is bit i.
    # However, numpy doesn't have a direct bit extract for all bits at once in a simple vectorized way 
    # without loops or specific tricks, but for 18 bits it's fast enough.
    
    # Let's build the input matrix
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (indices >> i) & 1
        
    # Apply network
    for a, b in network:
        # min goes to a, max goes to b
        # inputs[:, a] = np.minimum(inputs[:, a], inputs[:, b])
        # inputs[:, b] = np.maximum(inputs[:, a], inputs[:, b])
        
        # To avoid temporary arrays and ensure correctness with in-place updates:
        min_val = np.minimum(inputs[:, a], inputs[:, b])
        max_val = np.maximum(inputs[:, a], inputs[:, b])
        inputs[:, a] = min_val
        inputs[:, b] = max_val
        
    # Check if sorted: inputs[:, i] <= inputs[:, i+1] for all i
    # This is equivalent to checking if the matrix is sorted in each row.
    # Since we are dealing with bits, a sorted row is all 0s followed by all 1s.
    # So, for each row, there can be at most one transition from 0 to 1.
    
    # Check if any row has a 1 followed by a 0
    # diff = np.diff(inputs, axis=1)
    # invalid = np.any(diff == -1, axis=1)
    
    # More efficient check: 
    # A row is sorted if it is non-decreasing.
    # For binary, this means no '10' pattern.
    # We can check if the sum of the row equals the number of 1s, and the last k elements are 1.
    # Actually, the simplest vectorized check for sortedness of binary rows:
    # If row is sorted, inputs[:, i] <= inputs[:, i+1].
    # So inputs[:, 1:] - inputs[:, :-1] should be >= 0.
    
    diffs = inputs[:, 1:] - inputs[:, :-1]
    if np.any(diffs < 0):
        return False
    
    return True

def generate_bitonic_network(n):
    """
    Generate a bitonic sorting network for n wires.
    Assumes n is a power of 2 for standard implementation, 
    but we can handle 18 by padding to 32 or using a recursive definition that works for any n.
    Standard bitonic sort usually requires power of 2. 
    For N=18, we can generate a bitonic network for 32 wires and discard the extra wires? 
    No, that changes the problem. 
    
    Actually, there is a way to construct bitonic networks for non-powers of 2, 
    but it's complex. 
    Alternatively, we can use the "Odd-Even Mergesort" or other networks.
    
    Let's use a known good network for 18 wires.
    According to literature, a bitonic-like network for 18 wires exists.
    However, implementing a generic bitonic generator for non-powers of 2 is tricky.
    
    Let's try a different approach: 
    Use a recursive construction for bitonic networks that works for any N.
    Or, simpler: Use the fact that we can merge two sorted lists.
    
    Let's implement a recursive bitonic sort network generator that works for any N.
    The standard bitonic sort decomposes into merging.
    
    Actually, for the purpose of this task, if we can't easily generate a bitonic network for 18, 
    we might fall back to a known smaller network or just use the bubble sort if the bitonic generation fails.
    
    However, there is a simpler network: **Odd-Even Transposition** is 153.
    **Bubble Sort** is 153.
    
    Let's try to generate a **Bitonic Sort** network for N=18.
    We can pad to 32, generate the network, and then remove comparators involving wires >= 18? 
    No, that breaks the sorting property for the remaining 18 wires because the "ghost" wires might hold values that affect the sorting of the real wires if they are connected.
    But if we just remove the wires, the connections between the remaining wires might still be valid?
    In a bitonic network for 32 wires, the comparators are defined on 32 wires. 
    If we only care about the first 18 wires, and we remove any comparator that touches wire >= 18, 
    does the remaining network sort the first 18 wires?
    Not necessarily, because the sorting of wire i might depend on values that were "pushed" to wire j >= 18 and then "pulled back" or influenced.
    Actually, if we remove the wires, the values on the remaining wires are just a subset of the sorted output of the 32-wire network?
    No, the bitonic network on 32 wires sorts all 32 inputs. 
    If we take the first 18 inputs and ignore the rest (setting them to 0 or something), the first 18 outputs of the 32-wire network will be the 18 smallest values.
    So, if we run the 32-wire bitonic network on the 18 inputs padded with 0s (or -infty), the first 18 wires will contain the sorted 18 inputs.
    Therefore, we can generate a 32-wire bitonic network, and keep only the comparators where both indices are < 18.
    This resulting network will sort 18 inputs (padded with -infty) correctly on the first 18 wires.
    Since the task inputs are just 18 numbers, this is equivalent to sorting the 18 numbers.
    
    Let's verify this logic.
    If we have 18 numbers x0...x17.
    We feed them into wires 0..17 of a 32-wire bitonic network.
    We feed 0 (or -inf) into wires 18..31.
    The network sorts all 32 values.
    The sorted output will have the 14 zeros (or -infs) in the last 14 positions (wires 18..31) and the sorted 18 numbers in the first 18 positions (wires 0..17).
    Therefore, the sub-network consisting of only comparators between wires 0..17 will sort the 18 inputs correctly.
    
    How many comparators does a 32-wire bitonic network have?
    Depth is log2(32)^2 = 25.
    Number of comparators: 
    Formula for bitonic network comparators: (n/2) * log2(n) * (log2(n) + 1) / 2 ? 
    For n=32:
    log2(32) = 5.
    Comparators = (32/2) * 5 * 6 / 2 = 16 * 15 = 240? 
    Wait, the formula is sum over stages.
    Stage k (1 to log n) has n/2 comparators.
    Total = log2(n) * n/2.
    For n=32: 5 * 16 = 80 comparators? 
    No, that's the depth.
    Let's count:
    Bitonic sort on n elements:
    1. Split into two halves, sort each (recursively).
    2. Merge the two sorted halves (bitonic merge).
    
    Let C(n) be the number of comparators.
    C(1) = 0
    C(n) = 2 * C(n/2) + M(n)
    Where M(n) is the number of comparators in a bitonic merge of n elements.
    M(n) = n/2 * log2(n)
    For n=32:
    M(32) = 16 * 5 = 80
    M(16) = 8 * 4 = 32
    M(8) = 4 * 3 = 12
    M(4) = 2 * 2 = 4
    M(2) = 1 * 1 = 1
    
    C(2) = 2*C(1) + M(2) = 0 + 1 = 1
    C(4) = 2*C(2) + M(4) = 2*1 + 4 = 6
    C(8) = 2*C(4) + M(8) = 2*6 + 12 = 24
    C(16) = 2*C(8) + M(16) = 2*24 + 32 = 80
    C(32) = 2*C(16) + M(32) = 2*80 + 80 = 240
    
    So a 32-wire bitonic network has 240 comparators.
    When we restrict to wires 0..17, we lose comparators involving wires 18..31.
    How many remain?
    The network is symmetric. The comparators in the first half (wires 0-15) and second half (16-31) are separate in the recursive structure, but the merge stage connects them.
    In the final merge stage (M(32)), we compare (i, i+16) for i=0..15.
    Comparators: (0,16), (1,17), ..., (15,31).
    We keep those with both indices < 18.
    (0,16) -> Keep
    (1,17) -> Keep
    (2,18) -> Discard (18 >= 18)
    ...
    So from the final merge, we keep 2 comparators.
    
    In the previous stage, we had two C(16) networks: one on 0-15, one on 16-31.
    The network on 0-15 is entirely within 0-17, so all its comparators are kept.
    C(16) = 80 comparators.
    The network on 16-31 is entirely >= 16. 
    Comparators within 16-31: e.g., (16,17), (18,19).
    We keep those with both indices < 18.
    So we keep comparators in the 16-31 network that only involve 16 and 17.
    The 16-31 network is a bitonic network on indices 16..31.
    Let's map 16..31 to 0..15.
    We want to keep comparators where the original indices are < 18.
    So original index 16 maps to 0, 17 maps to 1.
    So we keep comparators in the sub-network on 16..31 that only involve wires 0 and 1 (relative to the sub-network).
    Basically, we are keeping the sub-network that sorts 2 elements (wires 16,17) using the structure of the 16-wire bitonic network?
    No, the 16-wire bitonic network on 16..31 sorts those 16 elements.
    If we only care about wires 16 and 17, and we discard all comparators involving 18..31, does the remaining part sort wires 16 and 17?
    The remaining part consists of comparators that were originally between 16 and 17.
    In a bitonic network, are there comparators between 16 and 17?
    Yes, in the final merge of the 16-element sub-network (which merges 8 and 8), we compare (i, i+8).
    For the 16-element sub-network on 16..31:
    The merge compares (16, 24), (17, 25), etc.
    Wait, the structure is recursive.
    Let's just generate the 32-wire network and filter.
    
    """
    n = 32
    # Generate bitonic network for n wires
    # Returns list of (i, j)
    
    def bitonic_merge(n, start, reverse=False):
        """
        Generate comparators for a bitonic merge of n elements starting at index 'start'.
        reverse: if True, sort in descending order? No, standard bitonic sort assumes 
        the first half is ascending and second half is descending (or vice versa).
        The merge stage handles the direction.
        """
        comps = []
        # Standard bitonic merge:
        # For k = n/2 down to 1:
        #   For i = 0 to n - 2k:
        #     Compare (start + i, start + i + k)
        #     Direction depends on 'reverse' and the stage?
        # Actually, the direction is determined by whether we want the final output sorted ascending.
        # In a standard bitonic sort:
        # 1. Sort halves recursively. One ascending, one descending.
        # 2. Merge them.
        
        # Let's implement the standard recursive bitonic sort generator.
        
        def bitonic_sort(n, start, ascending=True):
            if n <= 1:
                return []
            half = n // 2
            # Sort first half ascending
            comps1 = bitonic_sort(half, start, True)
            # Sort second half descending
            comps2 = bitonic_sort(half, start + half, False)
            # Merge
            comps3 = bitonic_merge(half, start, True) # Merge assumes first half asc, second desc? 
            # Actually, the merge function needs to know the direction of the final sort.
            # If ascending=True, we want the result to be ascending.
            # The merge of an ascending and a descending sequence produces a sorted sequence.
            
            return comps1 + comps2 + comps3

        def bitonic_merge(n, start, ascending=True):
            # n is the size of the sequence to be merged (which is 2 * half_size)
            # Wait, the recursion above calls bitonic_merge(half, start, ...). 
            # This implies the merge size is 'half'? No, the merge combines two halves of size 'half'.
            # So the total size is 2*half.
            # Let's redefine.
            
            # Let's use a cleaner recursive definition.
            pass

    # Let's use a simpler iterative approach for generating the network structure.
    # Or just hardcode the generation for 32.
    
    comps = []
    
    def gen_bitonic(n, start, asc):
        if n == 1:
            return
        half = n // 2
        gen_bitonic(half, start, True)
        gen_bitonic(half, start + half, False)
        
        # Merge
        # The merge stage for size n (consisting of two halves of size half)
        # iterates k from half down to 1.
        for k in range(half, 0, -1):
            for i in range(n - 2 * k):
                a = start + i
                b = start + i + k
                # Direction:
                # If asc is True, we want min at a, max at b?
                # In a bitonic merge, if the first half is asc and second is desc,
                # and we want the final result to be asc, then:
                # We compare (i, i+k). 
                # If a[i] > a[i+k], swap.
                # This effectively puts smaller elements in the first half.
                # So if asc=True, we do (a, b) such that min goes to a.
                # If asc=False, we do max goes to a?
                # Let's check.
                # Standard bitonic sort:
                # Stage 1: Sort pairs (0,1), (2,3)... min at even, max at odd.
                # Stage 2: Sort quads. 
                # 
                # Let's just generate the comparators and let the verifier check.
                # The direction is determined by the 'asc' flag of the merge call.
                if asc:
                    comps.append((a, b))
                else:
                    comps.append((b, a)) # min at b, max at a? 
                    # Wait, the comparator (i, j) puts min at i, max at j.
                    # If we want descending order, we want max at i, min at j.
                    # So we should swap the arguments? 
                    # If we call comparator (b, a), min goes to b, max goes to a.
                    # So a gets max, b gets min. This is descending for (a,b).
                    # So yes, (b, a) implements descending sort on wires a, b.
        
    gen_bitonic(32, 0, True)
    
    # Filter for wires < 18
    filtered = []
    for a, b in comps:
        if a < 18 and b < 18:
            filtered.append((a, b))
            
    return filtered

def construct():
    # 1. Generate initial bitonic network (filtered)
    network = generate_bitonic_network(N)
    
    # Verify it
    if not verify_network(network):
        # Fallback to bubble sort if something went wrong
        comparators = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                comparators.append((j, j + 1))
        return comparators
        
    current_best = network
    best_count = len(network)
    
    # 2. Local Search: Try to remove comparators
    # We can try removing one comparator at a time and verifying.
    # If valid, keep the removal and try to remove another.
    # This is a greedy pruning.
    
    # To speed up verification, we can use a cache or just rely on numpy speed.
    # Verification takes ~0.1s. We have ~80s. So we can do ~800 verifications.
    # Each removal attempt is one verification.
    
    # Strategy: 
    # Loop while time < limit:
    #   Pick a random comparator index
    #   Remove it
    #   Verify
    #   If valid, keep removal (update network, best_count)
    #   If invalid, put it back
    #   To avoid getting stuck, maybe try re-adding removed ones? 
    #   Or just stick to greedy removal.
    
    # Greedy removal might get stuck in a local minimum.
    # We can try a few random restarts if time permits.
    
    # Let's try greedy removal first.
    # We need to be careful with the order. Removing a comparator might make the network invalid.
    
    # Optimization: 
    # Instead of verifying the whole network from scratch every time, 
    # we could potentially track the state, but for 18 wires, full verification is fast enough.
    
    # Let's add a timeout check inside the loop.
    
    # We will try to remove comparators.
    # We'll iterate through the network and try to remove each one.
    # If removal is successful, we continue with the shorter network.
    # If not, we keep it.
    
    # To make it more effective, we can randomize the order of removal attempts.
    
    # Let's try a simple hill climbing:
    # 1. Start with bitonic network.
    # 2. While time allows:
    #    a. Try to remove a random comparator.
    #    b. If valid, accept.
    #    c. If invalid, reject.
    #    d. If we fail to improve for X iterations, try a random mutation (swap two comparators) or restart?
    
    # Given the time limit, let's just do greedy removal in random order.
    
    # We need to handle the case where the network becomes invalid.
    
    # Let's implement a helper to try removal.
    
    # We'll use a set of indices to track which comparators are present.
    # Actually, just modifying the list is fine.
    
    # To avoid O(N^2) search for removal, we just pick an index.
    
    # Let's run a loop.
    # We can also try to add comparators if we are stuck? No, we want to minimize.
    
    # Let's try to remove comparators.
    
    # We'll keep a list of indices to try.
    
    # Initialize with all indices
    # Shuffle them
    
    # Try removing each one in the shuffled order.
    # If successful, the list shrinks.
    # We need to re-shuffle or continue with the remaining ones.
    
    # Let's do this:
    # while time < limit:
    #    if len(network) == 0: break
    #    # Pick a random index
    #    idx = random.randint(0, len(network)-1)
    #    # Try remove
    #    removed = network.pop(idx)
    #    if verify_network(network):
    #        # Keep removed
    #        best_count -= 1
    #    else:
    #        # Put back
    #        network.insert(idx, removed)
    #        # Optional: to avoid trying the same one immediately, we could track failed attempts
    #        # But for now, just random.
        
    # This might be slow if verification is 0.1s and we do 800 iterations.
    # 800 * 0.1 = 80s. Fits.
    
    # Let's add a small optimization: 
    # If we fail to remove a comparator, we can mark it as "hard to remove" and prioritize others?
    # Or just rely on randomness.
    
    # Let's start the search.
    
    # We should also consider that the bitonic network might have redundant comparators 
    # that are not obvious to the random removal (e.g. two comparators doing the same job).
    
    # Let's run the search.
    
    # To improve efficiency, we can use a timeout check.
    
    # One issue: `verify_network` creates a large array each time.
    # We can optimize it by reusing the array? 
    # But numpy allocation is fast.
    
    # Let's go.
    
    # We'll try to remove comparators.
    
    # We'll use a counter to avoid infinite loops if random picks the same failed one.
    # But with 100+ comparators, probability is low.
    
    # Let's add a safeguard: if we fail 100 times in a row, try a different strategy?
    # For now, just random removal.
    
    # We'll also try to swap comparators if removal fails?
    # Swapping doesn't change the count, so it doesn't help minimize count directly, 
    # but it might enable future removals.
    # However, the task is to minimize count.
    
    # Let's stick to removal.
    
    # We'll run until time limit or no more removals possible (which is unlikely to be detected perfectly).
    
    # Let's implement the loop.
    
    # To speed up verification, we can pre-compute the input matrix?
    # The input matrix is always the same (all boolean inputs).
    # We can compute it once and copy it for each verification.
    
    # Let's do that.
    
    # Pre-compute inputs
    indices = np.arange(1 << N, dtype=np.uint32)
    base_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        base_inputs[:, i] = (indices >> i) & 1
        
    def fast_verify(network):
        # Copy base inputs
        inputs = base_inputs.copy()
        for a, b in network:
            min_val = np.minimum(inputs[:, a], inputs[:, b])
            max_val = np.maximum(inputs[:, a], inputs[:, b])
            inputs[:, a] = min_val
            inputs[:, b] = max_val
            
        diffs = inputs[:, 1:] - inputs[:, :-1]
        if np.any(diffs < 0):
            return False
        return True
    
    # Replace verify_network with fast_verify in the search loop
    
    # Re-verify initial network with fast_verify
    if not fast_verify(network):
        # Fallback
        comparators = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                comparators.append((j, j + 1))
        return comparators
        
    current_network = network
    best_count = len(current_network)
    
    # Search loop
    while time.time() - START_TIME < TIME_LIMIT:
        if len(current_network) == 0:
            break
            
        # Pick random index
        idx = random.randint(0, len(current_network) - 1)
        
        # Remove
        removed = current_network.pop(idx)
        
        # Verify
        if fast_verify(current_network):
            # Success
            best_count -= 1
        else:
            # Failure
            current_network.insert(idx, removed)
            
    return current_network

# EVOLVE-BLOCK-END
