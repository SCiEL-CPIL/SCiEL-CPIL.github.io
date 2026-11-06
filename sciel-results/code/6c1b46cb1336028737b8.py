import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _bitonic_network(n):
    """
    Generates Batcher's Bitonic Sort network for n wires.
    n must be a power of 2.
    Returns a list of (i, j) comparators.
    """
    if n == 1:
        return []
    
    k = n // 2
    # Recursively sort the two halves
    left = _bitonic_network(k)
    right = [(i + k, j + k) for i, j in _bitonic_network(k)]
    
    # Merge the two sorted halves into a bitonic sequence, then sort it
    # Bitonic Merge of two sorted lists of length k
    merge = _bitonic_merge(k)
    
    return left + right + merge

def _bitonic_merge(k):
    """
    Generates comparators to merge two sorted lists of length k 
    (assumed to be at 0..k-1 and k..2k-1) into a single sorted list of length 2k.
    This is the "Bitonic Merge" step.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # Step 1: Recursively merge the even-indexed and odd-indexed sub-sequences?
    # No, Bitonic Merge works by comparing elements i and i+k.
    # Actually, the standard recursive definition for Bitonic Merge (N) is:
    # 1. Compare (i, i+N/2) for all i in 0..N/2-1.
    # 2. Recursively merge the first N/2 elements and the last N/2 elements.
    
    # Wait, the inputs to Bitonic Merge are two sorted sequences:
    # A[0..k-1] (sorted ascending) and B[k..2k-1] (sorted ascending).
    # The standard Bitonic Sort generates a bitonic sequence first.
    # But here we are merging two *sorted* sequences.
    # Batcher's Bitonic Sort structure:
    # Sort(N) = Sort(N/2) + Sort(N/2) + Merge(N/2, N/2)
    # Where Merge(N) takes a bitonic sequence of length 2N?
    # No, Bitonic Sort(N):
    # 1. Sort(N/2) on first half.
    # 2. Sort(N/2) on second half (reverse direction? No, Bitonic Sort usually sorts both ascending, then merges).
    # Actually, standard Bitonic Sort:
    # 1. Sort N/2 ascending.
    # 2. Sort N/2 ascending.
    # 3. Merge the two N/2 sorted sequences into one sorted N sequence.
    
    # The Merge step for two sorted sequences of length k:
    # This is equivalent to an Odd-Even Merge? No, Bitonic Merge is different.
    # Bitonic Merge(N) where N is the length of the *combined* sequence?
    # Let's use the recursive definition for Bitonic Sort directly to avoid confusion.
    
    # Better approach: Generate the full Bitonic Sort network using the known structure.
    # C(1) = 0
    # C(2) = 1
    # C(4) = 4
    # C(8) = 19
    # C(16) = 65
    
    # Let's implement the comparator generation for Bitonic Sort N.
    # It consists of:
    # 1. Two Bitonic Sorts of size N/2.
    # 2. A Bitonic Merge of size N (which merges two sorted N/2 sequences).
    
    # How to generate Bitonic Merge of size N (merging two sorted N/2s)?
    # It is defined recursively:
    # Merge(N):
    #   if N == 1: return []
    #   # Compare (i, i + N/2) for i in 0..N/2-1
    #   # This creates two bitonic sequences of length N/2?
    #   # No, if inputs are sorted, (0, N/2), (1, N/2+1)...
    #   # After these comparisons, we have two sequences of length N/2.
    #   # Are they sorted? No, they are bitonic?
    #   # Actually, the standard Bitonic Merge algorithm:
    #   # 1. Compare (i, i+N/2) for i in 0..N/2-1.
    #   # 2. Recursively merge the first N/2 elements and the last N/2 elements.
    
    # Let's trace Merge(2) (merging two sorted 1s):
    # Compare (0, 1).
    # Recursively merge(1) on [0] and [1]. Merge(1) is empty.
    # Total: 1 comp. Correct.
    
    # Trace Merge(4) (merging two sorted 2s):
    # Compare (0, 2), (1, 3).
    # Recursively merge(2) on indices [0,1] and [2,3].
    # Merge(2) on [0,1] -> Compare (0,1).
    # Merge(2) on [2,3] -> Compare (2,3).
    # Total: 2 + 1 + 1 = 4 comps.
    # Is this correct for merging two sorted 2s?
    # Inputs: [a,b] and [c,d] with a<=b, c<=d.
    # 1. (0,2) -> min(a,c) at 0, max(a,c) at 2.
    # 1. (1,3) -> min(b,d) at 1, max(b,d) at 3.
    # 2. Merge(2) on 0,1: (0,1) -> min(min(a,c), min(b,d)) at 0, max(...) at 1.
    # 2. Merge(2) on 2,3: (2,3) -> min(max(a,c), max(b,d)) at 2, max(...) at 3.
    # This correctly sorts the 4 elements.
    # So Merge(4) has 4 comparators.
    
    # So C(4) = C(2) + C(2) + Merge(4) = 1 + 1 + 4 = 6?
    # But C(4) is known to be 4 for Bitonic Sort?
    # Wait. Bitonic Sort 4:
    # 1. Sort(2) on 0,1 -> (0,1).
    # 2. Sort(2) on 2,3 -> (2,3).
    # 3. Merge(4) on 0,1,2,3.
    # If Merge(4) is 4 comparators, total is 6.
    # But the standard Bitonic Sort 4 network is:
    # (0,1), (2,3), (0,2), (1,3).
    # This is 4 comparators.
    # Why the discrepancy?
    # Ah, the "Merge" step in Bitonic Sort usually assumes the inputs form a **bitonic** sequence, not two sorted sequences.
    # In Bitonic Sort(N):
    # 1. Sort(N/2) on first half.
    # 2. Sort(N/2) on second half.
    # Now we have two sorted sequences.
    # The Merge step must merge two sorted sequences.
    # The recursive merge I described above (Compare i, i+N/2 then recurse) works for two sorted sequences.
    # Let's re-evaluate C(4) with this merge.
    # C(4) = 1 + 1 + 4 = 6.
    # But the optimal sorting network for 4 is 5 (Odd-Even Merge).
    # Bitonic Sort 4 is 5?
    # Let's check the 4-comparator network: (0,1), (2,3), (0,2), (1,3).
    # Does (0,2), (1,3) merge two sorted 2s?
    # If 0,1 sorted and 2,3 sorted.
    # (0,2) and (1,3) is NOT a valid merge for two sorted lists.
    # Example: 0:1, 1:2, 2:3, 3:4.
    # (0,2): 1,3. (1,3): 2,4.
    # Result: 1,2,3,4. Sorted.
    # Example: 0:3, 1:4, 2:1, 3:2. (3,4) and (1,2).
    # (0,2): 1,3. (1,3): 2,4.
    # Result: 1,2,3,4. Sorted.
    # Example: 0:2, 1:3, 2:1, 3:4. (2,3) and (1,4).
    # (0,2): 1,2. (1,3): 3,4.
    # Result: 1,3,2,4. NOT sorted.
    # So (0,2), (1,3) is NOT a valid merge.
    
    # Therefore, Bitonic Sort 4 MUST use 5 comparators?
    # Standard Bitonic Sort 4:
    # 1. (0,1), (2,3).
    # 2. (0,2), (1,3).
    # 3. (0,1), (2,3).
    # Total 5?
    # Let's check the 5-comparator network:
    # (0,1), (2,3), (0,2), (1,3), (0,1), (2,3)? No.
    
    # Actually, the number of comparators in Bitonic Sort N is:
    # C(1)=0, C(2)=1, C(4)=5, C(8)=19, C(16)=65.
    # Yes, C(4)=5.
    # So my previous calculation of 6 was wrong because Merge(4) is 3 comparators?
    # If C(4)=5, and C(2)=1, then Merge(4) = 5 - 1 - 1 = 3.
    # So Merge(4) has 3 comparators.
    # Which 3?
    # (0,2), (1,3) is 2. We need 1 more.
    # The recursive merge for N=4:
    # 1. Compare (0,2), (1,3).
    # 2. Recursively merge(2) on [0,1] and [2,3].
    # Merge(2) is 1 comparator each.
    # Total 2 + 1 + 1 = 4.
    # Why is it 3?
    # Maybe the recursive step is different?
    # In Bitonic Sort, the second half is often sorted in **reverse** order to form a bitonic sequence.
    # If we sort 0,1 ascending and 2,3 descending, then (0,2) and (1,3) form a bitonic sequence?
    # No.
    
    # Let's just use the known count: C(16) = 65.
    # I will generate the network using the recursive structure that yields 65.
    
    # Correct Recursive Generation for Bitonic Sort N:
    # 1. Sort(N/2) on [0..N/2-1] (Ascending)
    # 2. Sort(N/2) on [N/2..N-1] (Ascending)
    # 3. Merge(N) on [0..N-1]
    
    # Merge(N) for two sorted halves:
    # If N == 1: return []
    # Else:
    #   # Compare (i, i + N/2) for i in 0..N/2-1
    #   # This step is key.
    #   # Then Recursively Merge(N/2) on the first half and the second half.
    
    # Let's re-verify Merge(4) with this rule.
    # N=4.
    # 1. Compare (0,2), (1,3).
    # 2. Recursively Merge(2) on [0,1] and [2,3].
    # Merge(2):
    #   N=2.
    #   1. Compare (0,1) relative to the sub-list.
    #   So (0,1) and (2,3).
    # Total 4 comparators.
    # C(4) = 1 + 1 + 4 = 6.
    
    # There is a contradiction with C(4)=5.
    # Is C(4) actually 5?
    # Yes, the minimum is 5.
    # Is Bitonic Sort 4 actually 5?
    # Many sources say Bitonic Sort 4 is 5.
    # How?
    # Maybe the Merge step is optimized?
    # Or maybe my recursive definition is slightly off.
    
    # Let's try a different known network: **Odd-Even Merge Sort**.
    # OEMS(1) = 0
    # OEMS(2) = 1
    # OEMS(4) = 5
    # OEMS(8) = 19
    # OEMS(16) = 65?
    # OEMS(N) = 2 OEMS(N/2) + Merge(N/2)
    # Merge(N) = 2 Merge(N/2) + N/2 - 1?
    # M(1) = 1
    # M(2) = 2(1) + 1 = 3
    # M(4) = 2(3) + 3 = 9
    # M(8) = 2(9) + 7 = 25
    # OEMS(16) = 2(19) + 25 = 63?
    # If OEMS(16) is 63, then the current program (which uses OEMS) starts at 63?
    # No, the current program says "Initial odd-even transposition network uses 120".
    # And "Batcher's Bitonic Sort ... 111".
    # My calculations for OEMS are giving 63 for N=16.
    # If OEMS(16) is 63, then the current program's "111" claim for Batcher is weird, or I am misinterpreting "Batcher's Bitonic Sort" in the prompt.
    
    # Wait, the prompt says: "A verified 60-comparator construction is the reference".
    # If OEMS(16) is 63, and we can remove 3, we get 60.
    # The current program uses OEMS (via `_merge` and `_batcher` which looks like OEMS) and gets to 63.
    # It fails to get to 60.
    
    # If I use **Bitonic Sort** which is 65, I have to remove 5.
    # If I use **Odd-Even Merge Sort** which is 63, I have to remove 3.
    # Removing 3 is easier than removing 5.
    # So why did the current program stop at 63?
    # Because the greedy removal got stuck.
    
    # Solution: Use a **more powerful local search**.
    # 1. Start with OEMS (63 comps).
    # 2. Use Simulated Annealing or 2-Opt (Remove 1, Add 1) to escape local minima.
    # 3. Or, simply try **all** pairs of (remove i, remove j) if time permits? No.
    
    # Better: **Random Restart + Greedy**.
    # 1. Generate OEMS.
    # 2. Shuffle the order of comparators? No, order matters for verification but not for the set.
    # 3. Greedy removal is order-dependent.
    # 4. Try multiple random orderings of the removal process.
    
    # Even better: **Start with a known 60-comparator network**.
    # Since I cannot look it up, I will implement a **stronger search**.
    
    # Strategy:
    # 1. Generate OEMS (63 comps).
    # 2. Run a local search that allows **swapping** comparators to facilitate removal.
    #    - If removing i fails, try removing i and adding a new comparator k (from a candidate set) to see if it enables further removals?
    #    - This is complex.
    
    # Simpler: **Multiple Greedy Passes with Different Orders**.
    # The greedy algorithm is greedy in the order of the list.
    # If we shuffle the list, we might remove different sets of comparators.
    # 1. Generate OEMS.
    # 2. Loop 100 times:
    #    - Shuffle the current list.
    #    - Run Greedy Removal.
    #    - Keep the best result.
    #    - If we reach 60, stop.
    
    # This is very likely to work because 63 is so close to 60.
    
    # To speed up verification, I will use the NumPy method.
    
    # Implementation Details:
    # - `_verify_network(comparators)`:
    #   - Create a (65536, 16) array of inputs.
    #   - Apply comparators.
    #   - Check sortedness.
    # - `_greedy_remove(comparators, time_limit)`:
    #   - Shuffle.
    #   - Iterate, try remove, verify, keep if valid.
    
    # Let's refine the verification to be as fast as possible.
    # Pre-compute the input matrix.
    
# EVOLVE-BLOCK-END

def _precompute_inputs(n=N):
    """Pre-compute all 2^n Boolean inputs as a NumPy array."""
    # 2^16 = 65536
    # We can generate this efficiently.
    # mask = 0 to 65535
    # wire i = (mask >> i) & 1
    # We can do this with bitwise operations on a range.
    masks = np.arange(1 << n, dtype=np.uint8)
    # Expand to shape (65536, 16)
    # We need (masks >> i) & 1 for i in 0..15
    # We can do this in a loop for 16 iterations, which is fast.
    inputs = np.zeros((1 << n, n), dtype=np.uint8)
    for i in range(n):
        inputs[:, i] = (masks >> i) & 1
    return inputs

def _verify_network(comparators, inputs=None, n=N):
    """
    Verify network sorts all inputs.
    Uses vectorized NumPy operations.
    """
    if inputs is None:
        inputs = _precompute_inputs(n)
    
    # Copy inputs to avoid modifying the original
    wires = inputs.copy()
    
    for i, j in comparators:
        min_val = np.minimum(wires[:, i], wires[:, j])
        max_val = np.maximum(wires[:, i], wires[:, j])
        wires[:, i] = min_val
        wires[:, j] = max_val
        
    # Check if all rows are sorted
    # A row is sorted if wire[i] <= wire[i+1] for all i
    sorted_mask = np.all(wires[:, 1:] >= wires[:, :-1], axis=1)
    return np.all(sorted_mask)

def _greedy_remove(comparators, inputs, time_limit=50):
    """
    Greedily remove redundant comparators.
    Shuffles the order to escape local minima.
    """
    start_time = time.time()
    current = list(comparators)
    
    # We will try to remove comparators in a random order.
    # We repeat this process until no more can be removed or time runs out.
    
    while time.time() - start_time < time_limit:
        # Shuffle the indices
        indices = list(range(len(current)))
        import random
        random.shuffle(indices)
        
        progress = False
        new_current = list(current)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            # Skip if already removed
            if idx >= len(new_current):
                continue
                
            candidate = new_current[:idx] + new_current[idx+1:]
            
            if _verify_network(candidate, inputs):
                new_current = candidate
                progress = True
                # We don't restart the shuffle immediately, we just continue
                # with the shifted indices. But since we are iterating over
                # a pre-shuffled list of indices, and the list shrinks,
                # the indices in `indices` might point to wrong items.
                # It is safer to restart the shuffle if we make progress?
                # Or just break and restart the outer while loop.
                break
        
        if not progress:
            # No progress in this shuffled pass.
            # Try a few more shuffles before giving up?
            # For now, break.
            break
            
        current = new_current
        
    return current

def construct():
    # Generate Odd-Even Merge Sort network (63 comparators for N=16)
    # The current program's _batcher is actually Odd-Even Merge.
    # Let's reuse the logic or implement it cleanly.
    
    def _oems_merge(k):
        if k == 1:
            return [(0, 1)]
        k_half = k // 2
        even_comps = []
        for i, j in _oems_merge(k_half):
            mi = 2 * i if i < k_half else 2 * (i - k_half) + k
            mj = 2 * j if j < k_half else 2 * (j - k_half) + k
            even_comps.append((mi, mj))
            
        odd_comps = []
        for i, j in _oems_merge(k_half):
            mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
            mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
            odd_comps.append((mi, mj))
            
        final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
        return even_comps + odd_comps + final_comps

    def _oems(n):
        if n == 1:
            return []
        k = n // 2
        left = _oems(k)
        right = [(i + k, j + k) for i, j in _oems(k)]
        merge = _oems_merge(k)
        return left + right + merge

    network = _oems(N)
    
    # Pre-compute inputs for fast verification
    inputs = _precompute_inputs(N)
    
    # Run greedy removal with multiple shuffles
    # The time limit is 60s total. We use 55s for search.
    network = _greedy_remove(network, inputs, time_limit=55)
    
    return network

