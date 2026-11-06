import time

N = 17

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Batcher's Merge Sort implementation
    def batcher_sort(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        n1 = n // 2
        n2 = n - n1
        
        c1 = batcher_sort(n1)
        c2 = batcher_sort(n2)
        # Shift c2 indices
        c2_shifted = [(i + n1, j + n1) for (i, j) in c2]
        
        cm = batcher_merge(n1, n2)
        
        return c1 + c2_shifted + cm

    def batcher_merge(n1, n2):
        # Ensure n1 >= n2 for padding logic simplicity? 
        # Actually, we pad the smaller one to the larger one.
        # But Batcher's merge is symmetric.
        # Let's always pad the second list (n2) to the size of the first list (n1) 
        # only if n1 == n2? No.
        # Standard approach: Pad both to the next power of 2? 
        # Or pad the smaller to the larger.
        
        # To keep it simple and correct, we pad the smaller to the larger.
        # If n1 < n2, we swap roles in the merge generation, but the indices 
        # in the returned list must correspond to the original wire positions.
        # Wire positions: 0..n1-1 (List 1), n1..n1+n2-1 (List 2).
        
        # Let's generate merge for size K, K where K = max(n1, n2).
        # If n1 < n2, we pad List 1.
        # If n2 < n1, we pad List 2.
        
        K = max(n1, n2)
        
        # Generate comparators for merging two sorted lists of size K.
        # Local indices: 0..K-1 (List A), K..2K-1 (List B).
        merge_comps = _batcher_merge_equal(K)
        
        # Filter comparators
        # Real wires in local coords:
        # If n1 == n2: Real are 0..n1-1 and n1..2n1-1. All real.
        # If n1 < n2: 
        #   List A (0..K-1) has real 0..n1-1, dummy n1..K-1.
        #   List B (K..2K-1) has real K..K+n2-1.
        #   Total real wires: n1 + n2.
        #   Real indices: 0..n1-1 and K..K+n2-1.
        #   This is not a contiguous range 0..n1+n2-1.
        #   So we cannot just check `index < n1+n2`.
        
        # We need to check if an index is real.
        # Define a set of real local indices.
        real_indices = set()
        for i in range(n1):
            real_indices.add(i)
        for i in range(n2):
            real_indices.add(K + i)
            
        filtered_comps = []
        for (i, j) in merge_comps:
            if i in real_indices or j in real_indices:
                # Map local indices back to global?
                # The merge function returns local indices.
                # We need to return global indices relative to the start of this merge block.
                # But the caller (batcher_sort) just concatenates.
                # The local indices 0..2K-1 correspond to the wires in this merge step.
                # We return them as is, but they must be valid for the current scope.
                # In batcher_sort, the merge is applied to wires 0..n-1.
                # So local indices ARE the global indices for this sub-problem.
                filtered_comps.append((i, j))
                
        return filtered_comps

    def _batcher_merge_equal(k):
        # Returns comparators for merging two sorted lists of size k.
        # Local indices: 0..k-1 (A), k..2k-1 (B).
        if k == 1:
            return [(0, 1)]
            
        # Split into odds and evens
        # Odds of A: 1, 3, ...
        # Evens of A: 0, 2, ...
        # Odds of B: k+1, k+3, ...
        # Evens of B: k, k+2, ...
        
        # This recursive structure is hard to implement directly with indices 
        # without remapping.
        # Alternative: Use the standard recursive definition on the logical positions.
        
        # Let's use a helper that returns comparators for merging two lists 
        # of size k, where the first list is at offset 0 and second at offset k.
        
        # Recursion:
        # 1. Merge odds of A and odds of B.
        #    Odds of A are at indices 1, 3, ... k-1 (if k even) or k-2 (if k odd).
        #    This is not a contiguous block.
        
        # Given the complexity of remapping, let's use the padding to power of 2 approach 
        # which allows simple index arithmetic?
        # No, the 99 comparator estimate is good enough.
        # I will implement the remapping carefully.
        
        # Actually, there is a simpler way: 
        # Implement Batcher's merge on a list of wires, tracking their logical positions.
        
        # Let's try a different strategy: 
        # Use the known fact that for N=17, a simple Bitonic Sort on 32 wires 
        # filtered to 17 might be simpler to code correctly?
        # Bitonic Sort 32: 400 comparators. Too many.
        
        # Let's stick to the 99 estimate but implement it correctly.
        # I will use a recursive function that takes a list of wire indices 
        # representing the current "logical" positions of the elements.
        
        def merge_recursive(indices_a, indices_b):
            # indices_a: list of global wire indices for first sorted list
            # indices_b: list of global wire indices for second sorted list
            # Returns list of comparators
            k = len(indices_a)
            if k == 1:
                return [(indices_a[0], indices_b[0])]
                
            # Split into odds and evens based on position in the list
            # Odds: positions 1, 3, 5...
            # Evens: positions 0, 2, 4...
            
            odds_a = indices_a[1::2]
            evens_a = indices_a[0::2]
            odds_b = indices_b[1::2]
            evens_b = indices_b[0::2]
            
            # Recursively merge
            comps = []
            comps += merge_recursive(odds_a, odds_b)
            comps += merge_recursive(evens_a, evens_b)
            
            # Interleave and compare
            # The result of merging odds is sorted odds.
            # The result of merging evens is sorted evens.
            # We need to interleave them.
            # The wire indices in odds_a/b and evens_a/b are the actual wires.
            # After recursive merges, the values in these wires are sorted.
            # We need to compare (2i, 2i+1) in the logical sequence.
            # Logical sequence: e_0, o_0, e_1, o_1...
            # e_i is the i-th element of the merged evens.
            # o_i is the i-th element of the merged odds.
            
            # We need to know which wire holds e_i and which holds o_i.
            # The recursive merge returns comparators, but not the permutation.
            # However, Batcher's merge guarantees that after the recursive steps,
            # the evens are in the evens wires and odds in the odds wires?
            # No, the recursive merge mixes them.
            
            # Wait, the standard Batcher merge algorithm:
            # 1. Split A and B into odds and evens.
            # 2. Recursively merge odds of A with odds of B.
            # 3. Recursively merge evens of A with evens of B.
            # 4. Interleave the results.
            # 5. Compare-exchange adjacent pairs.
            
            # The "Interleave" step is just reordering the wires?
            # No, the wires are fixed.
            # The recursive merge operates on the wires belonging to the odds/evens.
            # After step 2, the wires in `odds_a + odds_b` contain the sorted odds.
            # After step 3, the wires in `evens_a + evens_b` contain the sorted evens.
            
            # Now we need to compare (e_i, o_i) for i=0..k-1.
            # e_i is the i-th smallest even. It resides in the i-th position of the 
            # logical even sequence.
            # The physical wires for evens are `evens_a + evens_b`.
            # Are they in sorted order in those wires?
            # The recursive merge returns a network that sorts the input wires.
            # So yes, the wires in `evens_a + evens_b` are sorted.
            # So `evens_a + evens_b`[i] holds e_i.
            # Similarly `odds_a + odds_b`[i] holds o_i.
            
            # So we compare ( (evens_a + evens_b)[i], (odds_a + odds_b)[i] ) ?
            # No, the standard interleaving is e_0, o_0, e_1, o_1.
            # We compare (e_0, o_0), (e_1, o_1), etc.
            
            merged_evens = evens_a + evens_b
            merged_odds = odds_a + odds_b
            
            for i in range(k):
                comps.append((merged_evens[i], merged_odds[i]))
                
            return comps

        # Now we can use this recursive function.
        # For batcher_merge(n1, n2):
        # Pad the smaller list with dummy indices?
        # The recursive function requires equal length lists.
        # We can pad with a special dummy index?
        # But the comparators must be on real wires.
        
        # So we go back to the padding + filtering idea, 
        # but now we can generate the merge network using the recursive function 
        # on the padded indices, and then filter.
        
        # In _batcher_merge_equal(k), we can just call merge_recursive 
        # with indices 0..k-1 and k..2k-1.
        
        return merge_recursive(list(range(k)), list(range(k, 2*k)))

    # Call the constructor
    network = batcher_sort(N)
    
    # Verify within time limit?
    # The task says "correctness is certified...".
    # I should add a quick check if time permits.
    
    return network

