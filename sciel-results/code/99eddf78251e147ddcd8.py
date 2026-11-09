import time

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 80 # Safety margin
    
    # 1. Generate Batcher's Odd-Even Merge Sort Network
    network = _batcher_odd_even_merge_sort(N)
    
    # 2. Reduce redundant comparators
    # We will try to remove comparators one by one.
    # To be efficient, we check validity using bit-parallel simulation.
    
    current_network = network
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in range(len(current_network)):
            if time.time() > deadline:
                break
            # Try removing current_network[i]
            candidate = current_network[:i] + current_network[i+1:]
            if _is_valid(candidate):
                current_network = candidate
                improved = True
                break # Restart loop as indices changed
    
    return current_network

def _batcher_odd_even_merge_sort(n):
    # Returns list of (i, j)
    if n <= 1:
        return []
    
    # Recursive step
    # Split into two halves
    half = n // 2
    left = _batcher_odd_even_merge_sort(half)
    right = _batcher_odd_even_merge_sort(half)
    
    # Map right half indices to global indices
    # Left wires: 0..half-1
    # Right wires: half..n-1
    # In the merge step, we need to interleave?
    # No, Batcher's OE Merge Sort works on contiguous blocks?
    # Actually, the standard implementation usually maps the right half's local indices to global by adding half.
    
    # But the Merge step is tricky.
    # Let's use the known explicit construction for N=16 or a robust recursive one.
    
    # A safer recursive definition for Batcher's OE Merge Sort:
    # Sort(2k) = Sort(k) on (0..k-1), Sort(k) on (k..2k-1), then Merge(k)
    # Merge(k) merges two sorted lists of size k located at (0..k-1) and (k..2k-1)
    # Merge(k) definition:
    #   If k==1: compare (0,1)
    #   Else:
    #     Odd indices of left: 0, 2, 4...
    #     Odd indices of right: k, k+2, k+4...
    #     Even indices of left: 1, 3, 5...
    #     Even indices of right: k+1, k+3, k+5...
    #     
    #     Merge(Odd Left, Odd Right) -> Recursively merge these?
    #     No, the merge step of Batcher's algorithm is:
    #     1. Merge the odd-indexed elements of the two lists.
    #     2. Merge the even-indexed elements of the two lists.
    #     3. Compare-exchange corresponding elements.
    
    # Let's implement this specific recursion.
    
    def merge(left_start, right_start, size):
        # Merges two sorted lists of 'size' elements starting at left_start and right_start
        # The lists are interleaved in the final output?
        # In Batcher's OE Merge, the output is interleaved?
        # No, the output of Merge(k) is a sorted list of 2k elements.
        # But the elements are not just concatenated.
        # The standard Batcher OE Merge produces a sorted list, but the "wires" are mapped such that:
        # The odd elements of the first list and odd elements of the second list are merged into the odd positions of the output?
        # Actually, it's easier to just use the known list of comparators for N=16 if recursion is tricky to get right with indices.
        
        # Let's try to get the recursion right.
        # Source: "Batcher's Sorting Network"
        # OE_Merge(A, B) where A and B are sorted lists of length N.
        # Output is a sorted list of length 2N.
        # The network consists of:
        # 1. OE_Merge(A_odd, B_odd)
        # 2. OE_Merge(A_even, B_even)
        # 3. Compare-exchange (A[i], B[i]) for i=0..N-1
        
        # Indices:
        # A is at indices 0..N-1
        # B is at indices N..2N-1
        # A_odd indices: 0, 2, ...
        # B_odd indices: N, N+2, ...
        # A_even indices: 1, 3, ...
        # B_even indices: N+1, N+3, ...
        
        # The recursive calls operate on subsets of indices.
        # This requires careful index mapping.
        
        pass

    # Given the complexity and risk of index bugs in recursive Batcher, 
    # and the fact that N=16 is small, I will hardcode the Batcher network 
    # or use a simpler known optimal-ish network like AKS? No, AKS is deep.
    # I will use the explicit list for Batcher's OE Merge Sort for N=16.
    # I can generate it using a trusted Python snippet in my head or derive it.
    
    # Actually, I can just use the `networkx` or similar? No, standard lib only.
    
    # Let's write a correct recursive generator for Batcher's OE Merge Sort.
    
    def generate(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        # Sort left half (0 to half-1)
        left_net = generate(half)
        # Sort right half (half to n-1)
        right_net_raw = generate(half)
        right_net = [(i + half, j + half) for (i, j) in right_net_raw]
        
        # Merge
        # We need to merge the two sorted halves.
        # The merge step of Batcher's OE Merge Sort is defined on the indices.
        # Let's define a helper that returns comparators for merging two lists
        # located at specific indices.
        
        def merge_step(l_start, r_start, size):
            # Merges lists at l_start..l_start+size-1 and r_start..r_start+size-1
            # size is power of 2.
            if size == 1:
                return [(l_start, r_start)]
            
            half_size = size // 2
            # Odd indices in left: l_start, l_start+2, ...
            # Odd indices in right: r_start, r_start+2, ...
            # Even indices in left: l_start+1, l_start+3, ...
            # Even indices in right: r_start+1, r_start+3, ...
            
            # Recursive merge on odds
            # The "odds" form a list of half_size elements.
            # But they are not contiguous in memory.
            # Batcher's network assumes contiguous blocks?
            # Yes, the standard construction maps the logical "odd" and "even" elements to contiguous blocks in the sub-networks?
            # No, the comparators are between specific wires.
            
            # The recursive call `OE_Merge` operates on the values.
            # The indices for the recursive merge on "odds" are:
            # Left odds: l_start, l_start+2, ...
            # Right odds: r_start, r_start+2, ...
            # This is not a contiguous block merge in the sense of `merge_step(start, start+size, size)`.
            # It's a merge of interleaved indices.
            
            # To make this work with a simple recursive function that assumes contiguous blocks,
            # we need to permute the indices?
            # No, Batcher's network is defined on the wires directly.
            
            # Let's look at the structure for N=4.
            # Sort(4):
            # Sort(2) on (0,1) -> (0,1)
            # Sort(2) on (2,3) -> (2,3)
            # Merge(2) on (0,1) and (2,3).
            # Merge(2) steps:
            # 1. Merge Odds of (0,1) and (2,3).
            #    Odds of (0,1) are index 0.
            #    Odds of (2,3) are index 2.
            #    Merge size 1: Compare (0,2).
            # 2. Merge Evens of (0,1) and (2,3).
            #    Evens of (0,1) are index 1.
            #    Evens of (2,3) are index 3.
            #    Merge size 1: Compare (1,3).
            # 3. Compare corresponding elements:
            #    (0,2) and (1,3).
            #    Wait, step 1 and 3 are the same?
            #    For size 2, the merge is:
            #    Compare (0, 2)
            #    Compare (1, 3)
            #    Compare (0, 2) ? No.
            
            # Let's check N=4 Batcher OE Merge Sort comparators:
            # Standard list:
            # (0,1), (2,3)
            # (0,2), (1,3)
            # (0,1), (2,3)
            # Total 6 comparators?
            # My previous count was 4.
            # Let's re-verify C(4).
            # C(1)=0
            # C(2)=1
            # C(4) = 2*C(2) + M(2).
            # M(2) for Batcher OE Merge:
            # Merge two sorted lists of size 2.
            # 1. Merge odds (size 1 list each): 1 comparator.
            # 2. Merge evens (size 1 list each): 1 comparator.
            # 3. Compare-exchange corresponding: 2 comparators.
            # Total M(2) = 4.
            # C(4) = 2*1 + 4 = 6.
            
            # C(8) = 2*C(4) + M(4).
            # M(4) = M(2) [odds] + M(2) [evens] + 4 [compare-exchange]
            # M(2) = 4.
            # M(4) = 4 + 4 + 4 = 12.
            # C(8) = 2*6 + 12 = 24.
            
            # C(16) = 2*C(8) + M(8).
            # M(8) = M(4) + M(4) + 8 = 12 + 12 + 8 = 32.
            # C(16) = 2*24 + 32 = 48 + 32 = 80.
            
            # Wait, earlier I thought 72.
            # Let's check online sources for Batcher's OE Merge Sort N=16 comparator count.
            # Some sources say 72.
            # Why the discrepancy?
            # Maybe M(k) is different.
            # In some variations, the final compare-exchange step is optimized?
            # Or maybe my M(2) is wrong.
            # Merge(2):
            # Lists: [a0, a1] and [b0, b1].
            # Odds: a0, b0. Merge: (a0, b0).
            # Evens: a1, b1. Merge: (a1, b1).
            # Compare-exchange: (a0, b0) and (a1, b1).
            # So (a0,b0) is done twice?
            # If the lists are sorted, a0<=a1 and b0<=b1.
            # After (a0,b0), we have min(a0,b0) at a0, max at b0.
            # After (a1,b1), we have min(a1,b1) at a1, max at b1.
            # Then compare-exchange (a0,b0) again?
            # If a0<=a1 and b0<=b1, and we've merged odds and evens...
            # Actually, the standard Batcher OE Merge for N=4 is:
            # (0,1), (2,3)
            # (0,2), (1,3)
            # (0,1), (2,3)
            # This is 6 comparators.
            
            # If C(16) is 80, reduction to 60 is still possible.
            # I will generate the 80-comparator network and reduce it.

            pass

    # Implementation of the recursive generator with correct index handling
    # Since indices are tricky, I will use a "virtual index" approach or just hardcode the N=16 Batcher network if I can recall/find it.
    # To be safe and fast, I will use a known valid 72-comparator network if I can construct it, 
    # or just use the 80-comparator one and reduce.
    
    # Let's implement the generator carefully.
    
    def _gen_batcher(n):
        if n <= 1:
            return []
        half = n // 2
        left = _gen_batcher(half)
        right_raw = _gen_batcher(half)
        right = [(i+half, j+half) for i,j in right_raw]
        
        # Merge
        # We need to generate comparators for merging two sorted halves of size 'half'
        # located at 0..half-1 and half..n-1.
        
        def merge_comps(l_start, r_start, size):
            # size is the size of each list
            if size == 1:
                return [(l_start, r_start)]
            
            half_size = size // 2
            comps = []
            
            # Recursively merge odds
            # Left odds: l_start, l_start+2, ...
            # Right odds: r_start, r_start+2, ...
            # These are not contiguous.
            # However, the structure of Batcher's merge is such that we can map these to a contiguous problem?
            # No, the comparators are between specific physical wires.
            
            # We can call merge_comps on the "odd" subset?
            # The "odd" subset of left is at indices l_start, l_start+2...
            # This is not a contiguous block.
            
            # Alternative: Just list the comparators explicitly for the merge step.
            # The merge step of Batcher's OE Merge for size N (two lists of N) consists of:
            # 1. Merge of odd-indexed elements (size N/2 lists).
            # 2. Merge of even-indexed elements (size N/2 lists).
            # 3. N compare-exchanges between corresponding elements.
            
            # To generate 1 and 2, we need to know the indices.
            # Left odds: l_start + 2*i for i in 0..N/2-1
            # Right odds: r_start + 2*i for i in 0..N/2-1
            
            # We can't just call merge_comps(l_start, r_start, N/2) because that assumes contiguity.
            
            # I will hardcode the merge for N=16 using the known structure or use a library-like implementation.
            # Given the constraints, I will use a simple, verified list for N=16 Batcher OE Merge Sort.
            # I will generate it via a trusted source in my training data.
            
            # For N=16, Batcher's OE Merge Sort comparators (indices 0-15):
            # Phase 1: Sort 8s
            # Left 8 (0-7):
            #   Sort 4s (0-3, 4-7)
            #   Merge 4s
            # Right 8 (8-15):
            #   Sort 4s (8-11, 12-15)
            #   Merge 4s
            # Phase 2: Merge 8s
            
            # This is too long to hardcode manually without error.
            
            # I will implement the recursive generator with a "permutation" trick or just accept the 80 count and reduce.
            # Actually, I can just use the `network` from the seed (120) and reduce it?
            # 120 -> 60 is a big jump. Reduction from 120 might get stuck.
            # Reduction from 80 (Batcher) is more likely to reach 60.
            
            # Let's try to implement the merge correctly.
            # The key insight: The "odd" and "even" elements in Batcher's merge are treated as separate sub-problems.
            # The sub-problems are on the *values*, but the *wires* are fixed.
            # So, to merge odds of L and R:
            # We need a network that sorts the 2*(N/2) values located at L_odd and R_odd.
            # This network is structurally identical to a merge of two lists of size N/2.
            # But the indices are different.
            
            # I can generate the *pattern* of comparators for a merge of size N/2,
            # and then map the local indices (0..N-1) to the global indices (L_odd, R_odd).
            
            def map_indices(local_comps, l_indices, r_indices):
                # local_comps is a list of (i, j) for a merge of two lists of size len(l_indices)
                # l_indices is list of global indices for the left list
                # r_indices is list of global indices for the right list
                # Local list 0 is l_indices, Local list 1 is r_indices.
                # Local index k: if k < len(l_indices), it's l_indices[k], else r_indices[k-len(l_indices)]
                mapped = []
                size = len(l_indices)
                for i, j in local_comps:
                    gi = l_indices[i] if i < size else r_indices[i-size]
                    gj = l_indices[j] if j < size else r_indices[j-size]
                    mapped.append((min(gi, gj), max(gi, gj)))
                return mapped

            # Now we can recurse.
            # Merge(size N) on L (start l_start) and R (start r_start):
            # N is size of each list.
            if size == 1:
                return [(l_start, r_start)]
            
            half_size = size // 2
            l_odds = [l_start + 2*i for i in range(half_size)]
            r_odds = [r_start + 2*i for i in range(half_size)]
            l_evens = [l_start + 2*i + 1 for i in range(half_size)]
            r_evens = [r_start + 2*i + 1 for i in range(half_size)]
            
            # Get pattern for merge of size half_size
            # We need a function that returns the *pattern* of comparators for merging two lists of size k.
            # Let's call it `merge_pattern(k)`.
            
            pass
            
        # To avoid circular dependency in `merge_comps`, I will define `merge_pattern(k)` separately.
        
        def merge_pattern(k):
            # Returns list of (i, j) for merging two sorted lists of size k,
            # where list 0 is indices 0..k-1 and list 1 is indices k..2k-1.
            if k == 1:
                return [(0, 1)]
            
            half_k = k // 2
            l_odds = list(range(0, k, 2))
            r_odds = list(range(k, 2*k, 2))
            l_evens = list(range(1, k, 2))
            r_evens = list(range(k+1, 2*k, 2))
            
            # Recursive pattern for half_k
            sub_pattern = merge_pattern(half_k)
            
            comps = []
            # Map sub_pattern to odds
            # In sub_pattern, list 0 is 0..half_k-1, list 1 is half_k..2*half_k-1
            # We need to map these local indices to l_odds and r_odds.
            # Local index x:
            # if x < half_k: it's in list 0 -> l_odds[x]
            # else: it's in list 1 -> r_odds[x - half_k]
            
            def map_sub(comps, l_idx, r_idx):
                mapped = []
                for i, j in comps:
                    gi = l_idx[i] if i < half_k else r_idx[i - half_k]
                    gj = l_idx[j] if j < half_k else r_idx[j - half_k]
                    mapped.append((min(gi, gj), max(gi, gj)))
                return mapped
            
            comps.extend(map_sub(sub_pattern, l_odds, r_odds))
            comps.extend(map_sub(sub_pattern, l_evens, r_evens))
            
            # Final compare-exchange
            for i in range(k):
                comps.append((i, i + k))
                
            return comps

        # Now we can use merge_pattern in the main generator.
        
        # But wait, the main generator `generate(n)` calls `merge_comps` which uses `merge_pattern`.
        # `generate(n)` produces the full network for N wires.
        # `merge_pattern(k)` produces the comparators for the merge step of two lists of size k.
        
        # Let's re-structure.
        
        def full_network(n):
            if n <= 1:
                return []
            half = n // 2
            left_net = full_network(half)
            right_net_raw = full_network(half)
            right_net = [(i+half, j+half) for i,j in right_net_raw]
            
            # Merge step
            # We need the comparators for merging two lists of size 'half'
            # located at 0..half-1 and half..n-1.
            # We use merge_pattern(half) to get the local indices.
            # Then map them to global.
            
            merge_pat = merge_pattern(half)
            l_indices = list(range(half))
            r_indices = list(range(half, n))
            
            # Map merge_pat (which assumes list 0 is 0..half-1, list 1 is half..2*half-1)
            # to l_indices and r_indices.
            # In merge_pat, local index x:
            # if x < half: l_indices[x]
            # else: r_indices[x - half]
            
            merged = []
            for i, j in merge_pat:
                gi = l_indices[i] if i < half else r_indices[i - half]
                gj = l_indices[j] if j < half else r_indices[j - half]
                merged.append((min(gi, gj), max(gi, gj)))
                
            return left_net + right_net + merged
        
        return full_network(n)

    net = _gen_batcher(N)
    
    # Reduce
    # ...

