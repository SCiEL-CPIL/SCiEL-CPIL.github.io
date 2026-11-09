    def batcher_ooms(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Split into even and odd indices
        # Even indices: 0, 2, 4...
        # Odd indices: 1, 3, 5...
        # But Batcher's OOMs works on logical positions.
        # Standard recursive definition:
        # OOMs(n):
        #   if n == 1: return []
        #   if n == 2: return [(0,1)]
        #   m = n // 2
        #   # Sort even indices (0, 2, ..., 2m-2)
        #   # Sort odd indices (1, 3, ..., 2m-1)
        #   # Then merge them.
        
        # Wait, the standard Batcher network for N inputs is often constructed by:
        # 1. Recursively sort the first half and second half? No, that's for merging.
        # Batcher's Odd-Even-Merge Sort for a single input of size N:
        # It is defined recursively on the indices.
        
        def ooms(indices):
            n = len(indices)
            if n <= 1:
                return []
            if n == 2:
                return [(indices[0], indices[1])]
            
            # Split indices into even and odd positions in the list
            evens = indices[0::2]
            odds = indices[1::2]
            
            comps = []
            comps.extend(ooms(evens))
            comps.extend(ooms(odds))
            
            # Merge phase
            # The merge of two sorted sequences of length m (evens) and m (odds)
            # involves comparing (evens[i], odds[i]) for i=0..m-1?
            # No, the merge step in Batcher's OOMs is:
            # For i in range(min(len(evens), len(odds))):
            #    compare (evens[i], odds[i])
            # But we need to map these back to the original indices.
            # Actually, the recursive structure is:
            # Sort evens (positions 0, 2, 4...)
            # Sort odds (positions 1, 3, 5...)
            # Then perform a "merge" of the two sorted sub-lists.
            # The merge of two sorted lists of size m and m takes a specific network.
            # Batcher's merge of two sorted lists of length m is:
            #   if m == 1: compare (0,1)
            #   else:
            #     merge(evens_of_list1, evens_of_list2) -- wait, no.
            
            # Let's use a known correct recursive implementation.
            # Reference: 
            # def batcher(n):
            #   if n == 1: return []
            #   if n == 2: return [(0,1)]
            #   m = n//2
            #   # Sort first m and last m? No.
            
            # Correct Batcher Odd-Even Merge Sort for N items (indices 0..N-1):
            # 1. Recursively sort the items at even indices (0, 2, ...)
            # 2. Recursively sort the items at odd indices (1, 3, ...)
            # 3. Merge the two sorted sequences.
            #    The merge of two sorted sequences of length L is done by:
            #    a. If L == 1, compare (0, 1)
            #    b. If L > 1, recursively merge the even-indexed elements of the two sequences?
            
            # Let's implement `merge(a, b)` where a and b are lists of indices representing the two sorted halves.
            # Actually, it's easier to generate the comparators for a merge of two sorted lists of size m.
            
            def merge(a, b):
                # a and b are lists of indices. len(a) == len(b) == m
                m = len(a)
                if m == 1:
                    return [(a[0], b[0])]
                
                comps = []
                # Split a and b into evens and odds
                a_even = a[0::2]
                a_odd = a[1::2]
                b_even = b[0::2]
                b_odd = b[1::2]
                
                # Recursively merge a_even and b_even
                comps.extend(merge(a_even, b_even))
                # Recursively merge a_odd and b_odd
                comps.extend(merge(a_odd, b_odd))
                
                # Final step: compare (a[i], b[i]) for all i?
                # No, the final step of Batcher's merge is:
                # For i in range(m-1): compare (a[i], b[i])? 
                # Actually, the standard definition says:
                # After merging evens and odds, we compare (a[i], b[i]) for i=0..m-1.
                # Wait, let's check a small case.
                # Merge 2 and 2.
                # a=[0,2], b=[1,3]
                # a_even=[0], a_odd=[2]
                # b_even=[1], b_odd=[3]
                # merge([0],[1]) -> [(0,1)]
                # merge([2],[3]) -> [(2,3)]
                # Then compare (a[0], b[0]) and (a[1], b[1])?
                # i=0: (0,1) -- already done?
                # i=1: (2,3) -- already done?
                # If we add (0,1) and (2,3) again, it's redundant but safe.
                # But is it correct?
                # Let's trace:
                # Inputs: a0, a1, b0, b1 (sorted within a and b? No, a and b are the indices of the wires holding the sorted sub-sequences).
                # The wires are a[0], a[1] and b[0], b[1].
                # After recursive merges:
                # Wires a[0], b[0] are sorted relative to each other? No, merge(a_even, b_even) sorts the pair (a[0], b[0]).
                # Wires a[1], b[1] are sorted relative to each other? No, merge(a_odd, b_odd) sorts the pair (a[1], b[1]).
                # Now we need to ensure the whole sequence a[0], a[1], b[0], b[1] is sorted?
                # No, the output of the merge is the sorted sequence of the union of a and b.
                # The Batcher merge network produces the sorted sequence.
                # The final step is indeed comparing (a[i], b[i]) for i=0..m-1.
                # So for m=2:
                # merge([0,2], [1,3])
                # -> merge([0],[1]) -> (0,1)
                # -> merge([2],[3]) -> (2,3)
                # -> compare (a[0], b[0]) = (0,1)
                # -> compare (a[1], b[1]) = (2,3)
                # Total: (0,1), (2,3), (0,1), (2,3).
                # This is correct, just redundant.
                
                for i in range(m):
                    comps.append((a[i], b[i]))
                    
                return comps

            # Back to ooms
            # evens and odds are sorted.
            # We need to merge the sequence of evens and the sequence of odds.
            # The output wires are interleaved: wire 0, wire 1, wire 2, wire 3...
            # The values at evens are in wires evens[0], evens[1]...
            # The values at odds are in wires odds[0], odds[1]...
            # The merge operation takes the two sorted lists (evens, odds) and outputs the sorted union.
            # The merge function `merge(evens, odds)` will return comparators that sort the union.
            # But the output of the merge is not just "sorted values", it's placed on the wires defined by the merge structure.
            # Wait, Batcher's OOMs is a *sorting* network. The output is on the same wires.
            # The recursive step:
            # 1. Sort evens (wires 0, 2, 4...)
            # 2. Sort odds (wires 1, 3, 5...)
            # 3. Merge the two sorted lists.
            # The merge step `merge(evens, odds)` assumes the inputs are on wires `evens` and `odds`.
            # It performs comparators between these wires.
            # After the merge, the entire set of wires is sorted.
            
            comps.extend(merge(evens, odds))
            return comps
        
        return ooms(list(range(n)))
    