import time

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Batcher Odd-Merge Sort Network Generator
    def batcher_oom(n):
        """
        Generates a list of comparators for a Batcher Odd-Merge Sort network of size n.
        Returns a list of tuples (i, j) where i < j.
        """
        if n == 1:
            return []
        
        # Step 1: Recursively sort evens and odds
        # Evens are at indices 0, 2, ..., n-2
        # Odds are at indices 1, 3, ..., n-1
        # We need to map the local indices of the sub-networks to the global indices.
        
        # Sub-network for evens: size n/2
        # Local index k in sub-network maps to global index 2*k
        # Sub-network for odds: size n/2
        # Local index k in sub-network maps to global index 2*k + 1
        
        comps = []
        
        # Generate evens sub-network
        sub_evens = batcher_oom(n // 2)
        for (i, j) in sub_evens:
            comps.append((2 * i, 2 * j))
            
        # Generate odds sub-network
        sub_odds = batcher_oom(n // 2)
        for (i, j) in sub_odds:
            comps.append((2 * i + 1, 2 * j + 1))
            
        # Step 2: Merge the two sorted sequences
        # This is the odd-even merge step
        comps.extend(batcher_merge(n // 2))
        
        return comps

    def batcher_merge(k):
        """
        Generates comparators to merge two sorted sequences of length k.
        The first sequence is on wires 0, 2, ..., 2k-2 (evens)
        The second sequence is on wires 1, 3, ..., 2k-1 (odds)
        Wait, the standard odd-even merge assumes the inputs are interleaved?
        No, the odd-even merge procedure takes two sorted lists A and B of length k.
        It merges them into a sorted list of length 2k.
        
        In the context of Batcher's OOM:
        After sorting evens and odds, we have:
        Wire 0: min of evens
        Wire 1: min of odds
        ...
        Wire 2k-2: max of evens
        Wire 2k-1: max of odds
        
        The merge step works on these 2k wires.
        It recursively merges:
        - The even-indexed wires of the combined set? No.
        - The "even" elements of the first list (A) and "even" elements of the second list (B).
        
        Let's stick to the standard recursive definition of Odd-Even Merge:
        Merge(A, B) where A and B are sorted lists of length k.
        1. Split A into A_even and A_odd.
        2. Split B into B_even and B_odd.
        3. Recursively merge A_even and B_even.
        4. Recursively merge A_odd and B_odd.
        5. Compare-exchange adjacent pairs in the resulting 2k elements.
        
        In our wire mapping:
        A is on wires 0, 2, ..., 2k-2.
        B is on wires 1, 3, ..., 2k-1.
        
        A_even: wires 0, 4, ..., 2k-4 (indices 2*i in A, which are 4*i in global? No.)
        Let's trace indices carefully.
        
        Global wires: 0, 1, 2, 3, ..., 2k-1.
        A (sorted evens of original N): wires 0, 2, 4, ..., 2k-2.
        B (sorted odds of original N): wires 1, 3, 5, ..., 2k-1.
        
        We want to merge A and B.
        A_even consists of A[0], A[2], ... which are global wires 0, 4, ...
        A_odd consists of A[1], A[3], ... which are global wires 2, 6, ...
        B_even consists of B[0], B[2], ... which are global wires 1, 5, ...
        B_odd consists of B[1], B[3], ... which are global wires 3, 7, ...
        
        So we recursively merge:
        - A_even (wires 0, 4, ...) with B_even (wires 1, 5, ...).
          This is a merge of two lists of length k/2.
          The wires involved are 0, 1, 4, 5, ...
          We can map these to a local 0..k-1 space?
          Local index 0 -> Global 0
          Local index 1 -> Global 1
          Local index 2 -> Global 4
          Local index 3 -> Global 5
          ...
          Local index 2*i -> Global 4*i
          Local index 2*i+1 -> Global 4*i+1
          
        - A_odd (wires 2, 6, ...) with B_odd (wires 3, 7, ...).
          Local index 2*i -> Global 4*i+2
          Local index 2*i+1 -> Global 4*i+3
          
        Finally, compare-exchange (0,1), (2,3), ..., (2k-2, 2k-1).
        
        This recursive structure is standard.
        """
        if k == 1:
            # Merging two single elements: just compare-exchange them.
            # A is on wire 0, B is on wire 1.
            return [(0, 1)]
            
        half_k = k // 2
        comps = []
        
        # Recursive merge of A_even and B_even
        # These are on global wires 0, 1, 4, 5, ...
        # We generate a merge network of size half_k, but mapped to specific wires.
        # The local indices 0..2*half_k-1 map to global wires:
        # Local 2*i -> Global 4*i
        # Local 2*i+1 -> Global 4*i+1
        
        sub_merge_even = batcher_merge(half_k)
        for (i, j) in sub_merge_even:
            # Map local i, j to global
            gi = 4 * (i // 2) + (i % 2)
            gj = 4 * (j // 2) + (j % 2)
            comps.append((gi, gj))
            
        # Recursive merge of A_odd and B_odd
        # These are on global wires 2, 3, 6, 7, ...
        # Local 2*i -> Global 4*i+2
        # Local 2*i+1 -> Global 4*i+3
        
        sub_merge_odd = batcher_merge(half_k)
        for (i, j) in sub_merge_odd:
            gi = 4 * (i // 2) + 2 + (i % 2)
            gj = 4 * (j // 2) + 2 + (j % 2)
            comps.append((gi, gj))
            
        # Final compare-exchange layer
        for i in range(k):
            comps.append((2 * i, 2 * i + 1))
            
        return comps

    network = batcher_oom(N)
    
    # Optional: Verify if time permits
    if time.time() < deadline:
        # Quick verification on a few random inputs
        import random
        for _ in range(100):
            input_seq = list(range(N))
            random.shuffle(input_seq)
            current = input_seq[:]
            for (i, j) in network:
                if current[i] > current[j]:
                    current[i], current[j] = current[j], current[i]
            if current != sorted(input_seq):
                # If invalid, fallback to odd-even transposition (though it shouldn't happen)
                return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
                
    return network

# EVOLVE-BLOCK-START
def construct():
    # Use the Batcher Odd-Merge network generated above
    return _construct_batcher()
# EVOLVE-BLOCK-END

def _construct_batcher():
    # Re-defining to keep the module clean and ensure the function is available
    # Actually, we can just define it inside construct or use a helper.
    # Let's put the logic in a helper to keep construct clean.
    
    def batcher_oom(n):
        if n == 1:
            return []
        
        comps = []
        
        sub_evens = batcher_oom(n // 2)
        for (i, j) in sub_evens:
            comps.append((2 * i, 2 * j))
            
        sub_odds = batcher_oom(n // 2)
        for (i, j) in sub_odds:
            comps.append((2 * i + 1, 2 * j + 1))
            
        comps.extend(batcher_merge(n // 2))
        
        return comps

    def batcher_merge(k):
        if k == 1:
            return [(0, 1)]
            
        half_k = k // 2
        comps = []
        
        sub_merge_even = batcher_merge(half_k)
        for (i, j) in sub_merge_even:
            gi = 4 * (i // 2) + (i % 2)
            gj = 4 * (j // 2) + (j % 2)
            comps.append((gi, gj))
            
        sub_merge_odd = batcher_merge(half_k)
        for (i, j) in sub_merge_odd:
            gi = 4 * (i // 2) + 2 + (i % 2)
            gj = 4 * (j // 2) + 2 + (j % 2)
            comps.append((gi, gj))
            
        for i in range(k):
            comps.append((2 * i, 2 * i + 1))
            
        return comps

    return batcher_oom(16)

