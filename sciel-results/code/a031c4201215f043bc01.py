"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a Batcher Odd-Merge sorting network for N=16 wires.
    Returns a list of (i, j) pairs representing compare-exchange operations.
    """
    def generate_odd_merge(n):
        if n <= 1:
            return []
        
        # Split into odd and even indices
        # Odd indices: 0, 2, 4, ...
        # Even indices: 1, 3, 5, ...
        # Note: Standard Odd-Merge splits by index parity.
        # Let's use the recursive definition:
        # Sort odd indices (0, 2, ...) recursively
        # Sort even indices (1, 3, ...) recursively
        # Merge them
        
        # The standard recursive formula for Odd-Merge:
        # 1. Split into two halves: A (first n/2) and B (second n/2)? 
        # No, Odd-Merge splits by parity of the index in the current list.
        # Let's implement the standard Batcher Odd-Merge.
        
        # Actually, a simpler recursive formulation often used in code:
        # 1. Sort the odd-indexed elements (indices 0, 2, 4...)
        # 2. Sort the even-indexed elements (indices 1, 3, 5...)
        # 3. Merge the two sorted sub-sequences.
        
        # Let's implement a helper that generates comparators for a list of wires
        # assuming they are to be sorted.
        
        pass

    # A more direct implementation of Batcher's Odd-Merge for N=16
    # N=16
    # Step 1: Sort odd indices (0, 2, 4, 6, 8, 10, 12, 14) and even indices (1, 3, 5, 7, 9, 11, 13, 15)
    # This is equivalent to sorting two independent 8-wire networks on the remapped indices.
    # Then merge.
    
    # Let's use a known hardcoded list or generate it properly.
    # Generating it properly is safer to avoid hardcoding errors.
    
    def batcher_odd_merge(n):
        """
        Generates comparators for Batcher's Odd-Merge network on n wires.
        Returns a list of (i, j) where i < j.
        """
        if n == 1:
            return []
        
        # Split indices into odd and even positions
        # Indices: 0, 1, ..., n-1
        # Odd positions (0-based): 0, 2, 4, ...
        # Even positions (0-based): 1, 3, 5, ...
        
        # Map original indices to sub-indices for recursion
        # Sub-network 1: indices 0, 2, 4, ... (size n//2)
        # Sub-network 2: indices 1, 3, 5, ... (size n//2)
        
        comps = []
        
        # Recursively sort the odd-indexed wires (0, 2, ...)
        # These are mapped to 0..n//2-1
        # We need to translate the sub-network comparators back to original indices
        for (i, j) in batcher_odd_merge(n // 2):
            # Map sub-index k to original index 2*k
            comps.append((2 * i, 2 * j))
            
        # Recursively sort the even-indexed wires (1, 3, ...)
        for (i, j) in batcher_odd_merge(n // 2):
            # Map sub-index k to original index 2*k + 1
            comps.append((2 * i + 1, 2 * j + 1))
            
        # Merge the two sorted halves
        # The merge step for Odd-Merge involves comparators between the two sub-seqs.
        # Let A be the sorted odd-indexed wires (indices 0, 2, ...)
        # Let B be the sorted even-indexed wires (indices 1, 3, ...)
        # 
        # The merge pattern for Odd-Merge:
        # We compare A[i] with B[i] and A[i] with B[i+1] (with appropriate bounds).
        # Specifically, for i from 0 to n/2 - 1:
        #   Compare (2*i, 2*i + 1)  -> A[i] with B[i]
        #   If i > 0, Compare (2*i - 1, 2*i) -> B[i-1] with A[i] ? 
        #   
        # Let's check the standard Odd-Merge merge step.
        # After sorting A (indices 0,2,..) and B (indices 1,3,..):
        # For k in 0..n/2-1:
        #   Comparator (2*k, 2*k+1)  # A[k] vs B[k]
        #   If k < n/2 - 1:
        #       Comparator (2*k+1, 2*k+2) # B[k] vs A[k+1]
        #
        # Wait, let's verify with N=4.
        # A: [0, 2], B: [1, 3]
        # Sort A: (0,2)
        # Sort B: (1,3)
        # Merge:
        # k=0: Compare (0,1). Then Compare (1,2).
        # k=1: Compare (2,3).
        # Total: (0,2), (1,3), (0,1), (1,2), (2,3).
        # Is this a valid sort for N=4?
        # Let's trace 0,1,2,3 (sorted).
        # (0,2): 0,1,2,3 -> 0,1,2,3
        # (1,3): 0,1,2,3 -> 0,1,2,3
        # (0,1): 0,1,2,3 -> 0,1,2,3
        # (1,2): 0,1,2,3 -> 0,1,2,3
        # (2,3): 0,1,2,3 -> 0,1,2,3
        #
        # Let's trace 3,2,1,0.
        # (0,2): 2,2,1,3 (min(3,2)=2, max=3) -> wire0=2, wire2=3. Seq: 2,2,3,0? No.
        # Input: [3, 2, 1, 0]
        # (0,2): min(3,1)=1, max(3,1)=3. Wires: [1, 2, 3, 0]
        # (1,3): min(2,0)=0, max(2,0)=2. Wires: [1, 0, 3, 2]
        # (0,1): min(1,0)=0, max(1,0)=1. Wires: [0, 1, 3, 2]
        # (1,2): min(1,3)=1, max(1,3)=3. Wires: [0, 1, 3, 2]
        # (2,3): min(3,2)=2, max(3,2)=3. Wires: [0, 1, 2, 3]
        # It works for N=4.
        
        # So the merge step is:
        for k in range(n // 2):
            # Compare A[k] (index 2*k) with B[k] (index 2*k+1)
            comps.append((2 * k, 2 * k + 1))
            # Compare B[k] (index 2*k+1) with A[k+1] (index 2*k+2)
            if k < n // 2 - 1:
                comps.append((2 * k + 1, 2 * k + 2))
                
        return comps

    return batcher_odd_merge(N)
# EVOLVE-BLOCK-END
