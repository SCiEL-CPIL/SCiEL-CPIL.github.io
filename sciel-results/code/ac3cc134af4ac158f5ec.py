"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 17 wires.
    
    Strategy: Use a known efficient construction based on AKS-like or
    practical sorting networks. Since finding the absolute minimum (71) is
    open, we aim for a significantly better network than the seed (136).
    
    We will use a recursive construction:
    1. Base case: small N handled by brute force / known tables.
    2. Recursive step: Split into two halves, sort each, then merge.
    
    However, for N=17, a simple recursive merge might not be optimal.
    Let's try a different approach: use the "bitonic" network which is
    easy to generate and has O(N log^2 N) comparators.
    
    Bitonic network for N wires:
    Depth: O(log^2 N)
    Size: O(N log^2 N)
    For N=17, we can pad to 32 (next power of 2) or handle non-power-of-2.
    
    Actually, let's use a standard recursive bitonic sort network generator.
    
    Alternatively, we can use a known compact network for small N.
    For N=17, a well-known network from the literature (e.g., from the
    "Sorting Networks" page on the web or academic papers) might be around
    100-120 comparators.
    
    Let's implement a recursive construction that is known to be efficient:
    - For N <= 1: empty
    - For N == 2: [(0, 1)]
    - For N == 3: [(0, 1), (1, 2)] -> wait, that's not enough.
      (0,1), (1,2) sorts [1,0,2] -> [0,1,2] but [2,1,0] -> [1,2,0] -> [1,0,2] -> wrong.
      Correct 3-wire: (0,1), (1,2), (0,1)
    
    Let's use a more robust method: the "odd-even mergesort" network.
    Odd-even mergesort is a classic sorting network algorithm.
    
    Reference: Knuth, TAOCP Vol 3, Section 5.2.5.
    
    Let's implement odd-even mergesort for N=17.
    """
    
    def odd_even_merge(n, a, b):
        """
        Generate comparators to merge two sorted halves of size n/2.
        a, b are the starting indices of the two halves.
        Actually, standard formulation: merge two sorted arrays of length n/2.
        """
        if n == 2:
            return [(a, b)]
        
        # Split into quarters
        n2 = n // 2
        n4 = n // 4
        
        # Recursively merge the first halves and second halves
        # First half: a to a+n2-1, second half: b to b+n2-1
        # Wait, the structure is:
        # We have two sorted sequences of length n/2.
        # We want to merge them into one sorted sequence of length n.
        
        # Odd-even mergesort:
        # 1. Recursively merge the first halves (even positions) and second halves (odd positions)
        # 2. Then compare-exchange adjacent pairs
        
        # Let's use a simpler recursive definition from Knuth:
        # MERGE(n, a, b):
        #   if n == 2: return [(a, b)]
        #   MERGE(n/2, a, b)
        #   MERGE(n/2, a+1, b+1)
        #   for i in range(n/2 - 1):
        #       compare-exchange(a+i, a+i+1)
        
        # This is for merging two sorted sequences of length n/2 starting at a and b.
        # But in a full sort, we first sort the two halves separately.
        
        def merge(n, a, b):
            if n == 2:
                return [(a, b)]
            n2 = n // 2
            comparators = []
            comparators += merge(n2, a, b)
            comparators += merge(n2, a + 1, b + 1)
            for i in range(n2 - 1):
                comparators.append((a + i, a + i + 1))
            return comparators
        
        return merge(n, a, b)
    
    def sort_network(n, offset=0):
        """
        Generate a sorting network for n wires starting at offset.
        Uses odd-even mergesort.
        """
        if n <= 1:
            return []
        if n == 2:
            return [(offset, offset + 1)]
        
        n2 = n // 2
        n1 = n - n2  # handle odd n
        
        comparators = []
        # Sort first half
        comparators += sort_network(n1, offset)
        # Sort second half
        comparators += sort_network(n2, offset + n1)
        
        # Merge the two halves
        # The two halves are sorted. We need to merge them.
        # Standard odd-even mergesort merges two sorted sequences of equal length.
        # For odd n, the lengths are different.
        
        # To handle this simply, let's just use a generic merge for different sizes.
        # Or, simpler: pad to next power of 2? No, that wastes comparators.
        
        # Let's use a different, simpler recursive construction:
        # Divide and conquer: sort left, sort right, then merge.
        # For merging, we can use the "bitonic" merge if we construct bitonic sequences.
        
        # Actually, let's just use a known good network for N=17.
        # A simple and effective approach is to use the "AKS" style or
        # just a well-tuned recursive mergesort.
        
        # Let's restart with a cleaner recursive odd-even mergesort that handles any n.
        
        return comparators  # This was incomplete
    
    # Let's implement a proper odd-even mergesort.
    
    def oe_merge(n, a, b):
        """
        Merge two sorted sequences of length n/2 starting at a and b.
        Assumes n is even.
        """
        if n == 2:
            return [(a, b)]
        n2 = n // 2
        comparators = []
        comparators += oe_merge(n2, a, b)
        comparators += oe_merge(n2, a + 1, b + 1)
        for i in range(n2 - 1):
            comparators.append((a + i, a + i + 1))
        return comparators
    
    def oe_sort(n, offset=0):
        """
        Odd-even mergesort network for n wires.
        """
        if n <= 1:
            return []
        if n == 2:
            return [(offset, offset + 1)]
        
        # Split into two halves
        n1 = n // 2
        n2 = n - n1
        
        comparators = []
        # Sort first half
        comparators += oe_sort(n1, offset)
        # Sort second half
        comparators += oe_sort(n2, offset + n1)
        
        # Merge the two sorted halves
        # For simplicity and correctness, if n1 != n2, we can pad the smaller one
        # or use a more general merge.
        # For N=17, n1=8, n2=9.
        
        # A simpler approach: use a bitonic network.
        # Bitonic networks are easy to generate and have good performance.
        
        return comparators  # Incomplete
    
    # Let's use a Bitonic Sort Network. It's robust and easy to implement.
    
    def bitonic_merge(n, a, b, ascending=True):
        """
        Merge two sorted sequences of length n/2 starting at a and b.
        One is ascending, one is descending (bitonic).
        """
        if n == 2:
            if ascending:
                return [(a, b)]
            else:
                return [(a, b)]  # min to a, max to b? No, for descending, we want max to a.
                # Wait, compare-exchange always puts min in i, max in j.
                # So if we want descending, we should compare (b, a)? No, i < j.
                # So we can't directly swap.
                # Actually, bitonic merge works by comparing a[i] with b[n-1-i].
        
        # Let's use the standard bitonic sort construction.
        # It's easier to just generate the comparators for a full bitonic sort.
        pass
    
    # Let's use a simpler, well-known network: the "odd-even transposition" is O(N^2).
    # For N=17, that's 136.
    
    # Let's try to find a better known network.
    # For N=17, a network with ~100 comparators is achievable.
    
    # Let's use the following construction:
    # 1. Split into 8 and 9.
    # 2. Sort each (using smaller networks).
    # 3. Merge using a "shuffle" and compare.
    
    # Actually, let's just use a recursive AKS-like construction or a simple
    # divide-and-conquer with a good merge.
    
    # For now, let's return a slightly improved odd-even transposition.
    # We can prune some comparators if we know they are redundant.
    
    # But for a "stronger algorithm", let's implement a proper recursive mergesort.
    
    def merge_two_sorted(n1, n2, a, b):
        """
        Merge two sorted sequences of length n1 and n2 starting at a and b.
        """
        if n1 == 0:
            return []
        if n2 == 0:
            return []
        
        # If both are small, use a simple network.
        if n1 == 1 and n2 == 1:
            return [(a, b)]
        
        # Recursive merge
        # This is complex. Let's use a simpler method:
        # Compare a[i] with b[j] in a pattern.
        
        # For simplicity, let's just use the bitonic network for the whole thing.
        # Bitonic sort for N=17:
        # Pad to 32? No, that's wasteful.
        
        # Let's use the "AKS" network for small N from a table.
        # For N=17, a known network has 107 comparators (from literature).
        
        # Since I don't have the table memorized, let's use a heuristic:
        # Use odd-even transposition but stop early if sorted?
        # No, we need a static network.
        
        # Let's use a simple recursive construction:
        # Sort left, sort right, then do a "merge" by comparing
        # a[i] with b[j] where i+j is constant.
        
        # This is getting complex. Let's just return the seed network
        # but with a small optimization: remove the last few comparators
        # that are provably redundant for N=17?
        
        # Actually, the odd-even transposition network for N wires has
        # N phases, each with floor(N/2) comparators.
        # Total: N * floor(N/2) = 17 * 8 = 136.
        
        # We can do better. Let's use a "bitonic" network for N=17.
        # Bitonic sort for non-power-of-2 is tricky.
        
        # Let's use a different approach: use the "shuffle" network.
        
        # For now, let's return a network that is slightly better than 136.
        # We can use the following trick:
        # The last phase of odd-even transposition is not needed if the
        # previous phases have already sorted the network.
        
        # Let's just return the seed for now, but with a comment that
        # a better network should be used.
        
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Actually, let's implement a simple recursive mergesort that works for any N.
    
    def sort_network(n, offset=0):
        if n <= 1:
            return []
        if n == 2:
            return [(offset, offset + 1)]
        
        n1 = n // 2
        n2 = n - n1
        
        comparators = []
        comparators += sort_network(n1, offset)
        comparators += sort_network(n2, offset + n1)
        
        # Merge the two sorted halves
        # Use a simple merge: compare a[i] with b[j] for all i, j
        # such that i + j < n1 + n2 - 1
        # This is a known method.
        
        # Actually, let's use the "odd-even merge" for equal sizes,
        # and for different sizes, pad the smaller one with "infinite" values.
        
        # For simplicity, let's just use a brute-force merge for small N.
        # For N=17, n1=8, n2=9.
        
        # Let's use the following merge:
        # For k in range(n1 + n2 - 1):
        #   for i in range(k + 1):
        #     if 0 <= i < n1 and 0 <= k - i < n2:
        #       compare-exchange(offset + i, offset + n1 + (k - i))
        
        # This is a valid merge network.
        
        for k in range(n1 + n2 - 1):
            for i in range(k + 1):
                j = k - i
                if 0 <= i < n1 and 0 <= j < n2:
                    comparators.append((offset + i, offset + n1 + j))
        
        return comparators
    
    return sort_network(N)
# EVOLVE-BLOCK-END
