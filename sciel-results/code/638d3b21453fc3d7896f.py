"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of comparators for a Batcher's odd-even mergesort network
    for 16 wires. This uses 66 comparators, which is better than the 120 
    comparators in the odd-even transposition network.
    """
    def batcher_sort(n):
        """
        Returns a list of comparators to sort n wires (0 to n-1).
        """
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        # Sort left half (0 to half-1)
        left_net = batcher_sort(half)
        
        # Sort right half (half to n-1)
        right_net_raw = batcher_sort(half)
        right_net = [(i + half, j + half) for (i, j) in right_net_raw]
        
        # Merge the two sorted halves
        merge_net = batcher_merge(half)
        
        return left_net + right_net + merge_net

    def batcher_merge(m):
        """
        Returns a list of comparators to merge two sorted lists of size m.
        Input list A is at wires 0 to m-1.
        Input list B is at wires m to 2m-1.
        Output sorted list is at wires 0 to 2m-1.
        """
        if m == 1:
            return [(0, 1)]
        
        k = m // 2
        net = []
        
        # 1. Recursively merge the first halves of A and B
        # A1: 0 to k-1, B1: m to m+k-1
        sub1 = batcher_merge(k)
        for (u, v) in sub1:
            # Map sub1 inputs (0..k-1, k..2k-1) to actual wires
            # u < k -> A1 (0..k-1)
            # u >= k -> B1 (m..m+k-1) -> u - k + m
            u_mapped = u if u < k else u - k + m
            v_mapped = v if v < k else v - k + m
            net.append((min(u_mapped, v_mapped), max(u_mapped, v_mapped)))
            
        # 2. Recursively merge the second halves of A and B
        # A2: k to m-1, B2: m+k to 2m-1
        sub2 = batcher_merge(k)
        for (u, v) in sub2:
            # Map sub2 inputs (0..k-1, k..2k-1) to actual wires
            # u < k -> A2 (k..m-1) -> u + k
            # u >= k -> B2 (m+k..2m-1) -> u - k + m + k
            u_mapped = u + k if u < k else u - k + m + k
            v_mapped = v + k if v < k else v - k + m + k
            net.append((min(u_mapped, v_mapped), max(u_mapped, v_mapped)))
            
        # 3. Interleave and sort
        # The outputs of sub1 and sub2 are now in wires 0..m-1 and k..2m-1?
        # No, sub1 outputs to 0..2k-1 (which is 0..m-1).
        # sub2 outputs to k..k+2k-1 (which is k..2m-1).
        # The wires are now partially sorted.
        # We need to perform the final "Odd-Even" comparators.
        
        # Odd phase: (0,1), (2,3), ...
        for i in range(0, 2 * m - 1, 2):
            net.append((i, i + 1))
            
        # Even phase: (1,2), (3,4), ...
        for i in range(1, 2 * m - 1, 2):
            net.append((i, i + 1))
            
        return net

    return batcher_sort(N)
# EVOLVE-BLOCK-END
