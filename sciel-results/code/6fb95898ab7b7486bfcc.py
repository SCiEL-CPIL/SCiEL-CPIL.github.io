"""Size-efficient sorting networks on 16 wires."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for 16 wires using a Batcher's Bitonic Sort network.
    
    The bitonic sort network for N=16 has:
    - 15 + 4*5 + 2*10 + 15 = 15 + 20 + 20 + 15 = 70 comparators? Let me recalculate.
    
    Actually, let's use a known optimal or near-optimal network.
    
    The known best known sorting network for 16 inputs has 57 comparators (per literature).
    However, constructing that from scratch is complex.
    
    Let's use Batcher's bitonic sort which is well-defined:
    For N=16, bitonic sort has depth 4*log2(N)/2 ... 
    
    Actually, let me just implement bitonic sort properly.
    
    Bitonic sort for N=2^k:
    - It creates bitonic sequences and then merges them.
    - Number of comparators for N=16: 
      * Stage 1: merge size 2 -> 4 stages of pairs... 
    
    Let me use the standard recursive bitonic sort construction:
    
    bitonic(N):
      if N == 2: return [(0,1)]
      left = bitonic(N//2)
      right = [(i + N//2, j + N//2) for (i,j) in left]
      merge = bitonic_merge(N)
      return left + right + merge
    
    bitonic_merge(N):
      if N == 2: return [(0,1)]
      left = bitonic_merge(N//2)
      right = [(i + N//2, j + N//2) for (i,j) in left]
      # Now cross-compare: for i in 0..N/2-1, compare (i, i+N/2)
      cross = [(i, i + N//2) for i in range(N//2)]
      return left + right + cross
    
    Let's compute the count for N=16:
    bitonic_merge(2) = 1 comparator
    bitonic_merge(4) = bitonic_merge(2)*2 + 2 = 1*2 + 2 = 4
    bitonic_merge(8) = bitonic_merge(4)*2 + 4 = 4*2 + 4 = 12
    bitonic_merge(16) = bitonic_merge(8)*2 + 8 = 12*2 + 8 = 32
    
    bitonic(2) = 1
    bitonic(4) = bitonic(2)*2 + bitonic_merge(4) = 1*2 + 4 = 6
    bitonic(8) = bitonic(4)*2 + bitonic_merge(8) = 6*2 + 12 = 24
    bitonic(16) = bitonic(8)*2 + bitonic_merge(16) = 24*2 + 32 = 80
    
    So bitonic sort uses 80 comparators for 16 wires. That's better than 120.
    """
    
    def bitonic_merge(n):
        """Generate comparators for bitonic merge of n elements (assumes n is power of 2)."""
        if n == 2:
            return [(0, 1)]
        half = n // 2
        left = bitonic_merge(half)
        right = [(i + half, j + half) for (i, j) in left]
        cross = [(i, i + half) for i in range(half)]
        return left + right + cross
    
    def bitonic_sort(n):
        """Generate comparators for bitonic sort of n elements."""
        if n == 2:
            return [(0, 1)]
        half = n // 2
        left = bitonic_sort(half)
        right = [(i + half, j + half) for (i, j) in left]
        merge = bitonic_merge(n)
        return left + right + merge
    
    return bitonic_sort(N)
# EVOLVE-BLOCK-END
