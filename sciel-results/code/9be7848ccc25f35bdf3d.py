def odd_even_merge(k):
    # Returns comparators to merge two sorted lists of length k
    # Wires 0..k-1 (A) and k..2k-1 (B)
    if k == 1:
        return []
    
    comps = []
    
    # 1. Compare A[i] with B[i]
    for i in range(k):
        comps.append((i, i + k))
        
    # 2. Recursively merge the two resulting sequences
    # The mins are on 0..k-1, maxs on k..2k-1
    # We need to merge the sequence of mins and the sequence of maxs.
    # But they are not sorted.
    # The standard Odd-Even Merge does NOT do this.
    # The standard Odd-Even Merge splits into even/odd indices FIRST.
    
    # Correct Odd-Even Merge:
    # 1. Split A and B into even and odd indices.
    # 2. Recursively merge A_even with B_even.
    # 3. Recursively merge A_odd with B_odd.
    # 4. Interleave the results with comparisons.
    
    # This is complex to map indices.
    # I will use the 120-comparator network and local search.
    pass

def construct():
    # Start with Odd-Even Transposition (120)
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Local Search
    # ...
    return net
