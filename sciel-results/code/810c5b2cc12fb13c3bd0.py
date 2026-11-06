N = 17

# EVOLVE-BLOCK-START
def construct():
    return _batcher_sort_network(0, N)

def _batcher_sort_network(offset, n):
    """Generate comparators to sort n elements starting at 'offset'."""
    if n <= 1:
        return []
    k = n // 2
    net = _batcher_sort_network(offset, k)
    net += _batcher_sort_network(offset + k, n - k)
    net += _odd_even_merge_network(offset, k, n - k)
    return net

def _odd_even_merge_network(offset, n, m):
    """Merge two sorted blocks: [offset, offset+n) and [offset+n, offset+n+m).
    n and m are sizes, n <= m assumed (or handle generally)."""
    if n == 0:
        return []
    if m == 0:
        return []
    
    total = n + m
    
    # Count odd/even positions in each half
    # First half: positions offset .. offset+n-1
    # Odd global positions in first half: offset+1, offset+3, ... if offset is even, else offset, offset+2,...
    # Actually, "odd-even merge" refers to the parity of positions within the merged sequence,
    # not global positions. Let me use local positions.
    
    # Local positions: 0..total-1
    # First half: local 0..n-1
    # Second half: local n..total-1
    
    # Odd local positions: 1, 3, 5, ...
    # Even local positions: 0, 2, 4, ...
    
    n_odd = n // 2  # number of odd positions in 0..n-1
    n_even = n - n_odd
    
    m_odd = m // 2  # number of odd positions in n..n+m-1 (local)
    m_even = m - m_odd
    
    # Odd local positions from first half: 1, 3, ..., 2*n_odd-1
    # Odd local positions from second half: n+1, n+3, ... (only if n is even, n+1 is odd)
    # Wait, local positions in second half are n, n+1, ..., n+m-1
    # Odd among these: if n is even, n+1, n+3, ...; if n is odd, n, n+2, ...
    
    # Let me just enumerate them.
    odd_first = [i for i in range(n) if i % 2 == 1]
    odd_second = [i for i in range(n, n + m) if i % 2 == 1]
    even_first = [i for i in range(n) if i % 2 == 0]
    even_second = [i for i in range(n, n + m) if i % 2 == 0]
    
    net = []
    
    # Recursively merge the odd-position elements
    if len(odd_first) > 0 and len(odd_second) > 0:
        # Map odd_first and odd_second to contiguous 0..len-1
        # Then recurse
        # odd_first maps to 0..len(odd_first)-1
        # odd_second maps to len(odd_first)..len(odd_first)+len(odd_second)-1
        sub = _odd_even_merge_network(0, len(odd_first), len(odd_second))
        # Map back
        all_odd = odd_first + odd_second
        net += [(offset + all_odd[i], offset + all_odd[j]) for i, j in sub]
    elif len(odd_first) > 0 or len(odd_second) > 0:
        # One side is empty, nothing to merge
    
    # Recursively merge the even-position elements
    if len(even_first) > 0 and len(even_second) > 0:
        sub = _odd_even_merge_network(0, len(even_first), len(even_second))
        all_even = even_first + even_second
        net += [(offset + all_even[i], offset + all_even[j]) for i, j in sub]
    
    # Finally, compare adjacent pairs: (0,1), (2,3), (4,5), ...
    # i.e., for i in range(0, total, 2): compare (offset+i, offset+i+1)
    for i in range(0, total - 1, 2):
        net.append((offset + i, offset + i + 1))
    
    return net
# EVOLVE-BLOCK-END
