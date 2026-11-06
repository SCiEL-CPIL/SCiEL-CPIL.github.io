def _batcher_network(n, offset=0):
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    
    half = n // 2
    network = []
    network.extend(_batcher_network(half, offset))
    network.extend(_batcher_network(half, offset + half))
    network.extend(_odd_even_merge(n, offset))
    return network

def _odd_even_merge(n, offset=0):
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    
    network = []
    # Phase 1
    for i in range(0, n - 1, 2):
        network.append((offset + i, offset + i + 1))
    # Phase 2
    for i in range(1, n - 1, 2):
        network.append((offset + i, offset + i + 1))
    
    # Recurse
    # Even subsequence: wires offset, offset+2, ...
    # Logical index k in subsequence maps to physical wire offset + 2*k
    even_net = _odd_even_merge(n // 2, 0)
    for i, j in even_net:
        network.append((offset + 2 * i, offset + 2 * j))
        
    # Odd subsequence: wires offset+1, offset+3, ...
    # Logical index k in subsequence maps to physical wire offset + 1 + 2*k
    odd_net = _odd_even_merge(n // 2, 0)
    for i, j in odd_net:
        network.append((offset + 1 + 2 * i, offset + 1 + 2 * j))
        
    return network
