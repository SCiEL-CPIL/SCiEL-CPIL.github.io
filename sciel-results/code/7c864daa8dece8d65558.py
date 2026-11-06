import time

N = 18

def _batcher_odd_merge(n):
    """Generates a Batcher's Odd-Merge sorting network for n inputs."""
    if n <= 1:
        return []
    
    # Split into n1 and n2
    n1 = (n + 1) // 2
    n2 = n // 2
    
    # Recursively sort the two halves
    net1 = _batcher_odd_merge(n1)
    net2 = _batcher_odd_merge(n2)
    
    # Offset net2 by n1
    net2 = [(i + n1, j + n1) for i, j in net2]
    
    # Merge the two sorted lists
    merge_net = _batcher_merge(n1, n2)
    
    return net1 + net2 + merge_net

def _batcher_bitonic(n):
    """Generates a Batcher's Bitonic sorting network for n inputs."""
    if n <= 1:
        return []
    
    # Find the largest power of 2 less than or equal to n
    # If n is not a power of 2, pad with dummy wires? 
    # Bitonic sort works best for powers of 2.
    # For non-powers of 2, we can use a hybrid or just use Odd-Merge.
    # For N=8, it is a power of 2.
    if n & (n - 1) != 0:
        # Fallback to Odd-Merge if not power of 2
        return _batcher_odd_merge(n)
        
    if n == 1:
        return []
    
    # Recursively sort two halves
    half = n // 2
    net1 = _batcher_bitonic(half)
    net2 = _batcher_bitonic(half)
    net2 = [(i + half, j + half) for i, j in net2]
    
    # Merge the two halves (Bitonic Merge)
    merge_net = _batcher_merge_bitonic(n)
    
    return net1 + net2 + merge_net

def _batcher_merge(n1, n2):
    """Generates a Batcher's Merge network for two sorted lists of size n1 and n2."""
    n = n1 + n2
    if n <= 1:
        return []
    
    # Determine the number of stages
    import math
    stages = int(math.log2(n))
    
    net = []
    for k in range(1, stages + 1):
        # The comparators in stage k are (i, i + 2^k) for appropriate i
        # Specifically, for Batcher's merge, the pattern is:
        # For k from 1 to log2(n):
        #   for i from 0 to n - 2^k - 1:
        #     if (i % 2^k) < 2^(k-1):
        #        compare (i, i + 2^k)
        
        step = 1 << k
        half_step = 1 << (k - 1)
        
        for i in range(n):
            if i + step < n:
                if (i % step) < half_step:
                    net.append((i, i + step))
                    
    return net

def _batcher_merge_bitonic(n):
    """Generates a Bitonic Merge network for n inputs (n is power of 2)."""
    # This is similar to _batcher_merge but specifically for bitonic sequences
    # Actually, the standard bitonic merge is:
    # for k from log2(n) down to 1:
    #   for i from 0 to n - 2^k - 1:
    #     compare (i, i + 2^k)
    
    import math
    log_n = int(math.log2(n))
    
    net = []
    for k in range(log_n, 0, -1):
        step = 1 << k
        for i in range(n - step + 1):
            net.append((i, i + step))
            
    return net

def construct():
    # Time guard
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Generate 10-input network (Odd-Merge)
    net_10 = _batcher_odd_merge(10)
    
    # Generate 8-input network (Bitonic)
    net_8 = _batcher_bitonic(8)
    
    # Offset net_8 by 10
    net_8_offset = [(i + 10, j + 10) for i, j in net_8]
    
    # Generate 10-8 merge network
    # We need a merge network that merges two sorted lists of size 10 and 8.
    # Batcher's merge works for any n1, n2.
    merge_net = _batcher_merge(10, 8)
    
    # Combine
    full_net = net_10 + net_8_offset + merge_net
    
    # Check time
    if time.time() > deadline:
        # Fallback to seed if too slow (unlikely)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
                
    return full_net
