import time

N = 16
TOTAL_INPUTS = 1 << N

# Precompute initialization masks and target masks
# M[w] represents the set of inputs (bitmask) for which wire w is 1
INIT_M = [0] * N
for w in range(N):
    mask = 0
    for k in range(TOTAL_INPUTS):
        if (k >> w) & 1:
            mask |= (1 << k)
    INIT_M[w] = mask

# TARGET[w] represents the set of inputs for which wire w is 1 in a sorted output
# In a sorted output, wire w is 1 iff the count of 1s in the input is > N - 1 - w
TARGET = [0] * N
for k in range(TOTAL_INPUTS):
    pc = k.bit_count()
    for w in range(N):
        if pc > N - 1 - w:
            TARGET[w] |= (1 << k)

def is_valid(network):
    """
    Check if the network sorts all 2^16 inputs using bitmask simulation.
    """
    # Copy initial state
    M = INIT_M[:]
    
    # Apply comparators
    for i, j in network:
        min_val = M[i] & M[j]
        max_val = M[i] | M[j]
        M[i] = min_val
        M[j] = max_val
    
    # Check against target
    for w in range(N):
        if M[w] != TARGET[w]:
            return False
    return True

def batcher_oes(n, offset):
    """
    Generate Batcher's Odd-Even Mergesort network for 'n' wires starting at 'offset'.
    """
    if n == 1:
        return []
    
    k = n // 2
    # Indices for odd positions (0, 2, ...) relative to offset
    odds = [offset + 2 * i for i in range(k)]
    # Indices for even positions (1, 3, ...) relative to offset
    evens = [offset + 2 * i + 1 for i in range(k)]
    
    def merge(a, b):
        """
        Batcher's merge of two sorted lists a and b of equal size.
        """
        if len(a) == 1:
            return [(a[0], b[0])]
        
        k = len(a) // 2
        a1, a2 = a[:k], a[k:]
        b1, b2 = b[:k], b[k:]
        
        comps = []
        comps += merge(a1, b1)
        comps += merge(a2, b2)
        
        # The cross comparisons
        for i in range(k):
            comps.append((a1[i], b2[k - 1 - i]))
            comps.append((a2[k - 1 - i], b1[i]))
            
        return comps
    
    return merge(odds, evens)

def construct():
    start_time = time.time()
    timeout = 60.0
    
    # Generate initial network
    network = batcher_oes(N, 0)
    
    # Prune the network
    # We iterate until no more comparators can be removed
    # To ensure we don't time out, we check the deadline
    # The pruning loop is typically very fast for N=16
    
    improved = True
    while improved:
        improved = False
        # Iterate from end to start (often more effective for removing redundant tail comparators)
        for idx in range(len(network) - 1, -1, -1):
            # Check deadline
            if time.time() - start_time > timeout - 5.0:
                break
            
            # Test removal
            test_net = network[:idx] + network[idx + 1:]
            if is_valid(test_net):
                network = test_net
                improved = True
                break # Restart the scan from the end
        
        if time.time() - start_time > timeout - 5.0:
            break

    return network
