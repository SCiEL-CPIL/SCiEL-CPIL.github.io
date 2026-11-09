import time
import numpy as np

N = 18
ALL_COUNT = 1 << N
DEADLINE = 85.0  # Safety margin under 90s limit

def _init_test_vectors():
    """Create the array of all 2^N boolean inputs, shape (2^N, N)."""
    idx = np.arange(ALL_COUNT, dtype=np.uint32)
    # We want column j to be the j-th bit of idx.
    # idx shape is (ALL_COUNT,)
    # We need to expand it to (ALL_COUNT, N)
    # Using broadcasting:
    # bits = (idx[:, None] >> np.arange(N)) & 1
    bits = (idx[:, None] >> np.arange(N, dtype=np.uint32)) & 1
    return bits.astype(np.uint8)

def _is_sorted(net, test_vectors):
    """
    Check if the network sorts all boolean inputs.
    net: list of (i, j)
    test_vectors: np.array of shape (ALL_COUNT, N)
    """
    if not net:
        # Empty network is only valid if N <= 1
        return N <= 1
        
    vals = test_vectors.copy()
    for a, b in net:
        va = vals[:, a]
        vb = vals[:, b]
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        vals[:, a] = lo
        vals[:, b] = hi
        
    # Check if sorted: vals[:, i] <= vals[:, i+1] for all i
    # This is equivalent to checking if the array is non-decreasing along axis 1
    # We can do: np.all(vals[:, :-1] <= vals[:, 1:])
    return np.all(vals[:, :-1] <= vals[:, 1:])

def construct():
    start_time = time.time()
    
    # 1. Initialize test vectors
    test_vectors = _init_test_vectors()
    
    # 2. Generate initial network: Odd-Even Transposition
    # Phases 0 to N-1
    # Phase p: comparators (i, i+1) where i % 2 == p % 2
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
            
    # Verify initial network
    if not _is_sorted(net, test_vectors):
        # Should not happen, but fallback
        return net

    current_len = len(net)
    
    # 3. Greedy Removal
    # We try to remove comparators one by one.
    # We do multiple passes until no more can be removed or time runs out.
    
    # To speed up, we can randomize the order or just go backwards.
    # Going backwards is often effective because later comparators might be redundant 
    # if earlier ones already sorted the wires sufficiently.
    
    max_passes = 100
    for _pass in range(max_passes):
        if time.time() > DEADLINE:
            break
            
        changed = False
        # Iterate in reverse to allow removal of later elements without shifting indices too much
        # Although we rebuild the list, so indices don't matter for the logic, 
        # but reversing might find redundancies introduced by previous removals?
        # Actually, just iterating 0 to len-1 is fine.
        
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > DEADLINE:
                break
                
            # Try removing net[idx]
            candidate_net = net[:idx] + net[idx + 1:]
            
            if _is_sorted(candidate_net, test_vectors):
                net = candidate_net
                changed = True
                # We don't break because removing one might allow removing another
                # But we continue to the next index (which is now idx-1)
                
        if not changed:
            break
            
    # Final verification
    if _is_sorted(net, test_vectors):
        return net
    else:
        # Fallback to original OET if something went wrong
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
