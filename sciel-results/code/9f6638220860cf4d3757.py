import time
import numpy as np

N = 18
TIME_LIMIT = 55  # Safety margin for 90s limit (55s allows 35s for overhead/evaluation)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states for fast verification
    # Number of 64-bit blocks needed for 2^18 inputs
    num_blocks = 2**N // 64
    
    # states[w] is an array of num_blocks uint64s
    # Bit b of states[w][k] is the value of wire w for input (k*64 + b)
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Efficiently fill states using vectorized operations
    # Input index k ranges from 0 to 2^N - 1
    # We can generate the bit for wire w for all k at once, but storing 2^18 bits 
    # per wire is 2^18 / 8 = 32KB per wire, total 576KB. This is small.
    # However, we need to pack them into uint64 blocks.
    
    # Let's use a faster method:
    # For each block index b (0 to num_blocks-1), the inputs are k = b*64 + i (i=0..63)
    # We can precompute the pattern for one block of 64 inputs for each wire, 
    # but the pattern repeats every 2^N? No, the pattern for wire w depends on k.
    # k = b*64 + i.
    # Bit w of k is 1 if (b*64 + i) has bit w set.
    
    # Alternative: Just use the loop but optimize it?
    # The loop takes ~0.5s for 262k iterations. That's acceptable.
    # Let's stick to the loop but ensure it's as fast as possible.
    # Actually, we can precompute the mask for each wire for the 64 sub-blocks.
    # But the bit position in the 64-bit word changes with i.
    
    # Let's try a vectorized approach for initialization:
    # Create an array of all indices k
    # indices = np.arange(2**N, dtype=np.uint32)
    # For each wire w, we want a bitmask of 2^N bits.
    # We can reshape this into (num_blocks, 64) and then pack into uint64.
    
    indices = np.arange(2**N, dtype=np.uint32)
    
    for w in range(N):
        # Get bit w of each index
        bits = (indices >> w) & 1
        # Reshape to (num_blocks, 64)
        bits = bits.reshape(num_blocks, 64)
        # Pack into uint64
        # bit 0 of the uint64 corresponds to i=0, bit 1 to i=1, etc.
        # We need to sum bits * (1 << i)
        weights = 1 << np.arange(64, dtype=np.uint64)
        packed = (bits.T * weights).sum(axis=1)
        states[w] = packed.astype(np.uint64)

    def _fast_verify(comparators):
        """
        Verify network using bit-parallel simulation.
        Returns True if network sorts all inputs.
        """
        # Copy states
        # Using list comprehension for copy is fast
        current = [s.copy() for s in states]
        
        for i, j in comparators:
            a = current[i]
            b = current[j]
            # min goes to i, max goes to j
            current[i] = a & b
            current[j] = a | b
            
        # Check if sorted
        # For each adjacent pair w, w+1, check if any bit is 1 in w and 0 in w+1
        # i.e., current[w] & ~current[w+1] should be all zeros
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    # Verify seed
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net):
        changed = True
        # We iterate until no more changes or time is up
        while changed and time.time() < deadline - 5:
            changed = False
            idx = 0
            while idx < len(net):
                if time.time() > deadline - 5:
                    return net
                # Try removing net[idx]
                # Optimization: Check if net[idx] is redundant
                # We can do this by checking if the network without it is still valid.
                # To speed this up, we might be able to check local properties, 
                # but full verification is safer and fast enough with bit-parallel.
                
                # Copy and remove
                # Instead of slicing, we can use a list and pop, but slicing is okay for small lists
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                    # Do not increment idx, check the new element at idx
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    # Phase 2: Random Restart Local Search
    np.random.seed(42)
    
    # We will alternate between local search on the current best and random restarts
    # from the best network
    
    while time.time() < deadline - 10:
        if len(current_network) <= 1:
            break
            
        # Random Swap
        idx = np.random.randint(len(current_network) - 1)
        trial = current_network.copy()
        trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
        if _fast_verify(trial):
            current_network = trial
            current_network = greedy_reduce(current_network)
            if len(current_network) < best_len:
                best_network = list(current_network)
                best_len = len(current_network)
                
        # Wire Mutation
        if time.time() < deadline - 10 and len(current_network) > 0:
            idx = np.random.randint(len(current_network))
            i, j = current_network[idx]
            a, b = np.random.choice(N, 2, replace=False)
            if a > b:
                a, b = b, a
            if a == i and b == j:
                continue
            trial = current_network.copy()
            trial[idx] = (a, b)
            if _fast_verify(trial):
                current_network = trial
                current_network = greedy_reduce(current_network)
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    
        # Occasionally, restart from a perturbed best network
        # This helps escape local minima
        if np.random.random() < 0.1 and time.time() < deadline - 10:
             # Perturb best_network
             perturbed = list(best_network)
             # Add a random comparator
             a, b = np.random.choice(N, 2, replace=False)
             if a > b: a, b = b, a
             perturbed.append((a, b))
             # Or remove a random one?
             # Let's try adding and then reducing
             current_network = greedy_reduce(perturbed)
             if len(current_network) < best_len:
                 best_network = list(current_network)
                 best_len = len(current_network)

    return best_network
# EVOLVE-BLOCK-END
