import numpy as np
import time

N = 18

def construct():
    start_time = time.time()
    deadline = 55.0  # seconds

    # Generate seed network: odd-even transposition sort
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    # Pre-compute all 2^18 Boolean inputs as bit-parallel representation
    # Each input is an integer 0..2^18-1, where bit k represents wire k
    # We'll process them in chunks for memory efficiency
    num_inputs = 1 << N

    # Precompute the initial state for all inputs
    # Use uint32 to hold 18 bits
    all_inputs = np.arange(num_inputs, dtype=np.uint32)

    def verify_bitparallel(net, inputs_chunk):
        """Verify network on a chunk of inputs using bit-parallel operations.
        
        inputs_chunk: array of shape (M,) where each element is an 18-bit integer
                      representing the initial state of 18 wires
        Returns True if net sorts all inputs in the chunk.
        """
        # state is an array of M 18-bit integers
        # bit k of state[m] represents wire k for input m
        state = inputs_chunk.copy()
        
        for i, j in net:
            # Extract bits i and j
            mask_i = 1 << i
            mask_j = 1 << j
            
            # For each input, get the values at wires i and j
            bit_i = (state & mask_i) != 0  # boolean array
            bit_j = (state & mask_j) != 0  # boolean array
            
            # min goes to wire i, max goes to wire j
            # min: 0 if either is 0, 1 if both are 1
            min_bit = bit_i & bit_j
            # max: 1 if either is 1, 0 if both are 0
            max_bit = bit_i | bit_j
            
            # Update state: clear bits i and j, then set them
            state &= ~(mask_i | mask_j)
            state = np.where(min_bit, state | mask_i, state)
            state = np.where(max_bit, state | mask_j, state)
        
        # Check if sorted: for all consecutive pairs (k, k+1), bit k <= bit k+1
        # This means: if bit k is 1, then bit k+1 must also be 1
        # Violation: bit k = 1 AND bit k+1 = 0
        for k in range(N - 1):
            mask_k = 1 << k
            mask_k1 = 1 << (k + 1)
            
            bit_k = (state & mask_k) != 0
            bit_k1 = (state & mask_k1) != 0
            
            # Check for violations: bit_k=1 and bit_k1=0
            if np.any(bit_k & ~bit_k1):
                return False
        
        return True

    # Pruning loop with bit-parallel verification
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to remove each comparator
        # Iterate in reverse to avoid index issues
        for idx_c in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break

            # Remove comparator
            comparator = network.pop(idx_c)
            
            # Verify using bit-parallel approach
            # Process in chunks to manage memory
            chunk_size = 65536
            valid = True
            for start in range(0, num_inputs, chunk_size):
                end = min(start + chunk_size, num_inputs)
                chunk = all_inputs[start:end]
                if not verify_bitparallel(network, chunk):
                    valid = False
                    break
            
            if valid:
                changed = True
            else:
                # Put it back
                network.insert(idx_c, comparator)

    return network
